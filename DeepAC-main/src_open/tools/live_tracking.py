import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src_open.dataset.utils import numpy_image_to_torch, crop, resize, zero_pad
from src_open.models import get_model
from src_open.models.deep_ac import calculate_basic_line_data
from src_open.utils.geometry.wrappers import Camera, Pose
from src_open.utils.lightening_utils import (
    MyLightningLogger,
    convert_old_model,
    load_model_weight,
)
from src_open.utils.utils import (
    get_bbox_from_p2d,
    get_closest_template_view_index,
    get_closest_k_template_view_index,
    project_correspondences_line,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepAC live tracking runner")
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to live tracking yaml config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cuda:0 / cpu",
    )
    return parser.parse_args()


def load_model(cfg, device, logger):
    train_cfg = OmegaConf.load(cfg.model.load_cfg)
    model_cfg = train_cfg.models if "models" in train_cfg else train_cfg
    model = get_model(model_cfg.name)(model_cfg)

    ckpt = torch.load(cfg.model.load_model, map_location="cpu")
    if "pytorch-lightning_version" not in ckpt:
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    model.to(device).eval()

    logger.info(f"Loaded weights from {cfg.model.load_model}")
    return model, train_cfg


def load_template(pre_render_path, device):
    with open(pre_render_path, "rb") as f:
        pre = pickle.load(f)

    head = pre.get("head", {})
    num_sample = head.get("num_sample_contour_point") or head.get("num_sample")
    if num_sample is None:
        raise RuntimeError("pre_render pkl missing `num_sample_contour_point` in head")

    template_view = pre.get("template_view", None)
    if template_view is None:
        template_view = pre.get("template_views", None)
    if template_view is None:
        raise RuntimeError("pre_render pkl missing `template_view(s)` field")

    template_view = np.array(template_view)
    if template_view.ndim == 2:
        if template_view.shape[1] != 8:
            raise RuntimeError("Unexpected template_view shape; expected (*, 8)")
        if template_view.shape[0] % num_sample != 0:
            raise RuntimeError("template_view rows must be multiple of num_sample")
        num_views = template_view.shape[0] // num_sample
        template_view = template_view.reshape(num_views, num_sample, 8)

    orientations = pre.get("orientation_in_body", None)
    if orientations is None:
        orientations = pre.get("orientations_in_body", None)
    if orientations is None:
        raise RuntimeError("pre_render pkl missing `orientation_in_body` field")
    orientations = torch.from_numpy(np.array(orientations)).float()

    template_views = torch.from_numpy(template_view).float()
    return template_views.to(device), orientations.to(device), num_sample


def prepare_initial_pose(cfg, template_views, orientations, device):
    track_cfg = cfg.tracking
    num_views = template_views.shape[0]
    if num_views == 0:
        raise RuntimeError("Template views required for initial pose")

    view_idx = int(track_cfg.get("initial_view_index", 0))
    view_idx = max(0, min(view_idx, num_views - 1))

    translation = track_cfg.get("initial_translation")
    if translation is None:
        depth = float(track_cfg.init_depth)
        translation = [0.0, 0.0, depth]
    if len(translation) != 3:
        raise ValueError("tracking.initial_translation must be a list of length 3")

    translation_tensor = torch.tensor(translation, dtype=torch.float32, device=device).unsqueeze(0)
    orientation = orientations[view_idx : view_idx + 1]
    initial_pose = Pose.from_aa(orientation, translation_tensor)
    initial_template = template_views[view_idx : view_idx + 1]

    return view_idx, initial_pose, initial_template


def build_camera_tensor(camera_cfg, frame_shape, device):
    height, width = frame_shape[:2]
    fx = float(camera_cfg.fx)
    fy = float(camera_cfg.fy)
    cx = float(camera_cfg.cx)
    cy = float(camera_cfg.cy)

    # 与demo_cube.py一致：使用1D张量 [w, h, fx, fy, cx, cy]
    intrinsic_param = torch.tensor(
        [width, height, fx, fy, cx, cy],
        dtype=torch.float32,
        device=device,
    )
    return intrinsic_param


def preprocess_image(img, bbox2d, camera_cpu, data_conf, device):
    """与demo_cube.py完全相同的预处理流程"""
    # 确保bbox2d是numpy数组格式
    if isinstance(bbox2d, torch.Tensor):
        bbox2d = bbox2d.detach().cpu().numpy()
    elif not isinstance(bbox2d, np.ndarray):
        bbox2d = np.array(bbox2d)
    
    bbox2d = bbox2d.copy()  # 避免修改原始数组
    bbox2d[2:] += data_conf.crop_border * 2
    img_crop, camera_out, _ = crop(img, bbox2d, camera=camera_cpu, return_bbox=True)
    
    if img_crop.size == 0:  # Fallback if crop results in empty image
        img_crop = img
        camera_out = camera_cpu
    
    if isinstance(data_conf.resize, int) and data_conf.resize > 0:
        img_crop, scales = resize(img_crop, data_conf.resize, fn=max if data_conf.resize_by == "max" else min)
        camera_out = camera_out.scale(scales)
    
    if isinstance(data_conf.pad, int) and data_conf.pad > 0:
        img_crop, = zero_pad(data_conf.pad, img_crop)
    
    img_tensor = numpy_image_to_torch(img_crop.astype(np.float32))
    _, h_t, w_t = img_tensor.shape
    pad_h = (32 - h_t % 32) % 32
    pad_w = (32 - w_t % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))
        cam_data = camera_out._data.clone()
        cam_data[..., 0] += pad_w
        cam_data[..., 1] += pad_h
        camera_out = Camera(cam_data)
    return img_tensor.to(device), camera_out.to(device)


def preprocess_frame(frame_bgr, camera_cfg, device):
    """构建原始相机，不进行resize"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    intrinsic_param = build_camera_tensor(camera_cfg, frame_rgb.shape, device)
    camera = Camera(intrinsic_param)
    return frame_rgb, camera


def tensor_bbox_to_xyxy(bbox):
    cx, cy, w, h = bbox.tolist()
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))
    return x1, y1, x2, y2


def bbox_iou(bbox_a, bbox_b):
    if bbox_a is None or bbox_b is None:
        return None
    a = bbox_a.detach().cpu()
    b = bbox_b.detach().cpu()
    ax1, ay1, ax2, ay2 = tensor_bbox_to_xyxy(a)
    bx1, by1, bx2, by2 = tensor_bbox_to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@torch.no_grad()
def detect_once_with_preprocess(
    frame_rgb,
    ori_camera_cpu,
    initial_pose,
    template_views,
    orientations,
    model,
    device,
    detect_cfg,
    data_conf,
    bbox_trim_ratio=0.0,
):
    """带预处理的检测函数，与demo_cube.py一致"""
    ori_camera = ori_camera_cpu.to(device)
    
    # 使用初始位姿获取bbox进行预处理
    idx0 = get_closest_template_view_index(initial_pose, orientations)
    initial_template = template_views[idx0 : idx0 + 1]
    data_lines = project_correspondences_line(initial_template, initial_pose, ori_camera)
    bbox2d = get_bbox_from_p2d(data_lines["centers_in_image"][0], trim_ratio=bbox_trim_ratio)
    
    # 处理无效bbox
    h_img, w_img = frame_rgb.shape[:2]
    if bbox2d.numel() < 4 or torch.isnan(bbox2d).any() or bbox2d[2] < 4 or bbox2d[3] < 4:
        bbox2d = torch.tensor([w_img / 2.0, h_img / 2.0, float(w_img), float(h_img)], dtype=torch.float32, device=device)
    
    img_tensor, camera = preprocess_image(frame_rgb, bbox2d.cpu().numpy().copy(), ori_camera_cpu, data_conf, device)
    
    # 简化的检测：只检测初始位姿附近的几个视角
    num_views = template_views.shape[0]
    max_views = int(detect_cfg.get("max_detect_views", 240))
    max_views = num_views if max_views <= 0 else min(max_views, num_views)
    step = max(1, num_views // max_views)
    
    init_depth = float(detect_cfg.init_depth)
    best = {
        "score": -float("inf"),
        "idx": None,
        "pose": None,
        "bbox": None,
        "centers_in_image": None,
        "centers_valid": None,
    }
    
    cam_data = camera._data
    for idx in range(0, num_views, step):
        template_view = template_views[idx : idx + 1]
        aa = orientations[idx : idx + 1]
        pose = Pose.from_aa(aa, torch.tensor([[0.0, 0.0, init_depth]], device=device))
        
        (
            _,
            _,
            centers_in_image,
            centers_valid,
            normals_in_image,
            fg_dist,
            bg_dist,
            _,
        ) = calculate_basic_line_data(template_view, pose._data, cam_data, 1, 0)
        
        if not centers_valid.any():
            continue
        
        fore_hist, back_hist = model.histogram.calculate_histogram(
            img_tensor[None], centers_in_image, centers_valid, normals_in_image, fg_dist, bg_dist, True
        )
        score = torch.mean(fore_hist).item()
        if score > best["score"]:
            # 将centers_in_image转换回原始图像坐标系
            valid_mask = centers_valid[0]
            if valid_mask.any():
                # 使用原始相机投影到原始图像坐标系
                centers_ori, valid_ori = ori_camera.view2image(pose.transform(template_view[0, :, :3]))
                bbox = get_bbox_from_p2d(centers_ori[valid_ori], trim_ratio=bbox_trim_ratio)
            else:
                bbox = None
                centers_ori = None
                valid_ori = None
            best.update({
                "score": score,
                "idx": idx,
                "pose": pose,
                "bbox": bbox,
                "centers_in_image": centers_ori,
                "centers_valid": valid_ori.unsqueeze(0) if valid_ori is not None else None,
            })
    
    if best["idx"] is None:
        return None
    
    return best


@torch.no_grad()
def detect_once(
    image_tensor,
    camera,
    template_views,
    orientations,
    model,
    device,
    detect_cfg,
    bbox_trim_ratio=0.0,
):
    num_views = template_views.shape[0]
    if num_views == 0:
        raise RuntimeError("No template views available for detection")

    max_views = int(detect_cfg.max_detect_views)
    max_views = num_views if max_views <= 0 else min(max_views, num_views)
    step = max(1, num_views // max_views)

    init_depth = float(detect_cfg.init_depth)
    best = {
        "score": -float("inf"),
        "idx": None,
        "pose": None,
        "bbox": None,
        "centers_in_image": None,
        "centers_valid": None,
        "normals_in_image": None,
        "fg": None,
        "bg": None,
    }

    cam_data = camera._data
    for idx in range(0, num_views, step):
        template_view = template_views[idx : idx + 1]
        aa = orientations[idx : idx + 1]
        pose = Pose.from_aa(aa, torch.tensor([[0.0, 0.0, init_depth]], device=device))

        (
            _,
            _,
            centers_in_image,
            centers_valid,
            normals_in_image,
            fg_dist,
            bg_dist,
            _,
        ) = calculate_basic_line_data(template_view, pose._data, cam_data, 1, 0)

        if not centers_valid.any():
            continue

        fore_hist, back_hist = model.histogram.calculate_histogram(
            image_tensor, centers_in_image, centers_valid, normals_in_image, fg_dist, bg_dist, True
        )
        score = torch.mean(fore_hist).item()
        if score > best["score"]:
            valid_mask = centers_valid[0]
            if valid_mask.any():
                bbox = get_bbox_from_p2d(
                    centers_in_image[0][valid_mask], trim_ratio=bbox_trim_ratio
                )
            else:
                bbox = None
            best.update(
                {
                    "score": score,
                    "idx": idx,
                    "pose": pose,
                    "bbox": bbox,
                    "centers_in_image": centers_in_image,
                    "centers_valid": centers_valid,
                    "normals_in_image": normals_in_image,
                    "fg": fg_dist,
                    "bg": bg_dist,
                    "fore_hist": fore_hist,
                    "back_hist": back_hist,
                }
            )

    if best["idx"] is None:
        return None

    return best


@torch.no_grad()
def tracking_step(
    image_tensor,
    camera,
    current_pose,
    template_views,
    orientations,
    model,
    fore_hist,
    back_hist,
    track_cfg,
    data_conf,
    bbox_trim_ratio=0.0,
):
    """与demo_cube.py完全相同的追踪逻辑，使用多个模板视图"""
    device = image_tensor.device
    
    # 使用多个模板视图，与demo_cube.py一致
    # 确保current_pose没有batch维度（用于get_closest_k_template_view_index）
    if isinstance(current_pose, Pose) and current_pose._data.ndim > 1:
        current_pose_for_index = Pose(current_pose._data[0])
    else:
        current_pose_for_index = current_pose
    
    indices_full = get_closest_k_template_view_index(
        current_pose_for_index, orientations, data_conf.get_top_k_template_views * data_conf.skip_template_view
    )
    if indices_full.ndim > 1:
        indices_full = indices_full.view(-1)
    indices_full = indices_full[::data_conf.skip_template_view]
    indices_list = indices_full.tolist()
    closest_template_views = template_views[indices_list].contiguous()
    closest_orientations = orientations[indices_full].contiguous()

    # 确保current_pose有batch维度（用于模型），与demo_cube.py一致
    # demo_cube.py中：init_pose是Pose.from_Rt(R[3,3], t[3])，_data是[12]，然后init_pose[None]后_data是[1, 12]
    if isinstance(current_pose, Pose):
        if current_pose._data.ndim == 1:
            # _data是[12]，需要添加batch维度变成[1, 12]
            body2view_pose = current_pose[None]
        else:
            # _data是[1, 12]或其他，确保是[1, 12]格式
            if current_pose._data.ndim == 2:
                if current_pose._data.shape[0] > 1:
                    # 取第一个batch
                    body2view_pose = Pose(current_pose._data[0:1])
                else:
                    # 已经是[1, 12]，直接使用
                    body2view_pose = current_pose
            else:
                # 其他情况，取第一个并添加batch维度
                body2view_pose = Pose(current_pose._data.view(-1)[:12].unsqueeze(0))
    else:
        body2view_pose = current_pose
    
    data = {
        "image": image_tensor,
        "camera": camera,
        "body2view_pose": body2view_pose,
        "closest_template_views": closest_template_views[None],  # 与demo_cube.py一致：[K, N, 8] -> [1, K, N, 8]
        "closest_orientations_in_body": closest_orientations[None],  # 与demo_cube.py一致：[K, 3] -> [1, K, 3]
        "fore_hist": fore_hist,
        "back_hist": back_hist,
    }

    pred = model._forward(data, visualize=False, tracking=True)
    new_pose: Pose = pred["opt_body2view_pose"][-1][0]

    # 使用第一个模板视图计算直方图更新
    # 与demo_cube.py一致：closest_template_views是[K, N, 8]，需要[1, N, 8]
    (
        _,
        _,
        centers_in_image,
        centers_valid,
        normals_in_image,
        fg_dist,
        bg_dist,
        _,
    ) = calculate_basic_line_data(closest_template_views[None][:, 0], new_pose[None]._data, camera[None]._data, 1, 0)
    
    # 确保维度正确，移除多余的维度
    while centers_in_image.ndim > 3:
        centers_in_image = centers_in_image.squeeze(0)
    while centers_valid.ndim > 2:
        centers_valid = centers_valid.squeeze(0)
    while normals_in_image.ndim > 3:
        normals_in_image = normals_in_image.squeeze(0)
    while fg_dist.ndim > 2:
        fg_dist = fg_dist.squeeze(0)
    while bg_dist.ndim > 2:
        bg_dist = bg_dist.squeeze(0)

    fore_hist_new, back_hist_new = model.histogram.calculate_histogram(
        image_tensor, centers_in_image, centers_valid, normals_in_image, fg_dist, bg_dist, True
    )

    alpha_f = float(track_cfg.fore_learn_rate)
    alpha_b = float(track_cfg.back_learn_rate)
    fore_hist = (1 - alpha_f) * fore_hist + alpha_f * fore_hist_new
    back_hist = (1 - alpha_b) * back_hist + alpha_b * back_hist_new

    bbox = None
    valid_mask = centers_valid[0]
    if valid_mask.any():
        bbox = get_bbox_from_p2d(
            centers_in_image[0][valid_mask], trim_ratio=bbox_trim_ratio
        )

    return new_pose, fore_hist, back_hist, bbox, centers_in_image, centers_valid


@torch.no_grad()
def tracking_step_with_preprocess(
    frame_rgb,
    ori_camera_cpu,
    current_pose,
    template_views,
    orientations,
    model,
    fore_hist,
    back_hist,
    track_cfg,
    data_conf,
    device,
    bbox_trim_ratio=0.0,
):
    """带预处理的追踪函数，与demo_cube.py一致"""
    ori_camera = ori_camera_cpu.to(device)
    
    # 获取bbox进行预处理
    idx0 = get_closest_template_view_index(current_pose, orientations)
    initial_template = template_views[idx0 : idx0 + 1]
    data_lines = project_correspondences_line(initial_template, current_pose, ori_camera)
    bbox2d = get_bbox_from_p2d(data_lines["centers_in_image"][0], trim_ratio=bbox_trim_ratio)
    
    h_img, w_img = frame_rgb.shape[:2]
    if bbox2d.numel() < 4 or torch.isnan(bbox2d).any() or bbox2d[2] < 4 or bbox2d[3] < 4:
        bbox2d = torch.tensor([w_img / 2.0, h_img / 2.0, float(w_img), float(h_img)], dtype=torch.float32, device=device)
    
    img_tensor, camera = preprocess_image(frame_rgb, bbox2d.cpu().numpy().copy(), ori_camera_cpu, data_conf, device)
    
    # 调用追踪步骤
    new_pose, fore_hist, back_hist, _, centers_in_image_proc, centers_valid_proc = tracking_step(
        img_tensor[None],
        camera[None],
        current_pose[None],
        template_views,
        orientations,
        model,
        fore_hist,
        back_hist,
        track_cfg,
        data_conf,
        bbox_trim_ratio=bbox_trim_ratio,
    )
    # tracking_step返回的new_pose已经通过pred["opt_body2view_pose"][-1][0]移除了batch维度
    # 如果new_pose._data是[1, 12]，需要移除batch维度；如果已经是[12]，则不需要
    if isinstance(new_pose, Pose) and new_pose._data.ndim == 2 and new_pose._data.shape[0] == 1:
        new_pose = Pose(new_pose._data[0])
    elif isinstance(new_pose, Pose) and new_pose._data.ndim == 1:
        # 已经是[12]，不需要处理
        pass
    elif isinstance(new_pose, Pose) and new_pose._data.ndim == 2 and new_pose._data.shape[0] > 1:
        # 多个batch，取第一个
        new_pose = Pose(new_pose._data[0])
    
    # 在原始图像坐标系中计算bbox和centers
    idx_best = get_closest_template_view_index(new_pose, orientations)
    best_template = template_views[idx_best : idx_best + 1]
    centers_ori, valid_ori = ori_camera.view2image(new_pose.transform(best_template[0, :, :3]))
    
    bbox = None
    if valid_ori.any():
        bbox = get_bbox_from_p2d(centers_ori[valid_ori], trim_ratio=bbox_trim_ratio)
    
    return new_pose, fore_hist, back_hist, bbox, centers_ori, valid_ori


def initialize_from_pose_with_preprocess(
    frame_rgb,
    ori_camera_cpu,
    pose,
    template_views,
    orientations,
    model,
    device,
    data_conf,
    bbox_trim_ratio=0.0,
):
    """带预处理的初始化函数，与demo_cube.py一致"""
    ori_camera = ori_camera_cpu.to(device)
    
    # 获取bbox进行预处理
    idx0 = get_closest_template_view_index(pose, orientations)
    initial_template = template_views[idx0 : idx0 + 1]
    data_lines = project_correspondences_line(initial_template, pose, ori_camera)
    bbox2d = get_bbox_from_p2d(data_lines["centers_in_image"][0], trim_ratio=bbox_trim_ratio)
    
    h_img, w_img = frame_rgb.shape[:2]
    if bbox2d.numel() < 4 or torch.isnan(bbox2d).any() or bbox2d[2] < 4 or bbox2d[3] < 4:
        bbox2d = torch.tensor([w_img / 2.0, h_img / 2.0, float(w_img), float(h_img)], dtype=torch.float32, device=device)
    
    img_tensor, camera = preprocess_image(frame_rgb, bbox2d.cpu().numpy().copy(), ori_camera_cpu, data_conf, device)
    
    # 使用多个模板视图
    indices_full = get_closest_k_template_view_index(
        pose, orientations, data_conf.get_top_k_template_views * data_conf.skip_template_view
    )
    if indices_full.ndim > 1:
        indices_full = indices_full.view(-1)
    indices_full = indices_full[::data_conf.skip_template_view]
    indices_list = indices_full.tolist()
    closest_template_views = template_views[indices_list].contiguous()
    
    # 计算直方图
    # 与demo_cube.py完全相同的调用方式
    _, _, centers_in_image, centers_valid, normals_in_image, f_dist, b_dist, _ = calculate_basic_line_data(
        closest_template_views[None][:, 0], pose[None]._data, camera[None]._data, 1, 0
    )
    # 确保centers_in_image的形状是[batch, N, 2]，移除多余的维度
    while centers_in_image.ndim > 3:
        centers_in_image = centers_in_image.squeeze(0)
    while centers_valid.ndim > 2:
        centers_valid = centers_valid.squeeze(0)
    while normals_in_image.ndim > 3:
        normals_in_image = normals_in_image.squeeze(0)
    while f_dist.ndim > 2:
        f_dist = f_dist.squeeze(0)
    while b_dist.ndim > 2:
        b_dist = b_dist.squeeze(0)
    
    fore_hist, back_hist = model.histogram.calculate_histogram(
        img_tensor[None],
        centers_in_image,
        centers_valid,
        normals_in_image,
        f_dist,
        b_dist,
        True,
    )
    
    # 在原始图像坐标系中计算bbox
    centers_ori, valid_ori = ori_camera.view2image(pose.transform(initial_template[0, :, :3]))
    bbox = None
    if valid_ori.any():
        bbox = get_bbox_from_p2d(centers_ori[valid_ori], trim_ratio=bbox_trim_ratio)
    
    return fore_hist, back_hist, bbox, centers_ori, valid_ori


def initialize_from_pose(
    image_tensor, camera, pose, template_view, model, bbox_trim_ratio=0.0
):
    (
        _,
        _,
        centers_in_image,
        centers_valid,
        normals_in_image,
        fg_dist,
        bg_dist,
        _,
    ) = calculate_basic_line_data(template_view, pose._data, camera._data, 1, 0)

    fore_hist, back_hist = model.histogram.calculate_histogram(
        image_tensor, centers_in_image, centers_valid, normals_in_image, fg_dist, bg_dist, True
    )

    bbox = None
    if centers_valid[0].any():
        bbox = get_bbox_from_p2d(
            centers_in_image[0][centers_valid[0]], trim_ratio=bbox_trim_ratio
        )

    return fore_hist, back_hist, bbox, centers_in_image, centers_valid


def estimate_translation_from_bbox(camera, bbox, object_diameter):
    if bbox is None:
        return None

    # 正确处理Camera._data的格式：[width, height, fx, fy, cx, cy]
    # Camera._data可能是1D [6] 或 2D [1, 6]
    cam_data = camera._data
    if cam_data.ndim > 1:
        cam_data = cam_data[0]  # 如果有batch维度，取第一个
    
    # cam_data格式：[width, height, fx, fy, cx, cy]，现在是1D [6]
    fx = float(cam_data[2])
    fy = float(cam_data[3])
    cx = float(cam_data[4])
    cy = float(cam_data[5])

    bbox = bbox.detach().cpu()
    box_w = bbox[2].item()
    box_h = bbox[3].item()
    if box_w <= 1e-6 or box_h <= 1e-6 or fx <= 0 or fy <= 0:
        return None

    center_x = bbox[0].item()
    center_y = bbox[1].item()

    z = fx * object_diameter / box_w
    x = (center_x - cx) / fx * z
    y = (center_y - cy) / fy * z

    return torch.tensor([[x, y, z]], dtype=torch.float32, device=camera.device)


@torch.no_grad()
def draw_overlay(frame_rgb, bbox, centers_in_image, centers_valid, color=(0, 255, 0), input_is_bgr=False):
    frame_bgr = frame_rgb if input_is_bgr else cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if bbox is not None:
        x1, y1, x2, y2 = tensor_bbox_to_xyxy(bbox.cpu())
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    if centers_in_image is not None and centers_valid is not None:
        # 统一处理：移除所有batch维度，确保centers_in_image是[N, 2]，centers_valid是[N]
        try:
            # 处理centers_in_image
            if centers_in_image.ndim == 3:
                centers_flat = centers_in_image[0]  # [N, 2]
            elif centers_in_image.ndim == 2:
                centers_flat = centers_in_image  # [N, 2]
            else:
                centers_flat = None
            
            # 处理centers_valid，确保是1维的[N]
            if centers_valid.ndim == 3:
                valid_flat = centers_valid[0, 0] if centers_valid.shape[0] == 1 else centers_valid[0]
            elif centers_valid.ndim == 2:
                if centers_valid.shape[0] == 1:
                    valid_flat = centers_valid[0]  # [N]
                else:
                    valid_flat = centers_valid[0]  # 取第一个batch
            elif centers_valid.ndim == 1:
                valid_flat = centers_valid  # [N]
            else:
                valid_flat = None
            
            # 确保valid_flat是1维布尔张量
            if valid_flat is not None and valid_flat.ndim > 1:
                valid_flat = valid_flat.squeeze()
            if valid_flat is not None and valid_flat.dtype != torch.bool:
                valid_flat = valid_flat.bool()
            
            # 现在centers_flat是[N, 2]，valid_flat是[N]布尔张量
            # 绘制闭合多边形（与demo_cube.py一致），而不是小点
            if centers_flat is not None and valid_flat is not None:
                pts = centers_flat[valid_flat].detach().cpu().numpy()
                if pts.shape[0] >= 3:  # 至少需要3个点才能形成闭合多边形
                    h_img, w_img = frame_bgr.shape[:2]
                    # 限制点在图像范围内
                    pts = np.clip(pts, [0, 0], [w_img - 1, h_img - 1])
                    pts_int = pts.reshape(-1, 1, 2).astype(np.int32)
                    # 绘制闭合多边形
                    cv2.polylines(frame_bgr, [pts_int], isClosed=True, color=color, thickness=2)
        except Exception:
            # 如果出错，跳过绘制
            pass
    return frame_bgr


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger = MyLightningLogger("deepac-live", str(save_dir))
    logger.dump_cfg(cfg, "live_cfg.yaml")

    model, train_cfg = load_model(cfg, device, logger)
    logger.dump_cfg(train_cfg, "train_cfg.yaml")

    # 同步数据配置，与demo_cube.py一致
    data_conf = train_cfg.data
    data_conf.crop_border = cfg.tracking.get("crop_border", 10)
    data_conf.resize = cfg.tracking.get("resize", 320)
    data_conf.pad = cfg.tracking.get("pad", 0)
    data_conf.resize_by = cfg.tracking.get("resize_by", "max")
    template_top_k = cfg.tracking.get("template_top_k", 10)
    template_skip = cfg.tracking.get("template_skip", 1)
    if template_top_k is not None and hasattr(data_conf, "get_top_k_template_views"):
        data_conf.get_top_k_template_views = max(data_conf.get_top_k_template_views, template_top_k)
    if template_skip is not None and hasattr(data_conf, "skip_template_view"):
        data_conf.skip_template_view = template_skip

    template_views, orientations, _ = load_template(cfg.object.pre_render_pkl, device)
    initial_view_idx, initial_pose, initial_template = prepare_initial_pose(
        cfg, template_views, orientations, device
    )
    logger.info(f"Guide overlay uses template view index {initial_view_idx}")

    cap = cv2.VideoCapture(int(cfg.camera.camera_id))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.camera.set_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.camera.set_height))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera id={cfg.camera.camera_id}")

    logger.info(
        f"Camera opened (id={cfg.camera.camera_id}) at "
        f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
    )

    initialized = False
    current_pose = None
    fore_hist = None
    back_hist = None
    last_bbox = None
    last_centers = None
    last_valid = None

    detect_interval = max(1, int(cfg.tracking.detect_interval))
    detect_threshold = float(cfg.tracking.detect_threshold)
    bbox_trim_ratio = float(cfg.tracking.get("bbox_trim_ratio", 0.1))
    auto_start_enabled = bool(cfg.tracking.get("auto_start_enabled", True))
    auto_start_iou = float(cfg.tracking.get("auto_start_iou", 0.45))
    auto_start_score = float(cfg.tracking.get("auto_start_score", detect_threshold))
    frame_idx = 0
    guide_color = (255, 128, 0)
    guide_cache = {"bbox": None, "centers": None, "valid": None}
    obj_diameter = getattr(cfg.object, "diameter_in_meter", None)

    video_writer = None
    if cfg.tracking.output_video:
        out_size = tuple(cfg.tracking.output_size)
        video_path = save_dir / f"live_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            cfg.tracking.get("output_fps", 30),
            out_size,
        )
        logger.info(f"Recording video to {video_path}")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                logger.warn("Failed to grab frame from camera; stopping")
                break

            frame_rgb, ori_camera_cpu = preprocess_frame(frame_bgr, cfg.camera, device)
            ori_camera = ori_camera_cpu.to(device)

            alignment_metric = None

            if not initialized:
                # 使用原始相机渲染引导框
                (
                    _,
                    _,
                    guide_centers,
                    guide_valid,
                    _,
                    _,
                    _,
                    _,
                ) = calculate_basic_line_data(initial_template, initial_pose._data, ori_camera._data, 1, 0)
                guide_bbox = None
                if guide_valid[0].any():
                    guide_bbox = get_bbox_from_p2d(
                        guide_centers[0][guide_valid[0]], trim_ratio=bbox_trim_ratio
                    )
                overlay = draw_overlay(frame_rgb, guide_bbox, guide_centers, guide_valid, color=guide_color)
                guide_cache["bbox"] = guide_bbox
                guide_cache["centers"] = guide_centers
                guide_cache["valid"] = guide_valid
            else:
                overlay = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if not initialized:
                if frame_idx % detect_interval == 0:
                    # 检测时也需要使用预处理
                    detect_result = detect_once_with_preprocess(
                        frame_rgb,
                        ori_camera_cpu,
                        initial_pose,
                        template_views,
                        orientations,
                        model,
                        device,
                        cfg.tracking,
                        data_conf,
                        bbox_trim_ratio=bbox_trim_ratio,
                    )
                    if detect_result:
                        last_bbox = detect_result["bbox"]
                        last_centers = detect_result["centers_in_image"]
                        last_valid = detect_result["centers_valid"]
                        if (
                            guide_cache["bbox"] is not None
                            and detect_result["bbox"] is not None
                        ):
                            alignment_metric = bbox_iou(
                                guide_cache["bbox"], detect_result["bbox"]
                            )
                        if (
                            auto_start_enabled
                            and detect_result["score"] >= auto_start_score
                            and alignment_metric is not None
                            and alignment_metric >= auto_start_iou
                        ):
                            det_idx = int(detect_result["idx"])
                            current_pose = Pose(detect_result["pose"]._data.clone()).to(
                                device
                            )
                            if (
                                obj_diameter is not None
                                and detect_result["bbox"] is not None
                            ):
                                est_translation = estimate_translation_from_bbox(
                                    ori_camera,
                                    detect_result["bbox"],
                                    float(obj_diameter),
                                )
                                if est_translation is not None:
                                    current_pose._data[..., 9:12] = est_translation

                            # 使用预处理后的图像和相机初始化
                            (
                                fore_hist,
                                back_hist,
                                last_bbox,
                                last_centers,
                                last_valid,
                            ) = initialize_from_pose_with_preprocess(
                                frame_rgb,
                                ori_camera_cpu,
                                current_pose,
                                template_views,
                                orientations,
                                model,
                                device,
                                data_conf,
                                bbox_trim_ratio=bbox_trim_ratio,
                            )
                            initialized = True
                            frame_idx = 0
                            logger.info(
                                f"Auto-start tracking: view={det_idx} score={detect_result['score']:.3f} "
                                f"IoU={alignment_metric:.3f}"
                            )
                            overlay = draw_overlay(
                                frame_rgb,
                                last_bbox,
                                last_centers,
                                last_valid,
                                color=(0, 255, 0),
                            )
                            if video_writer is not None:
                                resized = cv2.resize(overlay, out_size)
                                video_writer.write(resized)
                            continue
                if last_centers is not None and last_valid is not None:
                    overlay = draw_overlay(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        last_bbox,
                        last_centers,
                        last_valid,
                        color=(0, 255, 255),
                        input_is_bgr=False,
                    )
                cv2.putText(
                    overlay,
                    "Align cube with guide...",
                    (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 200, 255),
                    2,
                )
                if alignment_metric is not None:
                    cv2.putText(
                        overlay,
                        f"IoU {alignment_metric:.2f}/{auto_start_iou:.2f}",
                        (16, 118),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 255),
                        2,
                    )
                cv2.putText(
                    overlay,
                    "Tracking starts automatically",
                    (16, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )
            else:
                # 追踪时使用预处理
                (
                    current_pose,
                    fore_hist,
                    back_hist,
                    last_bbox,
                    last_centers,
                    last_valid,
                ) = tracking_step_with_preprocess(
                    frame_rgb,
                    ori_camera_cpu,
                    current_pose,
                    template_views,
                    orientations,
                    model,
                    fore_hist,
                    back_hist,
                    cfg.tracking,
                    data_conf,
                    device,
                    bbox_trim_ratio=bbox_trim_ratio,
                )
                # 在原始图像上绘制结果
                overlay = draw_overlay(
                    frame_rgb,
                    last_bbox,
                    last_centers,
                    last_valid,
                    color=(0, 255, 0),
                )

            frame_idx += 1

            key = -1
            if cfg.tracking.show_window:
                cv2.imshow("DeepAC Live", overlay)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if not initialized and key == ord("s"):
                logger.info("Manual initialization requested (key 's') using guide pose directly")
                current_pose = Pose(initial_pose._data.clone()).to(device)

                guide_bbox = guide_cache.get("bbox")
                if obj_diameter is not None and guide_bbox is not None:
                    est_translation = estimate_translation_from_bbox(ori_camera, guide_bbox, float(obj_diameter))
                    if est_translation is not None:
                        current_pose._data[..., 9:12] = est_translation

                (
                    fore_hist,
                    back_hist,
                    last_bbox,
                    last_centers,
                    last_valid,
                ) = initialize_from_pose_with_preprocess(
                    frame_rgb,
                    ori_camera_cpu,
                    current_pose,
                    template_views,
                    orientations,
                    model,
                    device,
                    data_conf,
                    bbox_trim_ratio=bbox_trim_ratio,
                )
                overlay = draw_overlay(
                    frame_rgb,
                    last_bbox,
                    last_centers,
                    last_valid,
                    color=(0, 255, 0),
                )
                initialized = True
                frame_idx = 0
                if video_writer is not None:
                    resized = cv2.resize(overlay, out_size)
                    video_writer.write(resized)
                continue

            if key == ord("q") or key == 27:
                break

            if video_writer is not None:
                resized = cv2.resize(overlay, out_size)
                video_writer.write(resized)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        logger.info("Live tracking finished")


if __name__ == "__main__":
    main()

