import os
from pathlib import Path
import glob
import pickle

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src_open.utils.geometry.wrappers import Pose, Camera
from src_open.models import get_model
from src_open.utils.lightening_utils import MyLightningLogger, convert_old_model, load_model_weight
from src_open.dataset.utils import read_image, resize, numpy_image_to_torch, crop, zero_pad
from src_open.utils.utils import (
    project_correspondences_line,
    get_closest_template_view_index,
    get_closest_k_template_view_index,
    get_bbox_from_p2d,
)
from src_open.models.deep_ac import calculate_basic_line_data


@torch.no_grad()
def run_demo(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id

    logger = MyLightningLogger("DeepAC_Cube", cfg.save_dir)
    logger.dump_cfg(cfg, "demo_cfg.yml")

    train_cfg = OmegaConf.load(cfg.load_cfg)
    data_conf = train_cfg.data
    data_conf.crop_border = cfg.crop_border
    data_conf.resize = cfg.resize
    data_conf.pad = cfg.pad
    data_conf.resize_by = cfg.resize_by
    template_top_k = getattr(cfg, "template_top_k", None)
    template_skip = getattr(cfg, "template_skip", None)
    if template_top_k is not None and hasattr(data_conf, "get_top_k_template_views"):
        data_conf.get_top_k_template_views = max(data_conf.get_top_k_template_views, template_top_k)
    if template_skip is not None and hasattr(data_conf, "skip_template_view"):
        data_conf.skip_template_view = template_skip
    logger.dump_cfg(train_cfg, "train_cfg.yml")

    model = get_model(train_cfg.models.name)(train_cfg.models)
    ckpt = torch.load(cfg.load_model, map_location="cpu")
    if "pytorch-lightning_version" not in ckpt:
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    model.to(device).eval()

    fore_learn_rate = cfg.fore_learn_rate
    back_learn_rate = cfg.back_learn_rate

    data_dir = cfg.data_dir
    obj_name = cfg.obj_name
    img_dir = os.path.join(data_dir, cfg.image_subdir)
    pose_path = os.path.join(data_dir, cfg.pose_file)
    K_path = os.path.join(data_dir, "K.txt")
    template_path = os.path.join(data_dir, obj_name, "pre_render", f"{obj_name}.pkl")

    with open(template_path, "rb") as f:
        pre_render_dict = pickle.load(f)
    head = pre_render_dict["head"]
    num_sample = head["num_sample_contour_point"]
    template_views_flat = torch.from_numpy(pre_render_dict["template_view"]).float()
    num_views = template_views_flat.shape[0] // num_sample
    template_views = template_views_flat.view(num_views, num_sample, -1).to(device)
    orientations = torch.from_numpy(pre_render_dict["orientation_in_body"]).float().to(device)

    K = torch.from_numpy(np.loadtxt(K_path)).float()
    if K.numel() == 9:
        K = K.view(3, 3)

    poses_np = np.loadtxt(pose_path)
    if poses_np.ndim == 1:
        poses_np = poses_np[None, :]
    poses = torch.from_numpy(poses_np).float()
    translation_scale = cfg.translation_unit_in_meter

    img_lists = sorted(
        glob.glob(os.path.join(img_dir, "frame_*.jpg"))
        + glob.glob(os.path.join(img_dir, "frame_*.png"))
        + glob.glob(os.path.join(img_dir, "frame_*.jpeg"))
    )
    if not img_lists:
        raise FileNotFoundError(f"No frames found under {img_dir}")

    pose_mats = poses.view(-1, 3, 4)

    if cfg.get("start_frame"):
        name_to_idx = {Path(p).name: i for i, p in enumerate(img_lists)}
        start_index = name_to_idx.get(cfg.start_frame, 0)
        if cfg.start_frame not in name_to_idx:
            logger.info(f"start_frame {cfg.start_frame} not found, fallback to first frame.")
        img_lists = img_lists[start_index:]
    else:
        start_index = 0

    pose_idx = min(start_index, pose_mats.shape[0] - 1)
    init_pose_mat = pose_mats[pose_idx]
    init_R = init_pose_mat[:, :3]
    init_t = init_pose_mat[:, 3] * translation_scale
    init_pose = Pose.from_Rt(init_R, init_t).to(device)

    logger.info(f"Initial translation (mm): {(init_t / translation_scale).tolist()}")

    save_dir = Path(logger.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    smooth_alpha = float(getattr(cfg, "smooth_alpha", 0.0))
    use_smoothing = 0.0 < smooth_alpha < 1.0
    bbox_trim_ratio = float(getattr(cfg, "bbox_trim_ratio", 0.0))

    if cfg.output_video:
        video = cv2.VideoWriter(
            str(save_dir / f"{obj_name}.avi"),
            cv2.VideoWriter_fourcc(*"FMP4"),
            30,
            tuple(cfg.output_size),
        )
    else:
        video = None

    pose_file = open(save_dir / f"{obj_name}_pose.txt", "w")

    first_img = read_image(img_lists[0])
    fh, fw = first_img.shape[:2]
    intrinsic_first = torch.tensor([fw, fh, K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)
    cam_first = Camera(intrinsic_first).to(device)

    idx0 = get_closest_template_view_index(init_pose, orientations)
    initial_template = template_views[idx0, :, :3]
    pts_cam = init_pose.transform(initial_template)
    p2d, valid = cam_first.view2image(pts_cam)
    img_vis = first_img.copy()
    for (x, y), v in zip(p2d.detach().cpu().numpy(), valid.detach().cpu().numpy()):
        if v:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < fw and 0 <= yi < fh:
                cv2.circle(img_vis, (xi, yi), 2, (0, 0, 255), -1)
    init_check_path = save_dir / "initial_projection_check.jpg"
    cv2.imwrite(str(init_check_path), img_vis)
    logger.info(f"Saved initial projection check image to {init_check_path}")

    def preprocess_image(img, bbox2d, camera_cpu):
        bbox2d[2:] += data_conf.crop_border * 2
        img_crop, camera_out, _ = crop(img, bbox2d, camera=camera_cpu, return_bbox=True)
        if isinstance(data_conf.resize, int) and data_conf.resize > 0:
            img_crop, scales = resize(img_crop, data_conf.resize, fn=max if data_conf.resize_by == "max" else min)
            camera_out = camera_out.scale(scales)
        if isinstance(data_conf.pad, int) and data_conf.pad > 0:
            img_crop, = zero_pad(data_conf.pad, img_crop)
        img_tensor = numpy_image_to_torch(img_crop.astype(np.float32))
        _, h_t, w_t = img_tensor.shape
        pad_h = (32 - h_t % 32) % 32
        pad_w = (32 - w_t % 32) % 32
        if pad_h or pad_w:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))
            cam_data = camera_out._data.clone()
            cam_data[..., 0] += pad_w
            cam_data[..., 1] += pad_h
            camera_out = Camera(cam_data)
        return img_tensor.to(device), camera_out.to(device)

    total_fore_hist = None
    total_back_hist = None
    pose_smooth = init_pose.detach() if use_smoothing else None

    def smooth_pose(prev_pose: Pose, new_pose: Pose, alpha: float) -> Pose:
        if prev_pose is None or alpha <= 0.0:
            return new_pose.detach()
        device_local = new_pose._data.device
        prev_pose_d = prev_pose.to(device_local)
        prev_R = prev_pose_d.R
        new_R = new_pose.R
        blended = alpha * prev_R + (1.0 - alpha) * new_R
        u, _, vT = torch.linalg.svd(blended)
        R_s = u @ vT
        det_val = torch.det(R_s)
        if det_val.dim() == 0:
            if det_val < 0:
                u[:, -1] *= -1
                R_s = u @ vT
        else:
            neg_idx = det_val < 0
            if neg_idx.any():
                u[neg_idx, :, -1] *= -1
                R_s = u @ vT
        prev_t = prev_pose_d.t
        new_t = new_pose.t
        t_s = alpha * prev_t + (1.0 - alpha) * new_t
        pose = Pose.from_Rt(R_s, t_s)
        return pose.to(device_local)

    for i, img_path in enumerate(tqdm(img_lists, desc="Tracking")):
        ori_image = read_image(img_path)
        h, w = ori_image.shape[:2]
        intrinsic_param = torch.tensor([w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)
        ori_camera_cpu = Camera(intrinsic_param)
        ori_camera = ori_camera_cpu.to(device)

        indices_full = get_closest_k_template_view_index(
            init_pose, orientations, data_conf.get_top_k_template_views * data_conf.skip_template_view
        )
        if indices_full.ndim > 1:
            indices_full = indices_full.view(-1)
        indices_full = indices_full[::data_conf.skip_template_view]
        indices_list = indices_full.tolist()
        closest_template_views = template_views[indices_list].contiguous()
        closest_orientations = orientations[indices_full].contiguous()

        template_view_first = closest_template_views[:1]
        data_lines = project_correspondences_line(template_view_first, init_pose, ori_camera)
        bbox2d = get_bbox_from_p2d(data_lines["centers_in_image"][0], trim_ratio=bbox_trim_ratio)
        img_tensor, camera = preprocess_image(ori_image, bbox2d.cpu().numpy().copy(), ori_camera_cpu)

        if total_fore_hist is None:
            _, _, centers_in_image, centers_valid, normals_in_image, f_dist, b_dist, _ = calculate_basic_line_data(
                closest_template_views[None][:, 0], init_pose[None]._data, camera[None]._data, 1, 0
            )
            total_fore_hist, total_back_hist = model.histogram.calculate_histogram(
                img_tensor[None],
                centers_in_image,
                centers_valid,
                normals_in_image,
                f_dist,
                b_dist,
                True,
            )

        data = {
            "image": img_tensor[None],
            "camera": camera[None],
            "body2view_pose": init_pose[None],
            "closest_template_views": closest_template_views[None],
            "closest_orientations_in_body": closest_orientations[None],
            "fore_hist": total_fore_hist,
            "back_hist": total_back_hist,
        }
        data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
        pred = model._forward(data, visualize=False, tracking=True)

        opt_pose = pred["opt_body2view_pose"][-1][0]
        opt_pose_det = opt_pose.detach()
        if use_smoothing:
            pose_smooth = smooth_pose(pose_smooth, opt_pose_det, smooth_alpha)
            pose_for_output = pose_smooth
        else:
            pose_for_output = opt_pose_det
        init_pose = opt_pose_det

        pose_out_cpu = pose_for_output.cpu()
        Rt = torch.cat([pose_out_cpu.R.reshape(-1), (pose_out_cpu.t / translation_scale).reshape(-1)])
        pose_file.write(" ".join(f"{x:.8f}" for x in Rt.tolist()) + "\n")

        best_idx = get_closest_template_view_index(init_pose, orientations)
        closest_template_view_hist = template_views[best_idx]

        if video is not None:
            frame_vis = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)
            pose_for_vis = pose_for_output
            idx_vis = get_closest_template_view_index(pose_for_vis, orientations)
            template_view_vis = template_views[idx_vis]
            projected = project_correspondences_line(template_view_vis[None], pose_for_vis, ori_camera)
            centers = projected["centers_in_image"][0]
            valid_mask = projected["centers_valid"][0].bool()
            pts = centers[valid_mask].detach().cpu().numpy()
            if pts.shape[0] >= 3:
                h_vis, w_vis = frame_vis.shape[:2]
                pts = np.clip(pts, [0, 0], [w_vis - 1, h_vis - 1])
                pts_int = pts.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(frame_vis, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)
            text = f"frame {start_index + i:04d}"
            cv2.putText(frame_vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            video.write(cv2.resize(frame_vis, tuple(cfg.output_size)))

        _, _, centers_in_image, centers_valid, normals_in_image, f_dist, b_dist, _ = calculate_basic_line_data(
            closest_template_view_hist[None], init_pose[None]._data, camera[None]._data, 1, 0
        )
        fore_hist, back_hist = model.histogram.calculate_histogram(
            img_tensor[None],
            centers_in_image,
            centers_valid,
            normals_in_image,
            f_dist,
            b_dist,
            True,
        )
        total_fore_hist = (1 - fore_learn_rate) * total_fore_hist + fore_learn_rate * fore_hist
        total_back_hist = (1 - back_learn_rate) * total_back_hist + back_learn_rate * back_hist

    pose_file.close()
    if video is not None:
        video.release()
    logger.info(f"Tracking finished! Results saved to {save_dir}")


def main():
    cfg_path = "src_open/configs/demo/demo_cube.yaml"
    cfg = OmegaConf.load(cfg_path)
    run_demo(cfg)


if __name__ == "__main__":
    main()
