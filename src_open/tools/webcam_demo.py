import os
import time
from pathlib import Path
import numpy as np
import torch
import cv2
import pickle
from omegaconf import DictConfig

from ..utils.geometry.wrappers import Pose, Camera
from ..models import get_model
from ..utils.lightening_utils import MyLightningLogger, convert_old_model, load_model_weight
from ..utils.utils import project_correspondences_line, get_closest_template_view_index,\
    get_closest_k_template_view_index, get_bbox_from_p2d
from ..models.deep_ac import calculate_basic_line_data
from ..dataset.utils import numpy_image_to_torch, crop, resize, zero_pad

@torch.no_grad()
def main(cfg: DictConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    logger = MyLightningLogger('DeepAC-Webcam', cfg.save_dir)
    logger.dump_cfg(cfg, 'demo_webcam_cfg.yml')

    # Load trained model
    assert Path(cfg.load_cfg).exists()
    train_cfg = torch.load(cfg.load_cfg) if str(cfg.load_cfg).endswith('.pth') else None
    if train_cfg is None:
        from omegaconf import OmegaConf
        train_cfg = OmegaConf.load(cfg.load_cfg)
    model = get_model(train_cfg.models.name)(train_cfg.models)
    ckpt = torch.load(cfg.load_model, map_location='cpu')
    if "pytorch-lightning_version" not in ckpt:
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    model.cuda().eval()

    # Load pre-rendered template data
    with open(cfg.pre_render_pkl, "rb") as f:
        pre_render_dict = pickle.load(f)
    head = pre_render_dict['head']
    num_sample_contour_points = head['num_sample_contour_point']
    template_views = torch.from_numpy(pre_render_dict['template_view']).type(torch.float32)
    orientations = torch.from_numpy(pre_render_dict['orientation_in_body']).type(torch.float32)

    # Build intrinsic vector (will update width/height per frame)
    fx0, fy0, cx0, cy0 = float(cfg.fx), float(cfg.fy), float(cfg.cx), float(cfg.cy)
    calib_w = int(getattr(cfg, 'calib_width', -1))
    calib_h = int(getattr(cfg, 'calib_height', -1))

    # Init pose roughly in front of camera
    init_R = torch.eye(3, dtype=torch.float32)
    init_t = torch.tensor([0.0, 0.0, float(cfg.init_translation_z_m)], dtype=torch.float32)
    init_pose = Pose.from_Rt(init_R, init_t)

    # Open webcam
    cap = cv2.VideoCapture(int(cfg.camera_id))
    # Optionally force a specific capture size to match calibration
    set_w = int(getattr(cfg, 'set_width', -1))
    set_h = int(getattr(cfg, 'set_height', -1))
    if set_w > 0 and set_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, set_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, set_h)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cfg.camera_id}")

    # Prepare writer
    video_writer = None
    if cfg.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'MP42')

    total_fore_hist = None
    total_back_hist = None

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            ori_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = ori_image.shape[:2]

            if video_writer is None and cfg.output_video:
                out_w, out_h = int(cfg.output_size[0]), int(cfg.output_size[1])
                video_writer = cv2.VideoWriter(str(Path(logger.log_dir) / f"{cfg.obj_name}.avi"), fourcc, 30, (out_w, out_h))

            # Scale intrinsics if capture size differs from calibration size
            if calib_w > 0 and calib_h > 0:
                sx = width / float(calib_w)
                sy = height / float(calib_h)
            else:
                sx = 1.0
                sy = 1.0
            fx = fx0 * sx
            fy = fy0 * sy
            cx = cx0 * sx
            cy = cy0 * sy

            # Camera pack: [w, h, fx, fy, cx, cy]
            intrinsic_param = torch.tensor([width, height, fx, fy, cx, cy], dtype=torch.float32)
            ori_camera = Camera(intrinsic_param)

            # pick closest template view(s) based on current pose
            indices = get_closest_k_template_view_index(init_pose, orientations, train_cfg.data.get_top_k_template_views * train_cfg.data.skip_template_view)
            closest_template_views = torch.stack([
                template_views[ind * num_sample_contour_points:(ind + 1) * num_sample_contour_points, :]
                for ind in indices[::train_cfg.data.skip_template_view]
            ])
            closest_orientations_in_body = orientations[indices[::train_cfg.data.skip_template_view]]

            # compute bbox and preprocess
            data_lines = project_correspondences_line(closest_template_views[0], init_pose, ori_camera)
            bbox2d = get_bbox_from_p2d(data_lines['centers_in_image'])

            bbox2d_np = bbox2d.numpy().copy()
            bbox2d_np[2:] += train_cfg.data.crop_border * 2
            img, camera, _ = crop(ori_image, bbox2d_np, camera=ori_camera, return_bbox=True)

            scales = (1, 1)
            if isinstance(train_cfg.data.resize, int):
                if train_cfg.data.resize_by == 'max':
                    img, scales = resize(img, train_cfg.data.resize, fn=max)
                elif train_cfg.data.resize_by == 'min' or (train_cfg.data.resize_by == 'min_if' and min(*img.shape[:2]) < train_cfg.data.resize):
                    img, scales = resize(img, train_cfg.data.resize, fn=min)
            elif isinstance(train_cfg.data.resize, (list, tuple)) and len(train_cfg.data.resize) == 2:
                img, scales = resize(img, list(train_cfg.data.resize))
            if scales != (1, 1):
                camera = camera.scale(scales)

            img, = zero_pad(train_cfg.data.pad, img)
            img = img.astype(np.float32)
            img_t = numpy_image_to_torch(img)

            # init hist at first frame
            if total_fore_hist is None:
                tv0 = closest_template_views[None][:, 0]
                _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ = \
                    calculate_basic_line_data(tv0, init_pose[None]._data, camera[None]._data, 1, 0)
                total_fore_hist, total_back_hist = \
                    model.histogram.calculate_histogram(img_t[None], centers_in_image, centers_valid, normals_in_image, \
                                                        foreground_distance, background_distance, True)

            data = {
                'image': img_t[None].cuda(),
                'camera': camera[None].cuda(),
                'body2view_pose': init_pose[None].cuda(),
                'closest_template_views': closest_template_views[None].cuda(),
                'closest_orientations_in_body': closest_orientations_in_body[None].cuda(),
                'fore_hist': total_fore_hist.cuda(),
                'back_hist': total_back_hist.cuda()
            }
            pred = model._forward(data, visualize=False, tracking=True)

            # visualize and write
            if cfg.output_video:
                pred['optimizing_result_imgs'] = []
                model.visualize_optimization(pred['opt_body2view_pose'][-1], pred)
                frame = cv2.resize(pred['optimizing_result_imgs'][0][0], (int(cfg.output_size[0]), int(cfg.output_size[1])))
                video_writer.write(frame)

            # update pose and hist
            init_pose = pred['opt_body2view_pose'][-1][0].cpu()
            index = get_closest_template_view_index(init_pose, orientations)
            closest_template_view = template_views[index*num_sample_contour_points:(index+1)*num_sample_contour_points, :]
            _, _, centers_in_image, centers_valid, normals_in_image, foreground_distance, background_distance, _ = \
                calculate_basic_line_data(closest_template_view[None], init_pose[None]._data, camera[None]._data, 1, 0)
            fore_hist, back_hist = model.histogram.calculate_histogram(img_t[None], centers_in_image, centers_valid, normals_in_image, \
                                                                       foreground_distance, background_distance, True)
            total_fore_hist = (1 - cfg.fore_learn_rate) * total_fore_hist + cfg.fore_learn_rate * fore_hist
            total_back_hist = (1 - cfg.back_learn_rate) * total_back_hist + cfg.back_learn_rate * back_hist

            # show live window
            if bool(getattr(cfg, 'show_window', True)):
                cv2.imshow('DeepAC Webcam', bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
