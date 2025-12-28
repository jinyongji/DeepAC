# src_open/utils/image_utils.py
import numpy as np
import cv2
import torch

def crop(img, bbox, camera=None, return_bbox=False):
    """
    裁剪图像并返回相机内参更新后的 Camera
    bbox: [cx, cy, w, h]
    """
    cx, cy, w, h = bbox.astype(int)
    x1, y1 = int(cx - w // 2), int(cy - h // 2)
    x2, y2 = int(cx + w // 2), int(cy + h // 2)

    cropped = img[y1:y2, x1:x2]

    if camera is not None:
        from src_open.utils.geometry.wrappers import Camera
        new_camera = camera.translate((-x1, -y1))
    else:
        new_camera = None

    if return_bbox:
        return cropped, new_camera, (x1, y1, x2, y2)
    else:
        return cropped, new_camera

def resize(img, size, fn=min):
    """
    按最大边/最小边等比例缩放，或者直接 resize 到 size
    """
    if isinstance(size, int):
        h, w = img.shape[:2]
        scale = size / fn(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h))
        return resized, (scale, scale)
    elif isinstance(size, (list, tuple)):
        resized = cv2.resize(img, (size[0], size[1]))
        return resized, (size[0] / img.shape[1], size[1] / img.shape[0])
    else:
        raise ValueError("size 参数错误")

def zero_pad(pad, img):
    """
    四周补零
    """
    return (np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="constant"),)

def numpy_image_to_torch(img):
    """
    numpy HWC -> torch CHW，并归一化到 [0,1]
    """
    if img.ndim == 2:  # 灰度
        img = np.expand_dims(img, axis=-1)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    return img