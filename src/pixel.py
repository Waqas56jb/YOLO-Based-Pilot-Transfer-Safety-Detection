import cv2
import numpy as np


def enhance_contrast_yuv(img_bgr: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_eq = clahe.apply(y)
    img_yuv_eq = cv2.merge((y_eq, u, v))
    return cv2.cvtColor(img_yuv_eq, cv2.COLOR_YUV2BGR)


def denoise_frame(img_bgr: np.ndarray, h: float = 3.0, template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, template_window_size, search_window_size)


def unsharp_mask(img_bgr: np.ndarray, sigma: float = 1.0, amount: float = 1.5, threshold: int = 0) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)
    if threshold <= 0:
        return sharpened
    low_contrast_mask = np.max(np.abs(sharpened.astype(np.int16) - img_bgr.astype(np.int16)), axis=2) < threshold
    result = img_bgr.copy()
    result[~low_contrast_mask] = sharpened[~low_contrast_mask]
    return result


