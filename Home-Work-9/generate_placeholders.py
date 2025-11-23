import cv2
import numpy as np
from pathlib import Path

base = Path(r"D:\Intro2NLP\Home-Work-9")
base.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

def save_gradient(path, width, height, color=True):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv = np.tile(x, (height, 1))
    yv = np.tile(y[:, None], (1, width))
    if color:
        img = np.stack([
            yv,
            np.flipud(yv),
            xv
        ], axis=2)
    else:
        base_layer = (0.6 * yv + 0.4 * xv).astype(np.uint8)
        noise = rng.integers(0, 20, size=base_layer.shape, dtype=np.uint8)
        img = np.clip(base_layer + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(base / path), img)

save_gradient("almaty.jpg", 1200, 800, True)
save_gradient("input_image.jpg", 768, 512, False)
save_gradient("image_Venice.jpg", 1024, 768, True)
