from PIL import Image
import numpy as np
from ....classes.kimage import Kimage


def save_png(kimage: Kimage, path: str) -> None:
    img = Image.fromarray(kimage.kimage, mode='RGBA')
    img.save(path, format='PNG')


def load_png(path: str) -> Kimage:
    img = Image.open(path).convert('RGBA')
    arr = np.array(img, dtype=np.uint8)

    row, column = arr.shape[0], arr.shape[1]
    kimage = Kimage(row=row, column=column)
    kimage.kimage[:] = arr

    return kimage
