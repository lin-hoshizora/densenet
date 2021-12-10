from pathlib import Path
import cv2
from albumentations.augmentations.transforms import ElasticTransform


def blur(img, ksize):
  return cv2.GaussianBlur(img, (ksize, ksize), 0)


def dilate(img, size=1, shape=cv2.MORPH_ELLIPSE, iterations=1):
  kernel = cv2.getStructuringElement(
    shape,
    (2 * size + 1, 2 * size + 1),
    (size, size),
  )
  img = cv2.dilate(img, kernel, iterations=iterations)
  return img


def b53d2(img):
  return cv2.GaussianBlur(dilate(cv2.GaussianBlur(img, (5, 5), 0), iterations=2), (3, 3), 0)


def augment_folder(folder, func, tag, jpg_quality=100):
  folder = Path(folder)
  save_folder = folder.parent / (folder.stem + "_" + tag)
  save_folder.mkdir(exist_ok=True)
  for p in folder.glob("*.jpg"):
    save_path = str(save_folder / p.name)
    img = func(cv2.imread(str(p)))
    cv2.imwrite(save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])


def elastic(img):
  e = ElasticTransform(alpha=img.shape[0] / 5, sigma=5, alpha_affine=img.shape[0] * 0.01, always_apply=True)
  distort = e.apply(img)
  return img
