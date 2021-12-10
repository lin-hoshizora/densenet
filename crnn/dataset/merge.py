import cv2
import numpy as np


def combine_imgs(img_folder, filenames):
  imgs = [cv2.imread(str(img_folder / filename)) for filename in filenames]
  h = max([img.shape[0] for img in imgs])
  imgs = [cv2.resize(img, (int(h / img.shape[0] * img.shape[1]), h), cv2.INTER_LANCZOS4) for img in imgs]
  img = np.concatenate(imgs, axis=1)
  return img


def merge(gts, max_len, img_folders, save_folders, ori_folder, jpg_quality=100):
  merged_gts = []
  cur_imgs = []
  cur_lbs = []
  img_groups = []
  merged_labels = []
  for filename, label in gts:
    if len(cur_lbs) + len(label) >= max_len:
      if not cur_imgs: continue
      img_groups.append(cur_imgs)
      merged_labels.append(cur_lbs)
      cur_imgs.clear()
      cur_lbs.clear()
      continue
    cur_imgs.append(filename)
    cur_lbs += label
  for idx, (filenames, labels) in enumerate(zip(img_groups, merged_labels)):
    print(f"\r{idx + 1} / {len(img_groups)}", end="")
    for img_folder, save_folder in zip(img_folders, save_folders):
      save_folder.mkdir(exist_ok=True)
      img = combine_imgs(img_folder, filenames)
      cv2.imwrite(str(save_folder / f"{save_cnt}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
      save_cnt += 1
      if img_folder != ori_folder: continue
      merged_gts.append([f"{idx}.jpg", labels])
  print()
  return merged_gts
