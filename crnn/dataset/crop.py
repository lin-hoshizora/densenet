from pathlib import Path
import json
import cv2


def load_via(via_label: str) -> list:
  """
  Load VIA labels from raw JSON file
  """
  with open(via_label) as f:
    labels = json.load(f)
  labels = list(labels["_via_img_metadata"].values())
  return labels


def via_crop(via_label: str,
             ori_folder: Path,
             aug_folders: list,
             ori_save: Path,
             aug_save: list,
             char2id: dict,
             char_replace: dict,
             jpg_quality: int = 100) -> None:
  """
  Crop out textlines based on VIA labels
  """
  labels = load_via(via_label)
  gts = []
  for img_folder, save_folder in zip([ori_folder] + aug_folders, [ori_save] + aug_save):
    save_folder.mkdir(exist_ok=True)
    print(f"Processing {str(img_folder)} ")

    for rec_idx, r in enumerate(labels):
      print(f"\r{rec_idx + 1} / {len(labels)}", end="\n" if rec_idx == len(labels) - 1 else "")
      img = cv2.imread(str(img_folder / r["filename"]))
      if img is None:
        print(f"CANNOT read {r['filename']}")
        continue
      for r_idx, region in enumerate(r["regions"]):
        s = region["shape_attributes"]
        info = region["region_attributes"]
        if "text" not in info: continue
        if s["name"] == "polygon":
          xs = s["all_points_x"]
          ys = s["all_points_y"]
          x, y = min(xs), min(ys)
          w, h = max(xs) - x, max(ys) - y
        else:
          x, y, w, h = s["x"], s["y"], s["width"], s["height"]
        cx, cy = x + w / 2, y + h / 2
        new_h, new_w = h * 1.15, w + h * 0.15
        x1 = max(int(cx - new_w / 2), 0)
        y1 = max(int(cy - new_h / 2), 0)
        x2 = int(cx + new_w / 2)
        y2 = int(cy + new_h / 2)
        chip = img[y1:y2, x1:x2]
        save_path = save_folder / f'{r["filename"].split(".")[0]}_{r_idx}.jpg'
        cv2.imwrite(str(save_path), chip, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        if img_folder != ori_folder: continue
        try:
          codes = [char2id[char_replace[c]] if c in char_replace else char2id[c] for c in info["text"].lower()]
        except KeyError as e:
          print("\nInvalid char", str(e), "\n")
          continue
        gts.append([save_path.name, codes])
  return gts

