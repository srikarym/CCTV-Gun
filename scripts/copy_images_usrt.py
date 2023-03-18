from pathlib import Path
import shutil
import json

src_dir = Path("Images")
dst_dir = Path("data/usrt/images")
all_img_dir = Path("data/all_images")

dst_dir.mkdir(exist_ok=True, parents=True)
all_img_dir.mkdir(exist_ok=True, parents=True)
with open("data/usrt/annotation_detection/annotations_all.json", "r") as f:
    ann = json.load(f)

img_list = set([img['file_name'] for img in ann['images']])

for path in src_dir.iterdir():
    if path.name not in img_list:
        continue
    shutil.copyfile(path, dst_dir / path.name)
    shutil.copyfile(path, all_img_dir / path.name)