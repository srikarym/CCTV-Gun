from pathlib import Path
import matplotlib.pyplot as plt
import PIL 
import shutil
import json

src_dir = Path("MGD/MGD2020/JPEGImages/")
dst_dir = Path("data/mgd/images")
all_img_dir = Path("data/all_images")

dst_dir.mkdir(exist_ok=True, parents=True)
all_img_dir.mkdir(exist_ok=True, parents=True)
with open("data/mgd/annotation_detection/annotations_all.json", "r") as f:
    ann = json.load(f)

img_list = set([img['file_name'] for img in ann['images']])

for path in src_dir.iterdir():
    if path.name not in img_list:
        continue
    try:
        plt.imread(path)
        shutil.copyfile(path, dst_dir / path.name)
        shutil.copyfile(path, all_img_dir / path.name)

    except PIL.UnidentifiedImageError:
        # Some images in the original source have a byte error

        with path.open("rb") as f:
            temp = f.read()
        for dest_path in [dst_dir / path.name, all_img_dir / path.name]:
            with dest_path.open("wb") as f:
                f.write(temp.lstrip(b"\x00"))