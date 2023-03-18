from pathlib import Path
import json
import cv2
import numpy as np

def get_frames(path):
    vid = cv2.VideoCapture(path)

    frames = []

    while(vid.isOpened()):
      # Capture frame-by-frame
        ret, frame = vid.read()
        if ret:
            frames.append(frame)
        else:
            break
    return np.array(frames)

src_dir = Path("Anomaly-Videos-Part-3")
root = Path("data/ucf")
dst_dir = root / "images"
all_img_dir = Path("data/all_images")

dst_dir.mkdir(exist_ok=True, parents=True)
all_img_dir.mkdir(exist_ok=True, parents=True)
with open(root /  "annotation_detection/annotations_all.json", "r") as f:
    ann = json.load(f)

with open(root / "frames.json", "r") as f:
    frame_dict = json.load(f)

img_list = set([img['file_name'] for img in ann['images']])

for video_id, val in frame_dict.items():
    category = ''.join([i for i in video_id if i.isalpha()])

    frames = get_frames(f"Anomaly-Videos-Part-3/{category}/{video_id}_x264.mp4")

    for img_idx, frame_idx in val:
        img = frames[int(frame_idx)]

        cv2.imwrite(str(dst_dir / f"{img_idx:04d}.png"), img)
        cv2.imwrite(str(all_img_dir / f"{img_idx:04d}.png"), img)
