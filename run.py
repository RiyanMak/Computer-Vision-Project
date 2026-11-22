import argparse
import csv
import os
import time

import torch

from src.detect import Detector
from src.track import Tracker, mask_to_box
from src.video import draw, read_frames, write_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Detector(device)
    tracker = Tracker(device)
    frames, fps = read_frames(args.video)
    os.makedirs(args.out, exist_ok=True)

    rows, vis = [], []
    start = time.time()
    for idx, frame in enumerate(frames):
        init_box = None
        # first frame picks the top dino box
        if idx == 0:
            boxes, _ = detector(frame, args.prompt)
            if len(boxes):
                init_box = boxes[0]
        mask, score = tracker.step(idx, frame, init_box)
        box = mask_to_box(mask)
        rows.append([idx, score] + (box or ["", "", "", ""]))
        vis.append(draw(frame, mask, box))
    elapsed = time.time() - start

    name = os.path.splitext(os.path.basename(args.video))[0]
    write_video(os.path.join(args.out, f"{name}_tracked.mp4"), vis, fps)
    with open(os.path.join(args.out, f"{name}_boxes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)
    print(f"{len(frames)} frames {len(frames) / elapsed:.2f} fps on {device}")


if __name__ == "__main__":
    main()
