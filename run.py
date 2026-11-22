import argparse
import csv
import os
import time

import torch

from src.detect import Detector
from src.track import Tracker, is_lost, mask_to_box
from src.video import draw, read_frames, write_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--area-jump", type=float, default=2.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Detector(device)
    tracker = Tracker(device)
    frames, fps = read_frames(args.video)
    os.makedirs(args.out, exist_ok=True)

    rows, vis = [], []
    prev_area = 0
    start = time.time()
    for idx, frame in enumerate(frames):
        init_box = None
        # first frame picks the top dino box
        if idx == 0:
            boxes, _ = detector(frame, args.prompt)
            if len(boxes):
                init_box = boxes[0]
        mask, score = tracker.step(idx, frame, init_box)
        lost = is_lost(mask, score, prev_area, args.score_thresh, args.area_jump)
        prev_area = 0 if mask is None else int((mask > 0).sum())
        if lost:
            mask = None
        box = mask_to_box(mask)
        rows.append([idx, score, int(lost)] + (box or ["", "", "", ""]))
        vis.append(draw(frame, mask, box, lost))
    elapsed = time.time() - start

    name = os.path.splitext(os.path.basename(args.video))[0]
    write_video(os.path.join(args.out, f"{name}_tracked.mp4"), vis, fps)
    with open(os.path.join(args.out, f"{name}_boxes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score", "lost", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)
    print(f"{len(frames)} frames {len(frames) / elapsed:.2f} fps on {device}")


if __name__ == "__main__":
    main()
