import argparse
import csv

import cv2

from src.video import read_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()

    frames, _ = read_frames(args.video)
    rows = []
    for idx in range(0, len(frames), args.every):
        # press c if target not visible
        x, y, w, h = cv2.selectROI(f"frame {idx}", cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
        cv2.destroyAllWindows()
        if w == 0 or h == 0:
            rows.append([idx, "", "", "", "", 1])
        else:
            rows.append([idx, x, y, x + w, y + h, 0])
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x1", "y1", "x2", "y2", "absent"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
