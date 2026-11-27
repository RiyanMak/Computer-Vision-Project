import argparse
import csv
import os

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="clip_gt.csv from cvat_conversion.py")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--label", help="cvat label of the target")
    parser.add_argument("--data", default="data")
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args()

    with open(args.gt) as f:
        rows = [r for r in csv.DictReader(f) if args.label in (None, r["label"])]
    # first track is the target
    rows = [r for r in rows if r["track_id"] == rows[0]["track_id"]]
    boxes = {int(r["frame"]): r for r in rows}
    total = int(cv2.VideoCapture(os.path.join(args.data, args.clip + ".mp4")).get(cv2.CAP_PROP_FRAME_COUNT))

    with open(os.path.join(args.data, args.clip + ".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x1", "y1", "x2", "y2", "absent"])
        for i in range(0, total, args.every):
            r = boxes.get(i)
            if r is not None and r["scored"] == "0":
                continue
            # outside means hidden or out of view
            if r is None or r["outside"] == "1":
                writer.writerow([i, "", "", "", "", 1])
            else:
                writer.writerow([i] + [round(float(r[k])) for k in ("x1", "y1", "x2", "y2")] + [0])

    path = os.path.join(args.data, "clips.csv")
    with open(path) as f:
        clips = list(csv.DictReader(f))
    clips = [c for c in clips if c["clip"] != args.clip] + [{"clip": args.clip, "prompt": args.prompt}]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "prompt"])
        writer.writeheader()
        writer.writerows(clips)


if __name__ == "__main__":
    main()
