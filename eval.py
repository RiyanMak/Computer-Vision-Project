import argparse
import csv
import os
import time

import torch

from run import save, track
from src.detect import Detector
from src.embed import Embedder
from src.track import Tracker
from src.video import read_frames


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(path):
    with open(path) as f:
        return [(int(r["frame"]), None if r["absent"] == "1" else [float(r[k]) for k in ("x1", "y1", "x2", "y2")])
                for r in csv.DictReader(f)]


def score(gt, rows, window=30):
    pred = {r[0]: (r[4:] if r[4] != "" else None) for r in rows}
    ious, fp, switches, recovered, events = [], 0, 0, 0, 0
    on_target, prev_absent = None, False
    for frame, box in gt:
        p = pred[frame]
        o = iou(p, box) if p is not None and box is not None else 0.0
        if box is not None:
            ious.append(o)
        # box on something that is not the target
        wrong = p is not None and (box is None or o == 0)
        fp += wrong
        if p is not None:
            if on_target and wrong:
                switches += 1
            on_target = not wrong
        # reacquired within window of reappearing
        if box is not None and prev_absent:
            events += 1
            later = [g for g in gt if frame <= g[0] <= frame + window and g[1] is not None]
            recovered += any(pred[f] is not None and iou(pred[f], b) >= 0.5 for f, b in later)
        prev_absent = box is None
    return {
        "iou": sum(ious) / len(ious),
        "success": sum(o >= 0.5 for o in ious) / len(ious),
        "recovered": f"{recovered}/{events}",
        "id_switches": switches,
        "false_pos": fp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="outputs/eval")
    parser.add_argument("--head", help="trained appearance head")
    parser.add_argument("--sim-thresh", type=float, default=0.85)
    parser.add_argument("--methods", default="baseline,ours")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Detector(device)
    tracker = Tracker(device)
    embedder = Embedder(device, head=args.head)

    with open(os.path.join(args.data, "clips.csv")) as f:
        clips = list(csv.DictReader(f))
    results = []
    for clip in clips:
        frames, fps = read_frames(os.path.join(args.data, clip["clip"] + ".mp4"))
        gt = load_gt(os.path.join(args.data, clip["clip"] + ".csv"))
        for method in args.methods.split(","):
            redetect = method != "baseline"
            start = time.time()
            rows, vis = track(frames, clip["prompt"], detector, tracker, embedder, sim_thresh=args.sim_thresh,
                              redetect=redetect)
            res = {"clip": clip["clip"], "method": method, "fps": len(frames) / (time.time() - start)}
            res.update(score(gt, rows))
            save(rows, vis, fps, os.path.join(args.out, method), clip["clip"])
            results.append(res)
            print(res, flush=True)

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
