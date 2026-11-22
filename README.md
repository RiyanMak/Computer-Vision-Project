# Text-Guided Object Tracking in Real-World Videos

Track an object in a video from a text description using Grounding DINO and SAM 2.

## Setup

```
pip install -r requirements.txt
```

Also needs `ffmpeg` on the path. Runs on CPU but a GPU is much faster.

## Run

```
python run.py --video clip.mp4 --prompt "the red backpack"
```

Writes `outputs/<clip>_tracked.mp4` and `outputs/<clip>_boxes.csv` (box, SAM 2 object score, lost and reinit flags per frame).

Add `--no-redetect` for the baseline without re-detection.

## Web app

```
python app.py
```

Upload a video, type a description and get the tracked video back.

## Data

Each clip is `data/<clip>.mp4` with boxes on sampled frames in `data/<clip>.csv` (`frame,x1,y1,x2,y2,absent`, absent is 1 when the target is occluded or out of view). Prompts go in `data/clips.csv`.

Label a recorded clip every 10 frames:

```
python label.py --video data/backpack.mp4 --out data/backpack.csv
```

Get a 200 frame LaSOT test clip with its description and boxes:

```
python lasot.py --seq mouse-1 --start 380
```

The LaSOT clips in `data/clips.csv` use these start frames: mouse-1 380, cup-1 0, bottle-18 440, mouse-8 80, dog-1 520, hat-1 1560, person-5 2100, umbrella-9 0, dog-19 420.

## Eval

```
python eval.py
```

Runs the baseline and the full system on every clip in `data/clips.csv` and writes `outputs/eval/results.csv`.

- iou: mean box IoU over annotated frames where the target is visible
- success: share of those frames with IoU >= 0.5
- fps: processing speed
- recovered: reappearances where the target is found again within 30 frames
- id_switches: times the output moves from the target to something else
- false_pos: annotated frames with a box that does not touch the target
