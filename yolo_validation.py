"""YOLOv8m detection validation.

The main experiments use object counts from the official BDD100K
annotations (see build_features.py). This script covers the detection
validation phase described in Section 3.3 of the dissertation: it runs
YOLOv8m zero-shot (COCO weights) on a sample of frames so detector counts
can be compared against the annotation counts.

Requires the ultralytics package and downloaded images:
    python yolo_validation.py --img_dir path/to/images --limit 500
"""

import argparse
import json
import os

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True,
                    help='directory of .jpg frames to run detection on')
    ap.add_argument('--limit', type=int, default=500,
                    help='number of frames to sample')
    ap.add_argument('--output', default='results/yolo_validation.json')
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO('yolov8m.pt')

    # COCO class ids for the categories used in PCU scoring
    coco_to_cat = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                   5: 'bus', 7: 'truck'}

    frames = sorted(f for f in os.listdir(args.img_dir)
                    if f.lower().endswith('.jpg'))[:args.limit]
    results = {}
    for fname in frames:
        det = model(os.path.join(args.img_dir, fname),
                    conf=0.25, iou=0.45, verbose=False)[0]
        counts = {}
        for box in det.boxes:
            cat = coco_to_cat.get(int(box.cls[0]))
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        pcu = sum(counts.get(c, 0) * w
                  for c, w in config.PCU_WEIGHTS.items())
        results[fname] = {'counts': counts, 'pcu': pcu}

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Detected {len(results)} frames, saved to {args.output}')


if __name__ == '__main__':
    main()
