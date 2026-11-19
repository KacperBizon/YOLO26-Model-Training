# PCB Component Placement Classifier (`DNP` vs. `PLACED`)

Binary image classification pipeline built on **Ultralytics YOLO** to classify printed circuit board (PCB) component regions as either **DNP** (*Do Not Populate*) or **PLACED**.

Unlike standard YOLO classification pipelines that apply center crops—which can cut off solder pads, silkscreen markings, or component leads near image edges—this script enforces **full-frame letterboxing** across training, validation, and testing.

---

## Key Features

- **Full-Frame Letterboxing:** Overrides Ultralytics' default center-crop transform with `ClassifyLetterBox` so the entire component footprint and surrounding pads remain visible.
- **Leak-Free Evaluation Protocol:**
  1. Trains exclusively on `dataset/train`.
  2. Selects the best model weights (`best.pt`) and tunes the optimal `PLACED` decision threshold on `dataset/val` to maximize **balanced accuracy**.
  3. Freezes the validation-selected threshold and runs a single final evaluation on `dataset/test`.
- **Nested Folder Support:** Recursively discovers images inside subdirectories under each class folder during validation and testing.
- **Calibrated Probability Extraction:** Extracts single-softmax probabilities directly from the YOLO classification head for reliable threshold selection and downstream deployment.

---

## Project & Dataset Structure

Organize your dataset into `train`, `val`, and `test` splits, each containing `DNP` and `PLACED` directories:

```text
project_root/
├── train.py
├── outputs_yolo/
│   └── yolo_dnp_placed_best.pt # Generated after training
└── dataset/
    ├── train/
    │   ├── DNP/
    │   └── PLACED/
    ├── val/
    │   ├── DNP/
    │   └── PLACED/
    └── test/
        ├── DNP/
        └── PLACED/