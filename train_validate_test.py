import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from ultralytics import YOLO
from ultralytics.data.augment import ClassifyLetterBox
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import (
    ClassificationTrainer,
    ClassificationValidator,
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATASET = Path("dataset").resolve()
IMAGE_SIZE = 224
MODEL_NAME = "yolo26m-cls.pt"

EPOCHS = 150
BATCH_SIZE = 16
PATIENCE = 25
WORKERS = 0
SEED = 42

RUN_NAME = "pcb_component_classifier"
CHECKPOINT_PATH = Path("outputs_yolo/yolo_dnp_placed_best.pt").resolve()

CLASS_NAMES = ("DNP", "PLACED")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------
# CUSTOM YOLO DATASET
# --------------------------------------------------
class FullFrameDataset(ClassificationDataset):
    """Classification dataset using full-frame letterboxing, not center crops."""

    def __init__(self, root, args, augment=False, prefix=""):
        super().__init__(root, args, augment, prefix)

        letterbox = T.Compose(
            [
                np.asarray,
                ClassifyLetterBox(args.imgsz),
                T.ToPILImage(),
            ]
        )

        train_transforms = T.Compose(
            [
                letterbox,
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.08,
                            contrast=0.08,
                            saturation=0.03,
                        )
                    ],
                    p=0.5,
                ),
                T.ToTensor(),
                # Identity normalization; keep pixel values in [0, 1].
                T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ]
        )

        val_transforms = T.Compose(
            [
                letterbox,
                T.ToTensor(),
                T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ]
        )

        self.torch_transforms = train_transforms if augment else val_transforms


# --------------------------------------------------
# CUSTOM TRAINER / VALIDATOR
# --------------------------------------------------
class FullFrameTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return FullFrameDataset(
            root=img_path,
            args=self.args,
            augment=(mode == "train"),
            prefix="train" if mode == "train" else self.args.split,
            names=self.data["names"],
        )


class FullFrameValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        return FullFrameDataset(
            root=img_path,
            args=self.args,
            augment=False,
            prefix=self.args.split,
            names=self.names,
        )


# --------------------------------------------------
# EVALUATION DATASET
# --------------------------------------------------
class NestedBinarySplit(Dataset):
    """Reads images recursively under split/{DNP,PLACED}/... ."""

    def __init__(self, root, class_to_idx, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {self.root}")

        for class_name in CLASS_NAMES:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            image_paths = sorted(
                path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            self.samples.extend((path, class_to_idx[class_name]) for path in image_paths)

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

        counts = {
            class_name: sum(label == class_to_idx[class_name] for _, label in self.samples)
            for class_name in CLASS_NAMES
        }
        missing = [name for name, count in counts.items() if count == 0]
        if missing:
            raise RuntimeError(f"No images for {missing} in split {self.root}")

        print(f"{self.root.name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def make_eval_transform():
    """Use the same deterministic letterbox preprocessing as validation."""
    return T.Compose(
        [
            np.asarray,
            ClassifyLetterBox(IMAGE_SIZE),
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ]
    )


def get_model_class_indices(yolo_model):
    names = yolo_model.names
    if isinstance(names, dict):
        name_to_idx = {str(name): int(index) for index, name in names.items()}
    else:
        name_to_idx = {str(name): index for index, name in enumerate(names)}

    missing = [name for name in CLASS_NAMES if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Checkpoint classes {names} do not contain expected classes {missing}.")

    return name_to_idx["DNP"], name_to_idx["PLACED"]


def collect_split_predictions(yolo_model, split_dir, device, class_to_idx):
    dataset = NestedBinarySplit(
        root=split_dir,
        class_to_idx=class_to_idx,
        transform=make_eval_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    network = yolo_model.model.to(device)
    network.eval()

    all_labels = []
    all_placed_probabilities = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            output = network(images)

            if isinstance(output, (tuple, list)):
                # In eval mode, Ultralytics Classify returns (softmax_probs, raw_logits).
                if len(output) > 1:
                    probabilities = torch.softmax(output[1].float(), dim=1)
                else:
                    probabilities = output[0].float()
            elif isinstance(output, dict):
                logits = output.get("logits", output.get("output"))
                probabilities = torch.softmax(logits.float(), dim=1)
            elif isinstance(output, torch.Tensor):
                probabilities = torch.softmax(output.float(), dim=1)
            else:
                raise RuntimeError(f"Unexpected YOLO classification output type: {type(output)}")

            if probabilities.ndim != 2:
                raise RuntimeError("Unexpected YOLO classification output. Expected a [batch, classes] tensor.")

            placed_probabilities = probabilities[:, class_to_idx["PLACED"]]

            all_labels.extend(labels.numpy().tolist())
            all_placed_probabilities.extend(placed_probabilities.cpu().numpy().tolist())

    return np.asarray(all_labels, dtype=np.int64), np.asarray(all_placed_probabilities, dtype=np.float64)


# --------------------------------------------------
# THRESHOLD SELECTION / METRICS
# --------------------------------------------------
def select_placed_threshold(y_true, placed_probabilities, dnp_idx, placed_idx):
    """Choose the PLACED threshold maximizing validation balanced accuracy."""
    unique_scores = np.unique(placed_probabilities)

    # Every possible binary prediction boundary is represented by either an
    # observed score or a midpoint between adjacent observed scores.
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    all_negative_threshold = np.nextafter(unique_scores[-1], np.inf)
    candidates = np.unique(np.concatenate(([0.0], unique_scores, midpoints, [all_negative_threshold])))

    scores = []
    for threshold in candidates:
        y_pred = np.where(placed_probabilities >= threshold, placed_idx, dnp_idx)
        scores.append(balanced_accuracy_score(y_true, y_pred))

    scores = np.asarray(scores)
    best_score = scores.max()
    best_candidates = candidates[np.isclose(scores, best_score)]

    # Stable tie-break: prefer the threshold closest to 0.5; if still tied,
    # prefer the higher threshold (fewer false-positive PLACED predictions).
    order = np.lexsort((-best_candidates, np.abs(best_candidates - 0.5)))
    return float(best_candidates[order[0]])


def print_metrics(split_name, y_true, placed_probabilities, threshold, dnp_idx, placed_idx):
    y_pred = np.where(placed_probabilities >= threshold, placed_idx, dnp_idx)
    label_order = [dnp_idx, placed_idx]

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=["DNP", "PLACED"],
        digits=4,
        zero_division=0,
    )

    print(f"\n{split_name} results (PLACED threshold = {threshold:.4f})")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print("Confusion matrix (rows=true, columns=predicted; DNP, PLACED):")
    print(cm)
    print(report.rstrip())


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_dir = DATASET / "train"
    val_dir = DATASET / "val"
    test_dir = DATASET / "test"

    for required_dir in (train_dir, val_dir, test_dir):
        if not required_dir.is_dir():
            raise FileNotFoundError(
                f"Required split directory is missing: {required_dir}\n"
                "Expected dataset/train, dataset/val, and dataset/test."
            )

    for split_dir in (train_dir, val_dir, test_dir):
        for class_name in CLASS_NAMES:
            if not (split_dir / class_name).is_dir():
                raise FileNotFoundError(f"Missing class directory: {split_dir / class_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training classes: DNP and PLACED")
    print("Validation split: dataset/val (used for model selection and threshold selection)")
    print("Test split: dataset/test (not accessed until final evaluation)\n")

    # -------------------- TRAIN --------------------
    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    print("Starting training on dataset/train only...")
    model.train(
        data=str(DATASET),
        trainer=FullFrameTrainer,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        dropout=0.15,
        workers=WORKERS,
        name=RUN_NAME,
        seed=SEED,
        split="val",
        val=True,
    )

    print("\nTraining completed.")

    # -------------------- SAVE / RELOAD BEST CHECKPOINT --------------------
    best_weights = Path(model.trainer.best)
    if not best_weights.is_file():
        best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Could not find the best YOLO checkpoint: {best_weights}")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, CHECKPOINT_PATH)

    # Reload the selected checkpoint so all final metrics use best.pt.
    model = YOLO(str(CHECKPOINT_PATH))
    dnp_idx, placed_idx = get_model_class_indices(model)
    class_to_idx = {"DNP": dnp_idx, "PLACED": placed_idx}

    # -------------------- VALIDATE / SELECT THRESHOLD --------------------
    print("\nEvaluating validation split and selecting PLACED threshold...")
    val_labels, val_placed_probs = collect_split_predictions(model, val_dir, device, class_to_idx)
    placed_threshold = select_placed_threshold(val_labels, val_placed_probs, dnp_idx, placed_idx)
    print_metrics(
        "VALIDATION",
        val_labels,
        val_placed_probs,
        placed_threshold,
        dnp_idx,
        placed_idx,
    )
    print(f"\nSaved checkpoint to: {CHECKPOINT_PATH}")

    # -------------------- FINAL TEST (THRESHOLD IS FROZEN) --------------------
    print("\nEvaluating test split with the validation-selected threshold...")
    test_labels, test_placed_probs = collect_split_predictions(model, test_dir, device, class_to_idx)
    print_metrics(
        "TEST",
        test_labels,
        test_placed_probs,
        placed_threshold,
        dnp_idx,
        placed_idx,
    )


if __name__ == "__main__":
    main()
