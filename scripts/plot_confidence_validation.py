# scripts/plot_confidence_validation.py

from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


def load_data_yaml(data_yaml_path):
    data_yaml_path = Path(data_yaml_path)

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data.get("path", data_yaml_path.parent))

    val_path = Path(data["val"])
    if not val_path.is_absolute():
        val_path = dataset_root / val_path

    names = data.get("names", None)

    return dataset_root, val_path, names


def find_images(val_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if val_path.is_file():
        with open(val_path, "r", encoding="utf-8") as f:
            image_paths = [Path(line.strip()) for line in f if line.strip()]
        return image_paths

    image_paths = []
    for ext in image_extensions:
        image_paths.extend(val_path.rglob(f"*{ext}"))

    return sorted(image_paths)


def label_path_from_image_path(image_path):
    """
    Converts:
        data/vehicle_dataset/images/val/img001.jpg

    To:
        data/vehicle_dataset/labels/val/img001.txt
    """
    parts = list(image_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        return label_path

    return image_path.with_suffix(".txt")


def load_yolo_labels(label_path, image_width, image_height):
    """
    YOLO label format:
        class x_center y_center width height

    All coordinates are normalized.
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()

            if len(values) < 5:
                continue

            cls = int(float(values[0]))
            x_center = float(values[1]) * image_width
            y_center = float(values[2]) * image_height
            width = float(values[3]) * image_width
            height = float(values[4]) * image_height

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            boxes.append({
                "class_id": cls,
                "box": np.array([x1, y1, x2, y2], dtype=float),
                "matched": False
            })

    return boxes


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def match_predictions_to_ground_truth(predictions, ground_truth_boxes, iou_threshold):
    """
    A prediction is correct if:
        1. It has the same class as a ground-truth box
        2. IoU >= threshold
        3. That ground-truth box has not already been matched

    Each ground-truth object can only be matched once.
    """
    rows = []

    predictions = sorted(predictions, key=lambda p: p["confidence"], reverse=True)

    for pred in predictions:
        pred_box = pred["box"]
        pred_class = pred["class_id"]

        best_iou = 0.0
        best_gt_index = None

        for i, gt in enumerate(ground_truth_boxes):
            if gt["matched"]:
                continue

            if gt["class_id"] != pred_class:
                continue

            iou = calculate_iou(pred_box, gt["box"])

            if iou > best_iou:
                best_iou = iou
                best_gt_index = i

        is_correct = best_iou >= iou_threshold and best_gt_index is not None

        if is_correct:
            ground_truth_boxes[best_gt_index]["matched"] = True

        rows.append({
            "confidence": pred["confidence"],
            "class_id": pred_class,
            "iou": best_iou,
            "is_correct": is_correct
        })

    false_negatives = sum(1 for gt in ground_truth_boxes if not gt["matched"])

    return rows, false_negatives


def bin_results(results_df, bin_width):
    bins = np.arange(0, 1 + bin_width, bin_width)

    results_df["confidence_bin"] = pd.cut(
        results_df["confidence"],
        bins=bins,
        include_lowest=True,
        right=False
    )

    grouped = results_df.groupby("confidence_bin", observed=False).agg(
        correct=("is_correct", lambda x: int(x.sum())),
        incorrect=("is_correct", lambda x: int((~x).sum())),
        total=("is_correct", "count")
    ).reset_index()

    grouped["confidence_midpoint"] = grouped["confidence_bin"].apply(
        lambda interval: float((interval.left + interval.right) / 2)
    )

    grouped["accuracy_in_bin"] = grouped.apply(
        lambda row: row["correct"] / row["total"] if row["total"] > 0 else 0,
        axis=1
    )

    return grouped


def plot_confidence_counts(summary_df, output_plot_path, title):
    plt.figure(figsize=(10, 6))

    plt.plot(
        summary_df["confidence_midpoint"],
        summary_df["correct"],
        marker="o",
        label="Correct detections"
    )

    plt.plot(
        summary_df["confidence_midpoint"],
        summary_df["incorrect"],
        marker="o",
        label="Incorrect detections"
    )

    plt.xlabel("Confidence level")
    plt.ylabel("Number of detections")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot YOLO validation confidence vs correct/incorrect detections."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO model weights, for example runs/detect/.../weights/best.pt"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML file."
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold used to count a detection as correct. Default: 0.5"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Minimum prediction confidence. Use a low value to see the full confidence curve. Default: 0.001"
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.05,
        help="Confidence bin width. Default: 0.05"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size. Default: 640"
    )

    parser.add_argument(
        "--output-dir",
        default="runs/validation_confidence_plot",
        help="Directory where plot and CSV files will be saved."
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    data_yaml_path = Path(args.data)
    output_dir = Path(args.output_dir)

    dataset_root, val_path, names = load_data_yaml(data_yaml_path)
    image_paths = find_images(val_path)

    if len(image_paths) == 0:
        raise RuntimeError(f"No validation images found at: {val_path}")

    print(f"Loaded model: {model_path}")
    print(f"Validation path: {val_path}")
    print(f"Found {len(image_paths)} validation images")

    model = YOLO(str(model_path))

    all_rows = []
    total_false_negatives = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: could not read image: {image_path}")
            continue

        image_height, image_width = image.shape[:2]

        label_path = label_path_from_image_path(image_path)
        ground_truth_boxes = load_yolo_labels(label_path, image_width, image_height)

        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False
        )

        predictions = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(float)
                confidence = float(box.conf[0].cpu().item())
                class_id = int(box.cls[0].cpu().item())

                predictions.append({
                    "box": xyxy,
                    "confidence": confidence,
                    "class_id": class_id
                })

        image_rows, false_negatives = match_predictions_to_ground_truth(
            predictions,
            ground_truth_boxes,
            args.iou
        )

        total_false_negatives += false_negatives

        for row in image_rows:
            row["image"] = str(image_path)
            row["label_file"] = str(label_path)
            if names is not None:
                if isinstance(names, dict):
                    row["class_name"] = names.get(row["class_id"], str(row["class_id"]))
                elif isinstance(names, list):
                    row["class_name"] = names[row["class_id"]]
            all_rows.append(row)

    if len(all_rows) == 0:
        raise RuntimeError("No predictions were produced. Try lowering --conf or checking your model path.")

    results_df = pd.DataFrame(all_rows)
    summary_df = bin_results(results_df, args.bin_width)

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "confidence_validation_raw_predictions.csv"
    summary_csv_path = output_dir / "confidence_validation_summary.csv"
    output_plot_path = output_dir / "confidence_vs_correct_incorrect.png"

    results_df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    title = f"Validation Detections by Confidence, IoU ≥ {args.iou}"
    plot_confidence_counts(summary_df, output_plot_path, title)

    total_correct = int(results_df["is_correct"].sum())
    total_incorrect = int((~results_df["is_correct"]).sum())
    total_predictions = len(results_df)

    print()
    print("Done.")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct detections: {total_correct}")
    print(f"Incorrect detections: {total_incorrect}")
    print(f"False negatives / missed ground-truth objects: {total_false_negatives}")
    print()
    print(f"Saved raw prediction CSV: {raw_csv_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved plot: {output_plot_path}")


if __name__ == "__main__":
    main()