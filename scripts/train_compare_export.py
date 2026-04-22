from pathlib import Path
from ultralytics import YOLO

# ----------------------------
# Config
# ----------------------------
DATA_YAML = "data/vehicle_dataset/data.yaml"
BASE_MODEL = "yolov8n.pt"

IMG_SIZE = 640
EPOCHS = 50
BATCH = 16
DEVICE = 0          # use "cpu" if needed
WORKERS = 0         # often more stable on Windows
PROJECT_DIR = "runs/model_compare"
RUN_NAME = "finetuned"

# Set this to True for a fairer baseline comparison:
# - True: all classes treated as one "vehicle" class during validation
# - False: full 8-class validation
COMPARE_AS_SINGLE_CLASS = True


def print_metrics(title: str, metrics) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    # Main box metrics
    try:
        print(f"mAP50-95: {metrics.box.map:.6f}")
        print(f"mAP50:    {metrics.box.map50:.6f}")
        print(f"mAP75:    {metrics.box.map75:.6f}")
        print(f"Precision:{metrics.box.mp:.6f}")
        print(f"Recall:   {metrics.box.mr:.6f}")
    except Exception as e:
        print(f"Could not read box metrics: {e}")

    # Full results dict if available
    try:
        print("\nDetailed metrics:")
        for k, v in metrics.results_dict.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Could not read results_dict: {e}")

    # Per-class mAP list if available
    try:
        print("\nPer-class mAP50-95:")
        names = metrics.names if hasattr(metrics, "names") else None
        maps = metrics.box.maps
        if names is not None:
            for i, cls_map in enumerate(maps):
                cls_name = names.get(i, str(i)) if isinstance(names, dict) else str(i)
                print(f"  {cls_name}: {cls_map:.6f}")
        else:
            for i, cls_map in enumerate(maps):
                print(f"  class_{i}: {cls_map:.6f}")
    except Exception:
        pass


def main() -> None:
    val_kwargs = dict(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        single_cls=COMPARE_AS_SINGLE_CLASS,
        plots=False,
        verbose=True,
    )

    print("Loading baseline COCO-pretrained model...")
    baseline_model = YOLO(BASE_MODEL)

    print("\nValidating baseline model...")
    baseline_metrics = baseline_model.val(**val_kwargs)
    print_metrics("Baseline COCO-pretrained model", baseline_metrics)

    print("\nTraining fine-tuned model...")
    finetune_model = YOLO(BASE_MODEL)
    finetune_model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
    )

    best_path = Path(finetune_model.trainer.best)
    print(f"\nBest fine-tuned weights: {best_path}")

    print("\nLoading best fine-tuned model...")
    tuned_model = YOLO(str(best_path))

    print("\nValidating fine-tuned model...")
    tuned_metrics = tuned_model.val(**val_kwargs)
    print_metrics("Fine-tuned model", tuned_metrics)

    print(f"\nComparison finished.")
    print(f"Validation mode: {'single-class' if COMPARE_AS_SINGLE_CLASS else 'full multi-class'}")
    print(f"Best weights saved at: {best_path}")


if __name__ == "__main__":
    main()