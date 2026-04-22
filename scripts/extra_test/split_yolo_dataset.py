from pathlib import Path
import shutil
from collections import defaultdict

DATASET_DIR = Path("data/vehicle_dataset")
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
IMAGES_VAL = DATASET_DIR / "images" / "val"
LABELS_VAL = DATASET_DIR / "labels" / "val"

VAL_RATIO = 0.2  # about 20% of images, but split by video prefix

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

IMAGES_VAL.mkdir(parents=True, exist_ok=True)
LABELS_VAL.mkdir(parents=True, exist_ok=True)


def get_video_prefix(filename_stem: str) -> str:
    """
    Example:
        GH030092_frame_000001 -> GH030092
    """
    if "_frame_" in filename_stem:
        return filename_stem.split("_frame_")[0]
    return filename_stem


def main():
    image_files = sorted(
        [p for p in IMAGES_TRAIN.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )

    if not image_files:
        print(f"No training images found in {IMAGES_TRAIN}")
        return

    groups = defaultdict(list)
    for img_path in image_files:
        prefix = get_video_prefix(img_path.stem)
        groups[prefix].append(img_path)

    group_counts = {prefix: len(files) for prefix, files in groups.items()}
    total_images = sum(group_counts.values())
    target_val_images = max(1, int(round(total_images * VAL_RATIO)))

    # Pick full video groups for val until we get close to target
    sorted_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)

    selected_val_prefixes = []
    running_total = 0

    for prefix, count in sorted_groups:
        if running_total < target_val_images:
            selected_val_prefixes.append(prefix)
            running_total += count

    print(f"Total images: {total_images}")
    print(f"Target val images: {target_val_images}")
    print(f"Selected val groups: {selected_val_prefixes}")
    print(f"Actual val images: {running_total}")
    print()

    moved_images = 0
    moved_labels = 0
    missing_labels = []

    for prefix in selected_val_prefixes:
        for img_path in groups[prefix]:
            label_path = LABELS_TRAIN / f"{img_path.stem}.txt"

            new_img_path = IMAGES_VAL / img_path.name
            shutil.move(str(img_path), str(new_img_path))
            moved_images += 1

            if label_path.exists():
                new_label_path = LABELS_VAL / label_path.name
                shutil.move(str(label_path), str(new_label_path))
                moved_labels += 1
            else:
                missing_labels.append(label_path.name)

    print(f"Moved {moved_images} images to {IMAGES_VAL}")
    print(f"Moved {moved_labels} labels to {LABELS_VAL}")

    if missing_labels:
        print("\nWarning: missing label files for these images:")
        for name in missing_labels[:20]:
            print(f"  {name}")
        if len(missing_labels) > 20:
            print(f"  ... and {len(missing_labels) - 20} more")


if __name__ == "__main__":
    main()