from pathlib import Path

DATASET_DIR = Path("data/vehicle_dataset")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def create_empty_labels(split: str) -> None:
    images_dir = DATASET_DIR / "images" / split
    labels_dir = DATASET_DIR / "labels" / split

    if not images_dir.exists():
        print(f"Skipping missing images folder: {images_dir}")
        return

    labels_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    checked = 0

    for img_path in images_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        checked += 1
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            label_path.write_text("", encoding="utf-8")
            created += 1

    print(f"{split}: checked {checked} images, created {created} empty label files")

def main():
    for split in ("train", "val"):
        create_empty_labels(split)

if __name__ == "__main__":
    main()