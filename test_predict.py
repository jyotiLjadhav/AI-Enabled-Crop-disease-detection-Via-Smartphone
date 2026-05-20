import argparse
import os
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

DEFAULT_MODEL_PATH = "model.h5"
DEFAULT_DATASET_ROOT = Path("Dataset") / "Test" / "Test"
DEFAULT_CLASSES = ["Healthy", "Powdery", "Rust"]


def _guess_target_size(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        height, width = int(shape[1]), int(shape[2])
        if height > 0 and width > 0:
            return (height, width)
    except Exception:
        pass
    return (224, 224)


def predict_one(model, image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img).astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]
    return pred

def _first_image_in_dir(dir_path: Path):
    if not dir_path.exists():
        return None
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"):
        matches = sorted(dir_path.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Quick sanity-check for model inference.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to model file (default: model.h5)")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image path to run prediction on (repeatable). If omitted, uses one sample per class from Dataset/Test/Test/.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    model = load_model(args.model)
    target_size = _guess_target_size(model)

    images = [Path(p) for p in args.image]
    if not images:
        for class_name in DEFAULT_CLASSES:
            candidate = _first_image_in_dir(DEFAULT_DATASET_ROOT / class_name)
            if candidate is not None:
                images.append(candidate)

    if not images:
        raise SystemExit(
            "No images provided and no samples found under Dataset/Test/Test/<ClassName>/. "
            "Pass one or more --image paths."
        )

    for image_path in images:
        if not image_path.exists():
            print(f"{image_path}: NOT FOUND")
            continue
        pred = predict_one(model, str(image_path), target_size=target_size)
        print(f"{image_path}: {pred}")


if __name__ == "__main__":
    main()
