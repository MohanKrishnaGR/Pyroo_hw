import logging
import random
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class FireDatasetAugmentor:
    """
    Implements targeted augmentations for wildfire detection to improve
    robustness under diverse environmental conditions.
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)

    def add_synthetic_smoke(self, image):
        """
        Simulates smoke by overlaying semi-transparent Gaussian blobs.
        Targets 'Fire Condition' robustness.
        """
        h, w = image.shape[:2]
        smoke_layer = np.zeros((h, w, 3), dtype=np.uint8)

        for _ in range(random.randint(3, 8)):
            center = (random.randint(0, w), random.randint(0, h))
            axes = (random.randint(w // 10, w // 3), random.randint(h // 10, h // 3))
            angle = random.randint(0, 360)

            # Create a smoke-colored ellipse (grey/white)
            color = random.randint(180, 230)
            cv2.ellipse(
                smoke_layer, center, axes, angle, 0, 360, (color, color, color), -1
            )

        # Blur the smoke to make it realistic
        smoke_layer = cv2.GaussianBlur(smoke_layer, (101, 101), 0)

        # Alpha blend
        alpha = random.uniform(0.2, 0.5)
        augmented = cv2.addWeighted(image, 1.0, smoke_layer, alpha, 0)
        return augmented

    def simulate_low_light(self, image):
        """
        Simulates night-time or low-visibility conditions.
        Targets 'Multi-scenario' robustness.
        """
        gamma = random.uniform(0.3, 0.6)
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)

    def process_dataset(self, input_images_path, input_labels_path):
        """
        Iterates through the dataset and applies targeted augmentations.
        """
        image_files = list(Path(input_images_path).glob("*.jpg"))
        LOGGER.info(f"Processing {len(image_files)} images for augmentation...")

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            label_path = Path(input_labels_path) / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            # 1. Original
            self._save(img, img_path.stem, label_path)

            # 2. Smoke Augmentation
            smoke_img = self.add_synthetic_smoke(img)
            self._save(smoke_img, f"{img_path.stem}_smoke", label_path)

            # 3. Low Light Augmentation
            low_light_img = self.simulate_low_light(img)
            self._save(low_light_img, f"{img_path.stem}_lowlight", label_path)

    def _save(self, image, name, label_path):
        cv2.imwrite(str(self.output_dir / "images" / f"{name}.jpg"), image)
        # Copy original label (targeted augmentations here don't change bbox coordinates)
        with open(label_path, "r") as f:
            content = f.read()
        with open(self.output_dir / "labels" / f"{name}.txt", "w") as f:
            f.write(content)


if __name__ == "__main__":
    # Example usage for the 51GB dataset curation
    augmentor = FireDatasetAugmentor(output_dir="curated_dataset_v1")
    # augmentor.process_dataset("path/to/raw/images", "path/to/raw/labels")
    print("Augmentor initialized. Usage: augmentor.process_dataset(img_dir, label_dir)")
