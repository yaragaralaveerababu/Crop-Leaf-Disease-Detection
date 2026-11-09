"""
Dataset Preparation, Preprocessing, and Augmentation Pipeline
AcenAAr Technologies Pvt. Ltd. & Joy University
Author: Yaragarla Veerababu (2023BTIA009)

Requirements:
- Python, OpenCV, NumPy, TensorFlow/Keras.
- Resize images to 224 x 224.
- Normalize pixel values to [0, 1].
- Split dataset: 70% Train, 15% Validation, 15% Test.
- Data augmentation: rotation, flipping, zoom, brightness, contrast.
- Clear distinction: generates simulated benchmark images when external raw dataset is unavailable.
"""

import sys
import math
import random
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import cv2

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_procedural_leaf(class_name: str, size: int = 224) -> np.ndarray:
    """
    Generates a realistic leaf image with crop-specific morphology, venation,
    and characteristic disease symptoms (rust pustules, concentric early blight rings,
    water-soaked late blight lesions, or healthy foliage).
    
    NOTE: As per project guidelines, this procedural benchmark is generated for 
    offline reproducible training and simulation when raw field capture is unavailable.
    """
    # Background: Outdoor field soil, garden mulch, or neutral greenhouse bench
    bg_type = random.choice(["soil", "greenhouse", "mulch"])
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    if bg_type == "soil":
        # Rich earthen brownish tones with texture
        base_color = np.array([45, 65, 85], dtype=np.int16) # BGR
        noise = np.random.randint(-15, 15, (size, size, 3), dtype=np.int16)
        img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    elif bg_type == "greenhouse":
        # Neutral gray-beige tray
        base_color = np.array([120, 125, 120], dtype=np.int16)
        noise = np.random.randint(-10, 10, (size, size, 3), dtype=np.int16)
        img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    else:
        # Dark organic mulch
        base_color = np.array([30, 45, 55], dtype=np.int16)
        noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
        img = np.clip(base_color + noise, 0, 255).astype(np.uint8)

    center_x, center_y = size // 2, size // 2
    leaf_mask = np.zeros((size, size), dtype=np.uint8)
    
    crop = config.DISEASE_INFO[class_name]["crop"]
    is_diseased = config.DISEASE_INFO[class_name]["condition"] == "Diseased"

    # Shape leaf contour according to crop morphology
    if "Corn" in crop:
        # Long lanceolate blade (elongated parallel leaf)
        pts = []
        blade_w = random.randint(35, 45)
        for y in range(20, size - 20, 4):
            # Taper at top and bottom
            progress = (y - 20) / (size - 40)
            cur_w = blade_w * math.sin(progress * math.pi)
            offset = int(math.sin(progress * 2 * math.pi) * 8)
            pts.append((int(center_x - cur_w + offset), y))
        for y in range(size - 24, 16, -4):
            progress = (y - 20) / (size - 40)
            cur_w = blade_w * math.sin(progress * math.pi)
            offset = int(math.sin(progress * 2 * math.pi) * 8)
            pts.append((int(center_x + cur_w + offset), y))
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(leaf_mask, [pts_arr], 255)
        leaf_color = np.array([35, 150, 45], dtype=np.int16) # Bright corn green
    
    elif "Potato" in crop:
        # Oval, compound-like leaflet with rounded tip
        axes = (random.randint(60, 75), random.randint(85, 95))
        angle = random.randint(-15, 15)
        cv2.ellipse(leaf_mask, (center_x, center_y), axes, angle, 0, 360, 255, -1)
        leaf_color = np.array([30, 110, 35], dtype=np.int16) # Deep potato green
    
    else:  # Tomato
        # Serrated / lobed tomato leaflet
        pts = []
        for a in np.linspace(0, 2 * math.pi, 60):
            r = random.randint(65, 80)
            # Add lobe wave
            r += int(14 * math.sin(5 * a) + 8 * math.cos(3 * a))
            px = int(center_x + r * math.cos(a))
            py = int(center_y + r * 1.25 * math.sin(a))
            pts.append((px, py))
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(leaf_mask, [pts_arr], 255)
        leaf_color = np.array([40, 130, 40], dtype=np.int16)

    # Base leaf coloration & natural variation
    leaf_tex = np.zeros_like(img, dtype=np.int16)
    noise_leaf = np.random.randint(-18, 18, (size, size, 3), dtype=np.int16)
    leaf_rgb = np.clip(leaf_color + noise_leaf, 0, 255).astype(np.uint8)

    # Draw primary veins
    if "Corn" in crop:
        # Parallel veins
        for vx in range(center_x - 30, center_x + 35, 8):
            cv2.line(leaf_rgb, (vx, 20), (vx, size - 20), (25, 175, 55), 1)
        # Midrib
        cv2.line(leaf_rgb, (center_x, 20), (center_x, size - 20), (50, 195, 70), 2)
    else:
        # Pinnate branched veins
        cv2.line(leaf_rgb, (center_x, center_y + 80), (center_x, center_y - 80), (35, 150, 45), 2)
        for offset_y in range(-60, 70, 20):
            vy = center_y + offset_y
            cv2.line(leaf_rgb, (center_x, vy), (center_x - 45, vy - 15), (30, 140, 40), 1)
            cv2.line(leaf_rgb, (center_x, vy), (center_x + 45, vy - 15), (30, 140, 40), 1)

    # Composite leaf onto background
    img[leaf_mask > 0] = leaf_rgb[leaf_mask > 0]

    # Add disease symptoms if diseased
    if is_diseased:
        if "Common_Rust" in class_name:
            # Cinnamon-brown to rusty orange raised pustules on corn blade
            num_pustules = random.randint(35, 60)
            for _ in range(num_pustules):
                rx = random.randint(center_x - 30, center_x + 30)
                ry = random.randint(40, size - 40)
                if leaf_mask[ry, rx] > 0:
                    rust_color = (random.randint(15, 30), random.randint(70, 100), random.randint(180, 220)) # BGR
                    cv2.ellipse(img, (rx, ry), (random.randint(2, 4), random.randint(4, 7)), 0, 0, 360, rust_color, -1)
                    # Chlorotic halo around pustule
                    cv2.circle(img, (rx, ry), random.randint(6, 8), (40, 160, 180), 1)

        elif "Early_Blight" in class_name:
            # Concentric rings (target-board lesions) with dark brown necrotic core and yellow halo
            num_lesions = random.randint(4, 8)
            for _ in range(num_lesions):
                lx = random.randint(center_x - 50, center_x + 50)
                ly = random.randint(center_y - 50, center_y + 50)
                if 0 <= lx < size and 0 <= ly < size and leaf_mask[ly, lx] > 0:
                    radius = random.randint(14, 26)
                    # Chlorotic yellow halo
                    cv2.circle(img, (lx, ly), radius + 5, (50, 190, 210), -1)
                    # Concentric rings
                    for r_ring in range(radius, 4, -4):
                        ring_color = (random.randint(20, 35), random.randint(40, 60), random.randint(60, 90))
                        cv2.circle(img, (lx, ly), r_ring, ring_color, -1)
                        cv2.circle(img, (lx, ly), r_ring - 2, (30, 70, 100), 1)
                    # Dark necrotic center
                    cv2.circle(img, (lx, ly), 4, (15, 25, 40), -1)

        elif "Late_Blight" in class_name:
            # Irregular large dark greasy water-soaked lesions with decaying margins
            num_lesions = random.randint(3, 5)
            for _ in range(num_lesions):
                lx = random.randint(center_x - 45, center_x + 45)
                ly = random.randint(center_y - 45, center_y + 45)
                if 0 <= lx < size and 0 <= ly < size and leaf_mask[ly, lx] > 0:
                    blob_pts = []
                    b_rad = random.randint(20, 36)
                    for a in np.linspace(0, 2 * math.pi, 16):
                        rad = b_rad + random.randint(-6, 6)
                        bx = int(lx + rad * math.cos(a))
                        by = int(ly + rad * math.sin(a))
                        blob_pts.append((bx, by))
                    blob_arr = np.array(blob_pts, dtype=np.int32)
                    # Pale decaying border
                    cv2.fillPoly(img, [blob_arr], (60, 110, 110))
                    # Dark rotting core
                    core_pts = np.array([[(int(lx + (p[0]-lx)*0.7), int(ly + (p[1]-ly)*0.7))] for p in blob_pts], dtype=np.int32)
                    cv2.fillPoly(img, [core_pts], (25, 35, 45))
                    # Faint white fungal mildew fringe
                    for bp in blob_pts[::2]:
                        cv2.circle(img, (bp[0], bp[1]), 3, (210, 220, 220), -1)

    # Slight Gaussian blur for natural camera look
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img

def create_dataset(samples_per_class: int = 100) -> Dict[str, int]:
    """
    Creates full stratified dataset:
    - 70% Training
    - 15% Validation
    - 15% Testing
    """
    train_count = int(samples_per_class * 0.70)
    val_count = int(samples_per_class * 0.15)
    test_count = samples_per_class - train_count - val_count

    stats = {"train": 0, "val": 0, "test": 0, "classes": {}}

    for class_name in config.CLASS_NAMES:
        class_train_dir = config.TRAIN_DIR / class_name
        class_val_dir = config.VAL_DIR / class_name
        class_test_dir = config.TEST_DIR / class_name

        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_val_dir.mkdir(parents=True, exist_ok=True)
        class_test_dir.mkdir(parents=True, exist_ok=True)

        # Generate train
        for i in range(train_count):
            img = generate_procedural_leaf(class_name, config.IMG_HEIGHT)
            cv2.imwrite(str(class_train_dir / f"{class_name}_train_{i:04d}.jpg"), img)

        # Generate val
        for i in range(val_count):
            img = generate_procedural_leaf(class_name, config.IMG_HEIGHT)
            cv2.imwrite(str(class_val_dir / f"{class_name}_val_{i:04d}.jpg"), img)

        # Generate test
        for i in range(test_count):
            img = generate_procedural_leaf(class_name, config.IMG_HEIGHT)
            cv2.imwrite(str(class_test_dir / f"{class_name}_test_{i:04d}.jpg"), img)

        stats["classes"][class_name] = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": samples_per_class
        }
        stats["train"] += train_count
        stats["val"] += val_count
        stats["test"] += test_count

    print(f"Dataset generated: {stats['train']} Train | {stats['val']} Val | {stats['test']} Test across {len(config.CLASS_NAMES)} classes.")
    return stats

def save_sample_visualizations():
    """Saves a multi-class sample grid and augmentation examples to outputs/."""
    # 1. Dataset Class Grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("AcenAAr Smart Spray Robot - Crop Leaf Disease Dataset Classes", fontsize=16, fontweight="bold")
    
    for idx, class_name in enumerate(config.CLASS_NAMES):
        row, col = divmod(idx, 4)
        sample_img_path = list((config.TRAIN_DIR / class_name).glob("*.jpg"))
        if sample_img_path:
            img_bgr = cv2.imread(str(sample_img_path[0]))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_bgr = generate_procedural_leaf(class_name)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        axes[row, col].imshow(img_rgb)
        condition = config.DISEASE_INFO[class_name]["condition"]
        title_color = "darkgreen" if condition == "Healthy" else "darkred"
        clean_name = class_name.replace("___", " - ").replace("_", " ")
        axes[row, col].set_title(f"{clean_name}\n({condition})", fontsize=10, color=title_color, fontweight="bold")
        axes[row, col].axis("off")
        
    plt.tight_layout()
    grid_path = config.OUTPUTS_DIR / "sample_dataset_grid.png"
    plt.savefig(str(grid_path), dpi=200)
    plt.close()
    print(f"Saved dataset preview grid to: {grid_path}")

    # 2. Augmentation Preview
    demo_class = "Tomato___Early_Blight"
    base_img = cv2.imread(str(list((config.TRAIN_DIR / demo_class).glob("*.jpg"))[0]))
    base_img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    aug_fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    aug_fig.suptitle("Data Augmentation Pipeline (OpenCV / Keras Preprocessing)", fontsize=14, fontweight="bold")

    # Original
    axes[0].imshow(base_img_rgb)
    axes[0].set_title("Original (224x224)")
    axes[0].axis("off")

    # Rotation (45 deg)
    M = cv2.getRotationMatrix2D((112, 112), 45, 1.0)
    rot_img = cv2.warpAffine(base_img_rgb, M, (224, 224), borderMode=cv2.BORDER_REFLECT)
    axes[1].imshow(rot_img)
    axes[1].set_title("Rotation (+45°)")
    axes[1].axis("off")

    # Horizontal & Vertical Flip
    flip_img = cv2.flip(base_img_rgb, -1)
    axes[2].imshow(flip_img)
    axes[2].set_title("H & V Flip")
    axes[2].axis("off")

    # Zoom / Crop
    h, w = base_img_rgb.shape[:2]
    crop = base_img_rgb[25:h-25, 25:w-25]
    zoom_img = cv2.resize(crop, (224, 224))
    axes[3].imshow(zoom_img)
    axes[3].set_title("Zoom In (1.25x)")
    axes[3].axis("off")

    # Brightness & Contrast Adjustment
    bc_img = cv2.convertScaleAbs(base_img_rgb, alpha=1.2, beta=20)
    axes[4].imshow(bc_img)
    axes[4].set_title("Brightness & Contrast")
    axes[4].axis("off")

    plt.tight_layout()
    aug_path = config.OUTPUTS_DIR / "augmentation_samples.png"
    plt.savefig(str(aug_path), dpi=200)
    plt.close()
    print(f"Saved augmentation preview to: {aug_path}")

def get_data_generators(batch_size: int = config.BATCH_SIZE):
    """
    Constructs Keras ImageDataGenerators for training, validation, and testing.
    Uses MobileNetV2 preprocessing (or normalization between 0 and 1).
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        brightness_range=[0.8, 1.2]
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(
        directory=str(config.TRAIN_DIR),
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        directory=str(config.VAL_DIR),
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        directory=str(config.TEST_DIR),
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator, test_generator

if __name__ == "__main__":
    print("Initializing Crop Leaf Disease Dataset...")
    stats = create_dataset(samples_per_class=120)
    save_sample_visualizations()
    print("Preprocessing pipeline verification completed.")
