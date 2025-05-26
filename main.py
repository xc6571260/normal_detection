import os
import cv2
import numpy as np
from ultralytics import YOLO

# Import utility functions from separate modules
from utils.tiling import tile_image, untile_mask
from utils.mask_ops import split_mask_by_kmeans
from utils.detection import check_dx_jumps_within_mask
from utils.visualization import draw_mask_contour, annotate_result_text

# === Configuration ===
model_path = "models/best.pt"         # Path to your trained YOLO model
input_folder = "input"                # Folder containing images to process
output_folder = "output"              # Output folder to save results
tile_size = 1024                      # Tile size (should match training)
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO(model_path)

# === Main image processing loop ===
for fname in os.listdir(input_folder):
    # Only process image files
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Load image
    image_path = os.path.join(input_folder, fname)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipped: {fname} (cannot read)")
        continue

    # Step 1: Split the image into tiles
    tiles, positions, H, W = tile_image(image, tile_size)

    # Step 2: Perform inference on each tile
    masks = []
    for tile in tiles:
        results = model(tile, verbose=False)[0]  # Get YOLO result for one tile

        # Create binary mask of the result
        mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        if results.masks is not None:
            for m in results.masks.data:
                mask = np.maximum(mask, m.cpu().numpy().astype(np.uint8))
        masks.append(mask)

    # Step 3: Reconstruct full-size mask from all tiles
    full_mask = untile_mask(masks, positions, H, W)

    # Step 4: Split the full mask into K clusters (e.g., for 2 walls)
    split_masks = split_mask_by_kmeans(full_mask, k=2)

    # Step 5: Analyze each mask for horizontal misalignment
    overlay = image.copy()
    mask_statuses = []

    for smask in split_masks:
        has_jump = check_dx_jumps_within_mask(smask)
        mask_statuses.append(has_jump)

        # Visualize each wall's mask with green or red contour
        color = (0, 0, 255) if has_jump else (0, 255, 0)
        overlay = draw_mask_contour(overlay, smask, color, thickness=4)

    # Step 6: Annotate summary result on the image
    overlay = annotate_result_text(overlay, mask_statuses, filename=f"{os.path.splitext(fname)[0]}.jpg")

    # Save the final output image
    output_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}.jpg")
    cv2.imwrite(output_path, overlay)
    print(f"✅ Saved: {output_path}")
