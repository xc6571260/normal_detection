import cv2

def draw_mask_contour(image, mask, color, thickness=3):
    """
    Draws the contour (boundary) of a binary mask on the original image.

    Args:
        image (np.ndarray): Original image.
        mask (np.ndarray): Binary mask.
        color (Tuple[int, int, int]): BGR color for the contour.
        thickness (int): Thickness of the contour lines.

    Returns:
        np.ndarray: Image with contours overlaid.
    """
    contour_image = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, contours, -1, color, thickness)
    return contour_image

def annotate_result_text(image, mask_statuses, filename=None):
    """
    Adds a large annotation text on the bottom-left of the image
    describing the misalignment detection result.

    Args:
        image (np.ndarray): Image to annotate.
        mask_statuses (List[bool]): Misalignment status for each split mask.

    Returns:
        np.ndarray: Annotated image.
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if all(mask_statuses):
        text = "Both walls misaligned"
        text_color = (0, 0, 255)  # Red
    elif any(mask_statuses):
        text = "One wall misaligned"
        text_color = (0, 165, 255)  # Orange
    else:
        text = "No misalignment detected"
        text_color = (0, 200, 0)  # Green

    name_info = f"{filename}" if filename else ""
    print(f"[INFO] image_name:{name_info} result：{text}")

    font_scale = 4
    thickness = 10
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = 20
    y = annotated.shape[0] - 20

    # Draw white background box
    cv2.rectangle(annotated, (x - 10, y - text_size[1] - 10),
                  (x + text_size[0] + 10, y + 10), (255, 255, 255), -1)

    # Draw the actual text
    cv2.putText(annotated, text, (x, y), font, font_scale, text_color, thickness)

    return annotated
