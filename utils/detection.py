import numpy as np

def check_dx_jumps_within_mask(mask, x_jump_threshold=45, step=5, min_points=5):
    """
    Detects horizontal jumps (misalignments) in a binary mask.

    This function analyzes x-position trends along y-axis and identifies
    sudden jumps that may indicate wall misalignment.

    Args:
        mask (np.ndarray): Binary mask (H, W).
        x_jump_threshold (int): Pixel threshold to define a 'jump'.
        step (int): Vertical step size for checking x-position.
        min_points (int): Minimum number of pixels required in a vertical strip.

    Returns:
        bool: True if a misalignment (jump) is detected, else False.
    """
    height, width = mask.shape
    ys, xs = np.where(mask == 1)

    if len(xs) == 0:
        return False

    segments = []

    for y_start in range(0, height - step, step):
        y_end = y_start + step
        in_range = (ys >= y_start) & (ys < y_end)
        segment_xs = xs[in_range]

        if len(segment_xs) >= min_points:
            avg_x = np.mean(segment_xs)
            segments.append((y_start + step // 2, avg_x))

    for i in range(len(segments) - 1):
        _, x0 = segments[i]
        _, x1 = segments[i + 1]
        if abs(x1 - x0) > x_jump_threshold:
            return True

    return False
