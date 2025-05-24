import numpy as np
from sklearn.cluster import KMeans

def split_mask_by_kmeans(mask, k=2):
    """
    Splits a binary mask into 'k' clusters using KMeans based on x-coordinates.

    Args:
        mask (np.ndarray): Binary mask with shape (H, W).
        k (int): Number of clusters to split the mask into.

    Returns:
        masks (List[np.ndarray]): List of clustered binary masks.
    """
    ys, xs = np.where(mask == 1)

    # If not enough points for clustering
    if len(xs) < k:
        return [mask.copy()]

    coords = np.array(xs).reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(coords)
    labels = kmeans.labels_

    masks = [np.zeros_like(mask, dtype=np.uint8) for _ in range(k)]
    for label, x, y in zip(labels, xs, ys):
        masks[label][y, x] = 1

    return masks
