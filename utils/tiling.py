import numpy as np

def tile_image(img, tile_size=1024):
    """
    Splits a large image into non-overlapping tiles of fixed size.

    Args:
        img (np.ndarray): Original image (H, W, C).
        tile_size (int): Size of each square tile (default 1024x1024).

    Returns:
        tiles (List[np.ndarray]): List of image tiles.
        positions (List[Tuple[int, int]]): Top-left corner (x, y) of each tile.
        h (int): Original image height.
        w (int): Original image width.
    """
    h, w = img.shape[:2]
    tiles, positions = [], []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:min(y+tile_size, h), x:min(x+tile_size, w)]

            # Pad tile if it's smaller than the tile size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            tiles.append(tile)
            positions.append((x, y))

    return tiles, positions, h, w

def untile_mask(masks, positions, full_h, full_w, tile_size=1024):
    """
    Reconstructs a full-size mask by merging masks of individual tiles.

    Args:
        masks (List[np.ndarray]): List of binary masks from tiles.
        positions (List[Tuple[int, int]]): Tile positions in the original image.
        full_h (int): Original image height.
        full_w (int): Original image width.
        tile_size (int): Tile size used during tiling.

    Returns:
        merged_mask (np.ndarray): Reconstructed full-size binary mask.
    """
    merged_mask = np.zeros((full_h, full_w), dtype=np.uint8)

    for mask, (x, y) in zip(masks, positions):
        mh, mw = min(tile_size, full_h - y), min(tile_size, full_w - x)
        merged_mask[y:y+mh, x:x+mw] = np.maximum(
            merged_mask[y:y+mh, x:x+mw],
            mask[:mh, :mw]
        )

    return merged_mask
