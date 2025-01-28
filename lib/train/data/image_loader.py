import jpeg4py
import cv2 as cv
from PIL import Image
import numpy as np

# Define the Davis palette for indexed color images
davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
    [0, 128, 128], [128, 128, 128], [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
    [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128], [0, 64, 0], [128, 64, 0],
    [0, 191, 0], [128, 191, 0], [0, 64, 128], [128, 64, 128]
]


def default_image_loader(path):
    """Attempts to load an image, first using jpeg4py, falling back to opencv if jpeg4py fails."""
    if default_image_loader.use_jpeg4py is None:
        # Try using jpeg4py
        img = jpeg4py_loader(path)
        if img is None:
            default_image_loader.use_jpeg4py = False
            print('Using opencv_loader instead.')
        else:
            default_image_loader.use_jpeg4py = True
            return img

    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)
    return opencv_loader(path)


default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path):
    """Loads an image using jpeg4py."""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print(f"ERROR: Could not read image '{path}' with jpeg4py.")
        print(e)
        return None


def opencv_loader(path):
    """Loads an image using OpenCV and returns it in RGB format."""
    try:
        img = cv.imread(path, cv.IMREAD_COLOR)
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB and return
    except Exception as e:
        print(f"ERROR: Could not read image '{path}' with OpenCV.")
        print(e)
        return None


def jpeg4py_loader_w_failsafe(path):
    """Attempts to load an image using jpeg4py, falls back to OpenCV if it fails."""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print(f"jpeg4py failed, trying OpenCV for '{path}'.")
        try:
            img = cv.imread(path, cv.IMREAD_COLOR)
            return cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB and return
        except Exception as e:
            print(f"ERROR: Could not read image '{path}' with OpenCV either.")
            print(e)
            return None


def opencv_seg_loader(path):
    """Loads a segmentation image (annotation) using OpenCV."""
    try:
        return cv.imread(path, cv.IMREAD_UNCHANGED)  # Use unchanged to read the raw format
    except Exception as e:
        print(f"ERROR: Could not read segmentation image '{path}' with OpenCV.")
        print(e)
        return None


def imread_indexed(filename):
    """Reads an indexed image (typically used for segmentation annotations)."""
    try:
        im = Image.open(filename)
        annotation = np.atleast_3d(im)[..., 0]  # Take the first channel for segmentation
        return annotation
    except Exception as e:
        print(f"ERROR: Could not read indexed image '{filename}'.")
        print(e)
        return None


def imwrite_indexed(filename, array, color_palette=None):
    """Saves a 2D array (segmentation annotation) as an indexed PNG image."""
    if color_palette is None:
        color_palette = davis_palette

    if np.atleast_3d(array).shape[2] != 1:
        raise ValueError("Saving indexed PNGs requires a 2D array.")

    try:
        im = Image.fromarray(array)
        im.putpalette(color_palette.ravel())  # Apply the color palette
        im.save(filename, format='PNG')
    except Exception as e:
        print(f"ERROR: Could not save indexed image '{filename}'.")
        print(e)


# Example usage of the functions
if __name__ == "__main__":
    img_path = 'path_to_image.jpg'
    indexed_img_path = 'path_to_indexed_image.png'
    seg_img = default_image_loader(img_path)
    if seg_img is not None:
        print("Image loaded successfully!")

    # Save an indexed image
    segmentation_array = np.zeros((100, 100), dtype=np.uint8)  # Example 2D array for segmentation
    imwrite_indexed(indexed_img_path, segmentation_array)
