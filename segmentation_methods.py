import cv2
import numpy as np

def darken_image(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Darken the image by reducing pixel values."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def improved_segmentation(h_channel: np.ndarray, s_channel: np.ndarray, v_channel: np.ndarray, 
                          stage: int = 1, min_cell_area: int = 500) -> np.ndarray:
    """
    Perform hue-based segmentation with morphological cleaning and contour filtering.
    """
    if stage == 1:
        mask_blue = cv2.inRange(h_channel, 90, 135)
        mask_purple = cv2.inRange(h_channel, 136, 170)
        s_thresh = (60, 255)
        v_thresh = (70, 255)
    else:
        mask_blue = cv2.inRange(h_channel, 100, 130)
        mask_purple = cv2.inRange(h_channel, 140, 165)
        s_thresh = (80, 255)
        v_thresh = (90, 255)

    mask_hue = cv2.bitwise_or(mask_blue, mask_purple)
    s_mask = cv2.inRange(s_channel, s_thresh[0], s_thresh[1])
    v_mask = cv2.inRange(v_channel, v_thresh[0], v_thresh[1])

    combined_mask = cv2.bitwise_and(mask_hue, s_mask)
    combined_mask = cv2.bitwise_and(combined_mask, v_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(cleaned)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_cell_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return final_mask

def segment_cell_hue_channel(img: np.ndarray, darken_factor: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment cells in an image based on hue channel thresholding.

    Darken the image only for mask creation, but apply mask on original image.

    Args:
        img: Input RGB image.
        darken_factor: Factor to darken the image for better segmentation.

    Returns:
        segmented_result: RGB image masked with segmentation on original colors.
        mask: Binary segmentation mask.
    """
    if img is None or img.size == 0:
        raise ValueError("Empty or invalid image input.")

    # Darken the RGB image copy (do NOT convert to BGR, stay consistent)
    dark_rgb = darken_image(img, factor=darken_factor)

    # Convert the darkened image to HSV
    hsv = cv2.cvtColor(dark_rgb, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # Get mask from hue segmentation
    mask = improved_segmentation(h_channel, s_channel, v_channel, stage=1)

    # Apply mask to ORIGINAL image to keep original colors
    segmented_result = cv2.bitwise_and(img, img, mask=mask)

    return segmented_result, mask


import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

def segment_cell_otsu_dilation(bgr_img):
    """
    Segment cells using LAB space and multi-Otsu thresholding with dilation.
    
    Parameters:
        bgr_img (np.ndarray): Input BGR image.
        
    Returns:
        segmented_rgb (np.ndarray): Segmented RGB image with mask applied.
        final_mask (np.ndarray): Final binary segmentation mask.
    """
    DILATION_KERNEL_SIZE = (12, 12)
    MIN_AREA = 800
    N_CLASSES = 3

    # Convert to RGB for output
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Convert to LAB and get a-channel
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    _, a_channel, _ = cv2.split(lab)

    # Multi-Otsu thresholding on a-channel
    thresholds = threshold_multiotsu(a_channel, classes=N_CLASSES)
    regions = np.digitize(a_channel, bins=thresholds)
    binary_mask = np.uint8(regions == (len(thresholds)))  # Keep highest class

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # Dilation
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATION_KERNEL_SIZE)
    dilated_mask = cv2.dilate(cleaned_mask, dilation_kernel, iterations=5)

    # Area filtering
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(dilated_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    # Apply final mask to RGB image
    segmented_rgb = cv2.bitwise_and(rgb, rgb, mask=final_mask)

    return segmented_rgb, final_mask


import cv2
import numpy as np

def segment_nucleus_hue_channel(bgr_img):
    """
    Segment nuclei using HSV hue range for purple/pink and blue tones.
    
    Parameters:
        bgr_img (np.ndarray): Input BGR image.
        
    Returns:
        segmented_rgb (np.ndarray): Segmented RGB image with mask applied.
        cleaned_mask (np.ndarray): Final binary mask of nuclei.
    """
    # Parameters
    lower_purple = 110
    upper_pink = 170
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Convert to RGB
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Darken RGB before HSV conversion
    dark_rgb = np.clip(rgb.astype(np.float32) * 1.95, 0, 255).astype(np.uint8)

    # Convert to HSV and split channels
    hsv = cv2.cvtColor(dark_rgb, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # Hue-based segmentation
    mask_purple_pink = cv2.inRange(h_channel, lower_purple, upper_pink)
    mask_blue = cv2.inRange(h_channel, 100, 130)
    mask_hue = cv2.bitwise_or(mask_purple_pink, mask_blue)

    s_mask = cv2.inRange(s_channel, 50, 255)
    v_mask = cv2.inRange(v_channel, 50, 255)
    combined_mask = cv2.bitwise_and(mask_hue, s_mask)
    combined_mask = cv2.bitwise_and(combined_mask, v_mask)

    # Morphological cleaning
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask to image
    segmented_rgb = cv2.bitwise_and(rgb, rgb, mask=cleaned_mask)

    return segmented_rgb, cleaned_mask



import numpy as np
import cv2
from skimage.filters import threshold_multiotsu

def segment_nucleus_otsu(bgr_img):
    """
    Segment nuclei using Multi-Otsu thresholding on the LAB color space.

    Parameters:
        bgr_img (np.ndarray): Input BGR image.

    Returns:
        segmented_rgb (np.ndarray): Segmented RGB image with mask applied.
        cleaned_mask (np.ndarray): Final binary mask of nuclei.
    """
    # Parameters
    N_CLASSES = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Convert to RGB for display
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Multi-Otsu thresholding on a_channel
    thresholds = threshold_multiotsu(a_channel, classes=N_CLASSES)
    regions = np.digitize(a_channel, bins=thresholds)
    binary_mask = np.uint8(regions == (len(thresholds)))  # Select highest region

    # Morphological cleaning
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # Convert mask to 0/255 for bitwise_and
    cleaned_mask_255 = (cleaned_mask * 255).astype(np.uint8)

    # Final segmentation with cleaned mask
    segmented_rgb = cv2.bitwise_and(rgb, rgb, mask=cleaned_mask_255)

    return segmented_rgb, cleaned_mask_255
