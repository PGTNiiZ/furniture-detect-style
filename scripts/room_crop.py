"""Shared detector-box clipping and crop geometry in EXIF-oriented coordinates."""
import math


def prepare_crop(image, box, *, padding_fraction, min_size):
    if (not isinstance(min_size, int) or isinstance(min_size, bool) or min_size < 1
            or not isinstance(padding_fraction, (int,float)) or isinstance(padding_fraction, bool)
            or not math.isfinite(padding_fraction) or not 0 <= padding_fraction <= 1):
        raise ValueError('Invalid crop padding or minimum size')
    try:
        coords = [float(value) for value in box]
    except (TypeError, ValueError):
        raise ValueError('Detector box must have four finite coordinates') from None
    if len(coords) != 4 or not all(math.isfinite(value) for value in coords) or coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError('Detector box is nonfinite or empty')
    width,height = image.size
    x1 = min(width,max(0,math.floor(coords[0])))
    y1 = min(height,max(0,math.floor(coords[1])))
    x2 = min(width,max(0,math.ceil(coords[2])))
    y2 = min(height,max(0,math.ceil(coords[3])))
    if x2-x1 < min_size or y2-y1 < min_size:
        raise ValueError('Clipped detector box is empty or too small')
    pad_x, pad_y = math.ceil((x2-x1)*padding_fraction), math.ceil((y2-y1)*padding_fraction)
    crop_box = [max(0,x1-pad_x),max(0,y1-pad_y),min(width,x2+pad_x),min(height,y2+pad_y)]
    return [x1,y1,x2,y2], crop_box, image.crop(crop_box)
