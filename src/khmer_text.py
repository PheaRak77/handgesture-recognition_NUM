import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# System-level Khmer font fallbacks (in priority order)
_SYSTEM_KHMER_FONTS = [
    "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifKhmer-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansKhmer-Regular.otf",
]

def _load_font(font_path: str, font_size: int):
    """Load font from given path; fall back to system Khmer fonts if needed."""
    # Try the requested font path first
    try:
        f = ImageFont.truetype(font_path, font_size)
        # Quick sanity check — empty file will raise OSError
        return f
    except (IOError, OSError):
        pass

    # Try known system Khmer fonts
    for sys_font in _SYSTEM_KHMER_FONTS:
        try:
            f = ImageFont.truetype(sys_font, font_size)
            print(f"[khmer_text] Using system font: {sys_font}", flush=True)
            return f
        except (IOError, OSError):
            continue

    # Last resort — PIL default (will show boxes for Khmer but won't crash)
    print("[khmer_text] WARNING: No Khmer font found, using PIL default.", flush=True)
    return ImageFont.load_default()


def put_khmer_text(image, text, position, font_path,
                   font_size: int = 25, color=(0, 255, 0)):
    """Render *text* (including Khmer Unicode) onto *image* using PIL."""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = _load_font(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)