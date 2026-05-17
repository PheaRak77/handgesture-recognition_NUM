import os
import sys
import ctypes
from ctypes import wintypes

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUNDLED_KHMER_FONT = os.path.join(_PROJECT_ROOT, "assets", "fonts", "Hanuman.ttf")

# Windows face names for known font files
_FACE_BY_FILENAME = {
    "hanuman.ttf": "Hanuman",
    "khmerui.ttf": "Khmer UI",
    "khmeruib.ttf": "Khmer UI",
    "leelawui.ttf": "Leelawadee UI",
    "leelauib.ttf": "Leelawadee UI",
    "daunpenh.ttf": "DaunPenh",
    "notosanskhmer-regular.ttf": "Noto Sans Khmer",
}

_LINUX_KHMER_FONTS = [
    "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifKhmer-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansKhmer-Regular.otf",
    "/usr/share/fonts/truetype/khmeros/KhmerOS.ttf",
]

_private_fonts_loaded: set[str] = set()
_resolved_font_cache: tuple[str, str] | None = None


def _windows_khmer_fonts():
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    names = [
        "KhmerUI.ttf",
        "KhmerUIb.ttf",
        "daunpenh.ttf",
        "Hanuman.ttf",
        "NotoSansKhmer-Regular.ttf",
        "LeelawUI.ttf",
        "LeelaUIb.ttf",
    ]
    return [os.path.join(fonts_dir, n) for n in names]


def _resolve_font_path(font_path: str) -> str:
    if not font_path:
        return _BUNDLED_KHMER_FONT
    if os.path.isabs(font_path) and os.path.isfile(font_path):
        return font_path
    if os.path.isfile(font_path):
        return os.path.abspath(font_path)
    under_project = os.path.join(_PROJECT_ROOT, font_path)
    if os.path.isfile(under_project):
        return under_project
    return font_path


def _candidate_font_paths(font_path: str):
    resolved = _resolve_font_path(font_path)
    seen: set[str] = set()
    for path in (resolved, _BUNDLED_KHMER_FONT):
        if path and path not in seen:
            seen.add(path)
            yield path
    if sys.platform == "win32":
        for path in _windows_khmer_fonts():
            if path not in seen:
                seen.add(path)
                yield path
    elif sys.platform == "darwin":
        for path in (
            "/System/Library/Fonts/Supplemental/Khmer MN.ttc",
            "/Library/Fonts/NotoSansKhmer-Regular.ttf",
            os.path.expanduser("~/Library/Fonts/NotoSansKhmer-Regular.ttf"),
        ):
            if path not in seen:
                seen.add(path)
                yield path
    else:
        for path in _LINUX_KHMER_FONTS:
            if path not in seen:
                seen.add(path)
                yield path


def _font_face_name(font_path: str) -> str:
    base = os.path.basename(font_path).lower()
    if base in _FACE_BY_FILENAME:
        return _FACE_BY_FILENAME[base]
    try:
        from fontTools.ttLib import TTFont

        tt = TTFont(font_path)
        for name_id in (16, 1, 4):
            name = tt["name"].getBestName(name_id)
            if name:
                return str(name)
    except Exception:
        pass
    return "Khmer UI"


def _resolve_khmer_font(font_path: str) -> tuple[str, str]:
    """Return (font_file_path, GDI/PIL face name)."""
    global _resolved_font_cache
    if _resolved_font_cache is not None:
        return _resolved_font_cache

    for path in _candidate_font_paths(font_path):
        if os.path.isfile(path):
            _resolved_font_cache = (path, _font_face_name(path))
            return _resolved_font_cache

    _resolved_font_cache = ("", "Khmer UI")
    return _resolved_font_cache


def _ensure_private_font(font_path: str) -> None:
    if sys.platform != "win32" or not font_path or font_path in _private_fonts_loaded:
        return
    if not os.path.isfile(font_path):
        return
    gdi32 = ctypes.windll.gdi32
    FR_PRIVATE = 0x10
    if gdi32.AddFontResourceExW(font_path, FR_PRIVATE, 0):
        _private_fonts_loaded.add(font_path)
        ctypes.windll.user32.SendMessageW(0xFFFF, 0x001D, 0, 0)


def _win32_render_text(text: str, face_name: str, font_size: int, color_rgb):
    """Render Khmer text with Windows GDI (correct complex-script shaping)."""
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    hdc_screen = user32.GetDC(0)
    hdc = gdi32.CreateCompatibleDC(hdc_screen)

    class LOGFONTW(ctypes.Structure):
        _fields_ = [
            ("lfHeight", wintypes.LONG),
            ("lfWidth", wintypes.LONG),
            ("lfEscapement", wintypes.LONG),
            ("lfOrientation", wintypes.LONG),
            ("lfWeight", wintypes.LONG),
            ("lfItalic", wintypes.BYTE),
            ("lfUnderline", wintypes.BYTE),
            ("lfStrikeOut", wintypes.BYTE),
            ("lfCharSet", wintypes.BYTE),
            ("lfOutPrecision", wintypes.BYTE),
            ("lfClipPrecision", wintypes.BYTE),
            ("lfQuality", wintypes.BYTE),
            ("lfPitchAndFamily", wintypes.BYTE),
            ("lfFaceName", wintypes.WCHAR * 32),
        ]

    class SIZE(ctypes.Structure):
        _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

    lf = LOGFONTW()
    lf.lfHeight = -int(font_size)
    lf.lfWeight = 400
    lf.lfQuality = 5  # CLEARTYPE_QUALITY
    lf.lfFaceName = face_name[:31]

    hfont = gdi32.CreateFontIndirectW(ctypes.byref(lf))
    old_font = gdi32.SelectObject(hdc, hfont)

    size = SIZE()
    gdi32.GetTextExtentPoint32W(hdc, text, len(text), ctypes.byref(size))
    pad = 6
    w, h = max(size.cx + pad, 1), max(size.cy + pad, 1)

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0

    bits = ctypes.c_void_p()
    hbmp = gdi32.CreateDIBSection(hdc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    old_bmp = gdi32.SelectObject(hdc, hbmp)

    gdi32.PatBlt(hdc, 0, 0, w, h, 0x00000042)  # BLACKNESS — clear bitmap

    gdi32.SetBkMode(hdc, 1)  # TRANSPARENT
    r, g, b = color_rgb
    gdi32.SetTextColor(hdc, (int(b) << 16) | (int(g) << 8) | int(r))

    text_rect = RECT(2, 2, w - 2, h - 2)
    DT_LEFT = 0x00000000
    DT_NOPREFIX = 0x00000200
    DT_SINGLELINE = 0x00000020
    user32.DrawTextW(hdc, text, -1, ctypes.byref(text_rect), DT_LEFT | DT_NOPREFIX | DT_SINGLELINE)

    buf_type = ctypes.c_uint8 * (w * h * 4)
    buf = ctypes.cast(bits, ctypes.POINTER(buf_type)).contents
    bgra = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4)).copy()

    gdi32.SelectObject(hdc, old_bmp)
    gdi32.SelectObject(hdc, old_font)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteObject(hfont)
    gdi32.DeleteDC(hdc)
    user32.ReleaseDC(0, hdc_screen)

    return bgra


def _blend_bgra_onto_bgr(frame: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + ow), min(fh, y + oh)
    if x0 >= x1 or y0 >= y1:
        return

    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)
    patch = overlay[oy0:oy1, ox0:ox1]
    region = frame[y0:y1, x0:x1]

    text_mask = patch[:, :, :3].any(axis=2)
    if not text_mask.any():
        return

    region[text_mask] = patch[text_mask, :3][:, ::-1]  # BGRA -> BGR


def _put_khmer_text_pil(image, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    path, _ = _resolve_khmer_font(font_path)
    try:
        font = ImageFont.truetype(path, font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


_gdi_text_cache: dict[tuple, np.ndarray] = {}


def put_khmer_text(image, text, position, font_path,
                   font_size: int = 25, color=(0, 255, 0)):
    """Render Khmer Unicode onto a BGR OpenCV image."""
    if sys.platform == "win32":
        path, face = _resolve_khmer_font(font_path)
        if path:
            _ensure_private_font(path)
        else:
            print(
                "[khmer_text] WARNING: No Khmer font file found; using system Khmer UI.",
                flush=True,
            )
        try:
            cache_key = (text, face, font_size, color)
            bgra = _gdi_text_cache.get(cache_key)
            if bgra is None:
                bgra = _win32_render_text(text, face, font_size, color)
                _gdi_text_cache[cache_key] = bgra
            out = image.copy()
            _blend_bgra_onto_bgr(out, bgra, position[0], position[1])
            return out
        except Exception as exc:
            print(f"[khmer_text] GDI render failed ({exc}); falling back to PIL.", flush=True)

    return _put_khmer_text_pil(image, text, position, font_path, font_size, color)
