"""
OCR helper for extracting text from uploaded images using Tesseract.
"""
import os
import shutil

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    if os.name == "nt":
        for candidate in (
            shutil.which("tesseract"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ):
            if candidate and os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                break

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def extract_text_from_image(file_storage) -> str:
    """
    Accept a Flask FileStorage object (request.files['image']).
    Returns extracted text string, or raises RuntimeError.
    """
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR not available. Install: pip install pytesseract pillow "
            "and tesseract-ocr system package."
        )

    try:
        img = Image.open(file_storage.stream).convert("RGB")

        # Pre-process: sharpen and increase contrast for better OCR accuracy.
        img = img.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, config=config)
        text = text.strip()

        if not text:
            raise RuntimeError(
                "No text could be extracted from the image. "
                "Try a clearer, higher-contrast screenshot."
            )
        return text

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}")
