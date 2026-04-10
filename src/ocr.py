"""
OCR helper for extracting text from uploaded images using Tesseract.
"""
import os
import shutil

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    tesseract_candidates = [shutil.which("tesseract")]

    if os.name == "nt":
        tesseract_candidates.extend([
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ])
    else:
        tesseract_candidates.extend([
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ])

    TESSERACT_CMD = next(
        (candidate for candidate in tesseract_candidates if candidate and os.path.exists(candidate)),
        None,
    )

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    OCR_AVAILABLE = bool(TESSERACT_CMD)
except ImportError:
    OCR_AVAILABLE = False
    TESSERACT_CMD = None


def extract_text_from_image(file_storage) -> str:
    """
    Accept a Flask FileStorage object (request.files['image']).
    Returns extracted text string, or raises RuntimeError.
    """
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR is not available on this deployment because the Tesseract "
            "system package is missing."
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
