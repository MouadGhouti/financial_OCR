from dataclasses import dataclass
from typing import Iterable

import fitz
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Box:
    page_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    text: str


def load_image_from_upload(uploaded_file) -> Image.Image:
    """Load a PIL image from a Streamlit UploadedFile (image types)."""
    # uploaded_file is a BytesIO-like object
    return Image.open(uploaded_file).convert("RGB")


def pdf_to_image_first_page(uploaded_file) -> Image.Image:
    """
    Convert the first page of an uploaded PDF to a PIL Image.

    Uses pdf2image.convert_from_bytes, so system must have poppler installed.
    """
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    if not images:
        raise ValueError("No pages found in PDF")
    return images


def draw_bounding_boxes(
    image: Image.Image,
    boxes: Iterable[Box],
) -> Image.Image:
    """
    Draw bounding boxes on a copy of the image.

    - If box coordinates are in [0, 1], they are treated as normalized.
    - Otherwise, they are treated as pixel coordinates.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    font = None
    try:
        base_font_size = 20
        font = ImageFont.truetype("Arial Bold.ttf", base_font_size)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        x1 = box.x1
        y1 = box.y1
        x2 = box.x2
        y2 = box.y2
        # Draw rectangle
        color = "red" if box.label == "text" else "blue"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background + text
        label_text = box.label
        if label_text and font is not None:
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:  # Fallback for older Pillow versions
                text_w, text_h = font.getsize(label_text)

            padding = 10
            bg_coords = [
                x1,
                max(0, y1 - text_h - 2 * padding),
                x1 + text_w + 2 * padding,
                y1,
            ]
            draw.rectangle(bg_coords, fill=color)
            draw.text(
                (x1 + padding, bg_coords[1] + padding),
                label_text,
                fill="white",
                font=font,
            )

    return annotated
