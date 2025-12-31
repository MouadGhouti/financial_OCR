import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional
import PIL
from landingai_ade import LandingAIADE

from processing import Box


class LandingAIClient:
    """
    Thin wrapper around Landing AI Agentic Document Extraction (ADE) APIs.

    Uses environment variables:
      - LANDINGAI_API_KEY
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("LANDINGAI_API_KEY")
        self._client = LandingAIADE(apikey=self.api_key)

    def parse_document(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Any:
        """
        Call the ADE Parse API on the given document via the official SDK.

        Returns the SDK response object, which exposes:
          - .markdown: combined markdown representation
          - .chunks: list of chunks with text, type, and grounding/boxes
        """
        # The SDK expects a path/Path, so write bytes to a temp file.
        suffix = Path(filename).suffix or ".bin"
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            response = self._client.parse(
                document=tmp_path,
                model="dpt-2-latest",
            )
        except Exception as exc:  # Surface as our own error type for the UI
            raise Exception(str(exc)) from exc
        finally:
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                # Best-effort cleanup; ignore failures
                pass

        return response


def extract_bounding_boxes(parse_result: Any, image: PIL.Image.Image) -> List[Dict[str, Any]]:
    """
    Extract a list of bounding boxes from the parse result.

    Because the exact schema may evolve, this function supports several
    common patterns, for example:
      - chunk['bbox'] with x/y/width/height/page
      - chunk['bounding_box'] with similar fields

    Returns a list of dictionaries with:
      - page_index (int)
      - x, y, width, height (float, normalized 0–1 if we can infer that)
      - label (str)
      - text (str)
    """
    # For the SDK, parse_result is a response object with .chunks.
    chunks = getattr(parse_result, "chunks", []) or []
    boxes: List[Dict[str, Any]] = []

    for chunk in chunks:
        # SDK chunk objects expose .type, .markdown, and .grounding.box
        label = str(getattr(chunk, "type", "chunk"))
        text = str(
            getattr(chunk, "markdown", "")
            or getattr(chunk, "text", "")
        )

        grounding = getattr(chunk, "grounding", None)
        box_obj = None
        if grounding is not None:
            box_obj = getattr(grounding, "box", None)
        if box_obj is None:
            continue

        image_width, image_height = image.size
        left = box_obj.left
        right = box_obj.right
        top = box_obj.top
        bottom = box_obj.bottom
        x1 = left * image_width
        y1 = top * image_height
        x2 = right * image_width
        y2 = bottom * image_height

        page_index_val = getattr(grounding, "page")
        page_index = int(page_index_val)
        boxes.append(
            Box(
                page_index=page_index,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=label,
                text=text,
            )
        )
    return boxes
