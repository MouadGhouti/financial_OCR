import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from landingai_client import (
    LandingAIClient,
    extract_bounding_boxes,
)
from processing import (
    draw_bounding_boxes,
    load_image_from_upload,
    pdf_to_image_first_page,
)


def _ensure_env_loaded() -> None:
    # Only load once; safe to call multiple times.
    load_dotenv()


def _preview_image(uploaded_file) -> Image.Image:
    if uploaded_file.type == "application/pdf":
        # Important: pdf_to_image_first_page reads from the uploaded_file;
        # we need a fresh buffer because Streamlit may re-use it.
        uploaded_file.seek(0)
        images = pdf_to_image_first_page(uploaded_file)
        images = [image.convert("RGB") for image in images]
        st.caption("Showing all pages of PDF (PoC).")
        return images
    else:
        uploaded_file.seek(0)
        image = load_image_from_upload(uploaded_file)
    return image


def main() -> None:
    _ensure_env_loaded()

    st.set_page_config(
        page_title="Financial Document Extractor",
        layout="wide",
    )
    st.title("Financial Document Extractor PoC")
    st.write(
        "Upload a financial document (image or PDF). "
        "This demo parses it with Landing AI's ADE API, "
        "overlays bounding boxes, and shows extracted fields."
    )
    api_key = st.text_input("Enter your Landing AI API key", type="password")
    if not api_key:
        st.warning("No API key provided, using default key")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["png", "jpg", "jpeg", "pdf"],
    )

    if not uploaded_file:
        st.info("Upload a PNG, JPG, or PDF to get started.")
        return

    # Load the document pages (but keep the visual layout for results per page
    # further down).
    try:
        image_or_images = _preview_image(uploaded_file)
        if isinstance(image_or_images, list):
            images = image_or_images
        else:
            images = [image_or_images]
        # Use the first page as reference for box extraction.
        image = images[0]
    except Exception as exc:
        st.error(f"Could not render preview: {exc}")
        return

    analyze_clicked = st.button("Analyze document with LandingAI")
    if not analyze_clicked:
        return

    # Important: re-seek before re-reading
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    try:
        if api_key:
            client = LandingAIClient(api_key=api_key)
        else:
            client = LandingAIClient()
    except Exception as exc:
        st.error(str(exc))
        return

    with st.spinner("Contacting LandingAI (parse)..."):
        try:
            parse_result = client.parse_document(
                file_bytes,
                uploaded_file.name,
            )
        except Exception as exc:
            st.error(str(exc))
            return

    boxes = extract_bounding_boxes(parse_result, image)

    # Aggregate markdown per page.
    pages = {}
    for chunk in parse_result.chunks:
        page_index = int(chunk.grounding.page)
        if page_index not in pages:
            pages[page_index] = chunk.markdown
        else:
            pages[page_index] += "\n\n" + chunk.markdown

    # For each page, show a row with the annotated image on the left and the
    # corresponding markdown on the right.
    for page_index, page_image in enumerate(images):
        col_img, col_md = st.columns([1.1, 1.3])

        with col_img:
            st.subheader(f"Page {page_index+1}")
            page_boxes = [box for box in boxes if box.page_index == page_index]
            if page_boxes:
                annotated = draw_bounding_boxes(page_image, page_boxes)
                st.image(
                    annotated,
                    caption=f"Page {page_index+1} with bounding boxes",
                    use_container_width=True,
                )
            else:
                st.info("No bounding boxes found for this page.")

        with col_md:
            markdown = pages.get(page_index)
            st.subheader(f"Page {page_index+1}")
            if markdown:
                st.markdown(markdown, unsafe_allow_html=True)
            else:
                st.info("No markdown found for this page.")


if __name__ == "__main__":
    main()
