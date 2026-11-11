
import fitz
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:

    extracted_pages = []

    with fitz.open(pdf_path) as pdf:

        total_pages = len(pdf)

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            page_data = {
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number,
                    "total_pages": total_pages
                },
                "needs_ocr": len(text) == 0
            }

            extracted_pages.append(page_data)

    return extracted_pages
