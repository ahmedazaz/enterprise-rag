
from PIL import Image
import pytesseract
from pathlib import Path


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from a scanned image using Tesseract OCR.
    """

    image = Image.open(image_path)

    text = pytesseract.image_to_string(
        image,
        config="--psm 6"
    )

    return text


if __name__ == "__main__":

    image_path = Path("page_179.png")

    text = extract_text_from_image(str(image_path))

    print("Extracted characters:", len(text))
    print("Extracted words:", len(text.split()))

    print("\n========== OCR OUTPUT ==========\n")
    print(text)
