
from pathlib import Path


def load_pdf_files(data_dir: str = "data/raw") -> list[Path]:

    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    pdf_files = sorted(directory.rglob("*.pdf"))

    return pdf_files
