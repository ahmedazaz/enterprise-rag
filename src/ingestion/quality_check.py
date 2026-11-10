
from src.ingestion.loader import load_pdf_files
from src.ingestion.extractor import extract_text_from_pdf


def check_all_pdfs():

    # 1. Load all PDF files
    pdf_files = load_pdf_files()

    # 2. Initialize counters
    total_documents = 0
    total_pages = 0
    total_characters = 0
    empty_pages = 0

    failed_documents = []
    empty_page_details = []

    # 3. Process each PDF
    for pdf_path in pdf_files:

        try:
            pages = extract_text_from_pdf(pdf_path)

            document_pages = len(pages)

            document_characters = sum(
                len(page["text"]) for page in pages
            )

            document_empty_pages = sum(
                not page["text"].strip() for page in pages
            )

            # 4. Collect empty page information
            for page in pages:
                if not page["text"].strip():

                    empty_page_details.append(
                        (
                            pdf_path.name,
                            page["metadata"]["page"]
                        )
                    )

            # 5. Update global counters
            total_documents += 1
            total_pages += document_pages
            total_characters += document_characters
            empty_pages += document_empty_pages

            print(
                f"Processed: {pdf_path.name} | "
                f"Pages: {document_pages} | "
                f"Characters: {document_characters}"
            )

        except Exception as error:

            failed_documents.append(
                (pdf_path.name, str(error))
            )

    # 6. Print quality report
    print("\n========== QUALITY REPORT ==========")

    print("Total PDFs:", len(pdf_files))
    print("Processed PDFs:", total_documents)
    print("Failed PDFs:", len(failed_documents))
    print("Total Pages:", total_pages)
    print("Total Characters:", total_characters)
    print("Empty Pages:", empty_pages)

    # 7. Print failed documents
    if failed_documents:

        print("\n========== FAILED DOCUMENTS ==========")

        for filename, error in failed_documents:
            print(filename, "|", error)

    # 8. Print empty page details
    print("\n========== EMPTY PAGE DETAILS ==========")

    for filename, page_number in empty_page_details[:30]:

        print(
            f"{filename} | Page {page_number}"
        )

    print("\nTotal empty pages:", len(empty_page_details))


if __name__ == "__main__":
    check_all_pdfs()
