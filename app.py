"""
Web demo: ask questions about a financial report in the browser.

Run:
    streamlit run app.py
"""

import time
from pathlib import Path

import streamlit as st

from src.config import LLM_MODEL, RAW_PDF_DIR
from src.generation.answer import answer_question
from src.retrieval.embeddings import load_embedding_model
from src.retrieval.evaluate import load_financebench
from src.retrieval.pipeline import build_document_index
from src.retrieval.statements import detect_question_statements


DEFAULT_REPORT = "3M_2018_10K"

STATEMENT_NAMES = {
    "cash_flow": "cash flow statement",
    "balance_sheet": "balance sheet",
    "income": "income statement"
}


# ==========================================
# STEP 1: LOAD THINGS ONCE (cached)
# ==========================================
# Streamlit re-runs this whole file on every click. @st.cache_resource
# keeps the result in memory, so the model and each report's index
# are built only the first time.

@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner="Reading the report and building the search index...",
                   max_entries=5)
def get_document_index(report_name: str):
    return build_document_index(RAW_PDF_DIR / f"{report_name}.pdf", get_embedding_model())


@st.cache_data
def get_example_questions() -> dict:
    """
    Return {report name: [FinanceBench questions about it]}.
    """
    examples = {}

    try:
        for record in load_financebench():
            report = Path(record["doc_name"]).stem
            examples.setdefault(report, []).append(record)

    except FileNotFoundError:
        pass  # the demo still works without the FinanceBench file

    return examples


@st.cache_data
def list_reports() -> list[str]:
    return sorted(pdf.stem for pdf in RAW_PDF_DIR.glob("*.pdf"))


# ==========================================
# STEP 2: PAGE LAYOUT
# ==========================================

st.set_page_config(page_title="Financial Report Q&A", page_icon="📊")

st.title("📊 Financial Report Q&A")
st.caption(
    f"Ask a question about a company's annual or quarterly report. "
    f"Everything runs locally: search with FAISS, answers by {LLM_MODEL} via Ollama."
)

reports = list_reports()

if not reports:
    st.error(f"No PDF reports found in `{RAW_PDF_DIR}`. See the README for the data setup.")
    st.stop()

# 2a. Choose a report
report = st.selectbox(
    "Report",
    reports,
    index=reports.index(DEFAULT_REPORT) if DEFAULT_REPORT in reports else 0
)

# 2b. Optional: pick a FinanceBench example question for this report
examples = get_example_questions().get(report, [])
example = None

if examples:
    labels = ["(write my own question)"] + [e["question"][:120] for e in examples]
    choice = st.selectbox(f"Example questions from FinanceBench ({len(examples)})", labels)

    if choice != labels[0]:
        example = examples[labels.index(choice) - 1]

# 2c. The question box (filled with the example, if one was picked)
question = st.text_area(
    "Your question",
    value=example["question"] if example else "",
    height=110,
    placeholder="e.g. What was the total revenue in this fiscal year?"
)

ask = st.button("Ask", type="primary", disabled=not question.strip())


# ==========================================
# STEP 3: ANSWER
# ==========================================

if ask:

    model = get_embedding_model()
    document_index = get_document_index(report)

    # 3a. Show if the question names a financial statement
    statements = detect_question_statements(question)

    if statements:
        names = ", ".join(STATEMENT_NAMES[s] for s in sorted(statements))
        st.info(f"Question mentions the **{names}**: those pages are searched first.")

    # 3b. Search + LLM answer
    with st.spinner("Searching the report and writing the answer..."):

        start = time.time()

        try:
            result = answer_question(question, document_index, model)

        except ConnectionError as error:
            st.error(f"{error}\n\nStart Ollama, then try again.")
            st.stop()

        seconds = time.time() - start

    # 3c. The answer
    st.subheader("Answer")
    st.markdown(result["answer"])
    st.caption(f"Answered in {seconds:.0f} s")

    # 3d. Compare with the FinanceBench answer, if this was an example
    if example and question == example["question"]:
        with st.expander("FinanceBench reference answer"):
            st.markdown(example["answer"])

    # 3e. The pages the LLM read, with their text
    st.subheader("Pages the model read")

    page_texts = {
        page["metadata"]["page"]: page["text"]
        for page in document_index["pages"]
    }

    for page in result["pages"]:

        label = document_index["page_statements"].get(page)
        title = f"Page {page}" + (f"  ·  {STATEMENT_NAMES[label]}" if label else "")

        with st.expander(title):
            st.text(page_texts[page])
