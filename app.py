import streamlit as st
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from doc_quality import load_and_score_document

st.set_page_config(page_title="Multilingual Enterprise RAG", layout="wide")
st.title("🔒 Multilingual Enterprise Data Assistant (Local & Secure)")

# Sidebar: Document Upload Area
with st.sidebar:
    st.header("Data Upload")
    st.write("Supports 100+ languages. 100% Private.")
    uploaded_file = st.file_uploader("Upload a PDF Report (Any Language)", type="pdf")


# Function to process document with Multilingual Embeddings
# NOTE: no longer using PyPDFLoader directly — load_and_score_document()
# handles digital extraction, scanned-page detection, OCR fallback, and
# per-page quality scoring before we ever build the vectorstore.
@st.cache_resource
def process_document(file_path):
    docs, quality_report = load_and_score_document(file_path)

    # chunk_size=2500, chunk_overlap=400. Tuned after diagnosing a failure where
    # a 5-item numbered list (items (1)-(5) in a source letter) was split across
    # chunk boundaries, so the retriever only pulled the chunk containing item (5)
    # and the summary sentence — items (1)-(4) were in an earlier chunk that
    # wasn't retrieved. A larger chunk_size keeps a numbered list together in one
    # chunk. (For very specific single-fact lookups, a smaller chunk can retrieve
    # more precisely — this value is a good general default, tune per document set.)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=400)
    splits = text_splitter.split_documents(docs)

    # Using 'bge-m3' for superior multilingual embedding (German, Danish, English, etc.)
    embeddings = OllamaEmbeddings(model="bge-m3")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # k=12: retrieve enough chunks so that if a related list or passage got
    # split across chunk boundaries, the additional chunks (including ones the
    # pure similarity score ranked slightly lower) are included together in the
    # context. This fixed the case where a 5-item list came back incomplete —
    # raising k to 12 let all list items be retrieved together.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    return retriever, quality_report


if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    with st.spinner("Checking document quality (OCR fallback for scanned pages)..."):
        try:
            retriever, quality_report = process_document(temp_file_path)
        except RuntimeError as e:
            # Surface the real reason clearly instead of letting a downstream
            # library (e.g. Chroma's "Expected Embeddings to be non-empty")
            # raise a cryptic, hard-to-diagnose error.
            st.error(f"⚠️ Could not process this document.\n\n{e}")
            os.remove(temp_file_path)
            st.stop()

    st.success("Document processed and securely loaded into the Multilingual Vector Database!")

    # --- Transparency layer: show the user exactly what happened ---
    with st.sidebar:
        st.divider()
        st.subheader("📋 Document Quality Report")
        st.text(quality_report.summary_text())
        if quality_report.has_warnings:
            with st.expander("⚠️ View low-confidence pages"):
                for pq in quality_report.flagged_pages:
                    st.write(
                        f"- Page {pq.page_number} "
                        f"(source: {pq.source}, quality: {pq.quality_score}/100) — {pq.note}"
                    )

    # Using Qwen 2.5 for exceptional multilingual reasoning and accuracy
    # num_predict: max tokens the model is allowed to generate for an answer.
    #   Ollama's own default is quite short (often ~128), which is why answers
    #   were getting cut off mid-sentence. -1 means "no limit" (model stops on
    #   its own); a fixed number like 1024 is safer for predictable latency.
    # num_ctx: total context window (prompt + retrieved chunks + answer).
    #   Needs to be large enough to hold the retrieved context AND the answer,
    #   or the model has no room left to finish its response.
    # Upgraded from qwen2.5:7b -> qwen2.5:32b. This stays 100% local/private
    # (no change to the privacy guarantee), but gives noticeably better
    # reasoning — e.g. correctly grouping a multi-item list ((1)-(5) in a
    # source document) even when the retrieved chunks split awkwardly.
    # Requires ~20-22GB VRAM; comfortable on a 32GB card.
    llm = OllamaLLM(model="qwen2.5:32b", num_predict=1024, num_ctx=8192)

    # Multilingual dynamic prompt
    prompt = ChatPromptTemplate.from_template(
        "You are a professional enterprise assistant. "
        "Answer the user's question using ONLY the provided 'Context' below. "
        "CRITICAL LANGUAGE RULE: First, identify the language of the 'Question' below — "
        "ignore the language(s) used in the 'Context' entirely when deciding this. "
        "The Context may be in a different language, or contain mixed/foreign characters "
        "from OCR — this must NOT influence your reply language. "
        "You must write your ENTIRE answer in the same language as the Question, and no other. "
        "If the answer is not contained in the 'Context', you MUST begin your reply with the exact tag "
        "[NO_ANSWER] (in English, exactly like that), and then, IN THE SAME LANGUAGE AS THE QUESTION, "
        "state that the information is not available in the document. "
        "The [NO_ANSWER] tag is the ONLY English text allowed; the message after it must match the "
        "Question's language exactly — do NOT switch to the Context's language or any other language "
        "when saying the information is not available. "
        "Only use the [NO_ANSWER] tag when the document genuinely does not contain the answer. "
        "Do not hallucinate or use outside knowledge. "
        "IMPORTANT: Reproduce numbers, doses, units, and scientific notation (e.g. 10-6 M/l) "
        "EXACTLY as they appear in the Context — do not paraphrase, round, or generalize them "
        "(e.g. do not replace a specific value with a vague phrase like 'an intermediate dose'). "
        "If a sentence in the Context appears to be cut off or incomplete, answer with what is "
        "available and note (in the same language as the Question) that the source passage "
        "appears incomplete, rather than silently ending your answer as if the sentence were complete.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "FINAL CHECK before you answer: What language is the Question written in? "
        "Your entire response — including any 'information not available' message — must be in that "
        "exact language (the only exception is the literal [NO_ANSWER] tag itself, if needed)."
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build a reliable page_number -> extraction_source lookup ONCE, from the
    # quality report (the authoritative source), not from chunk metadata.
    # Chunks can straddle page boundaries, which caused a chunk to sometimes
    # carry the wrong page's extraction_source and mislabel a clean digital
    # page as "(OCR)". The report knows the true per-page extraction method.
    page_source_lookup = {
        pq.page_number: pq.source for pq in quality_report.page_details
    }

    def format_docs_with_sources(docs):
        """Same as format_docs, but also returns source information so we can
        show the user (a) which pages the answer draws on, and (b) which of
        those pages were low-confidence (scanned/degraded). This makes every
        answer verifiable — the user can trace it back to specific pages."""
        # Which pages contributed to the retrieved context. We take the page
        # numbers from the retrieved chunks, but look up how each page was
        # extracted from the authoritative quality report — not from the
        # chunk's own metadata, which can be wrong at page boundaries.
        source_pages = {}  # page_number -> extraction_source
        for doc in docs:
            page = doc.metadata.get("page_number")
            if page is not None:
                source_pages[page] = page_source_lookup.get(page, "digital")

        low_conf_pages = sorted({
            doc.metadata.get("page_number")
            for doc in docs
            if doc.metadata.get("low_confidence")
        })
        return format_docs(docs), low_conf_pages, source_pages

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chat Interface
    # Wrapped in st.form so the question is only submitted when the user
    # explicitly presses Enter or clicks the button — not on every keystroke
    # or brief pause while typing. Without a form, Streamlit's text_input
    # can re-run the script mid-typing, sending an incomplete question
    # (e.g. two words) into the RAG chain and producing a confused answer
    # that looks like a system error but is really just a premature query.
    with st.form(key="question_form", clear_on_submit=False):
        user_input = st.text_input("Ask a question about the document (in any language):")
        submitted = st.form_submit_button("Ask")

    if submitted and user_input:
        with st.spinner("Analyzing securely across languages..."):
            # Retrieve once so we can both answer AND check source confidence
            retrieved_docs = retriever.invoke(user_input)
            context_text, low_conf_pages, source_pages = format_docs_with_sources(retrieved_docs)

            response = (
                prompt
                | llm
                | StrOutputParser()
            ).invoke({"context": context_text, "question": user_input})

            # Did the model signal that the document doesn't contain the answer?
            # If so, we strip the internal [NO_ANSWER] tag before displaying,
            # AND we skip the source citation — because there is no real source
            # to cite. Showing "Sources: page 1, 2, 3" under a "not found in the
            # document" answer is contradictory and undermines trust.
            no_answer = response.lstrip().startswith("[NO_ANSWER]")
            display_response = response.replace("[NO_ANSWER]", "", 1).lstrip()

            st.write("### Answer:")
            st.info(display_response)

            # --- Source attribution: show which pages this answer draws on ---
            # This makes every answer verifiable — the core of a trustworthy
            # RAG system. The user can open the original document at these
            # pages and confirm the answer is grounded in the source.
            # Only shown when the model actually answered from the document.
            if not no_answer and source_pages:
                parts = []
                for page in sorted(source_pages):
                    tag = " (OCR)" if source_pages[page] == "ocr" else ""
                    parts.append(f"page {page}{tag}")
                st.caption("📄 **Sources:** answer drawn from " + ", ".join(parts))

            # Low-confidence warning is also only meaningful when we actually
            # used the document to answer.
            if not no_answer and low_conf_pages:
                pages_str = ", ".join(str(p) for p in low_conf_pages)
                st.warning(
                    f"⚠️ This answer draws on page(s) {pages_str}, which had "
                    f"low text-extraction confidence (e.g. scanned/degraded "
                    f"content). Please verify against the original document."
                )

    # Cleanup temporary file
    os.remove(temp_file_path)
else:
    st.info("Please upload a company document from the left sidebar to begin.")