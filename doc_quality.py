"""
Document quality & OCR module for the Multilingual Enterprise RAG system.

Pipeline:
1. Extract text per page with PyPDFLoader (fast path, works for digital PDFs).
2. Flag pages with suspiciously little extracted text as "likely scanned".
3. Re-render those pages as images and run Tesseract OCR on them.
4. Score the quality of the resulting text (digital or OCR'd) so low-confidence
   pages are never silently trusted.
5. Return enriched documents (with per-page metadata: source, ocr_used, quality_score)
   plus a summary report the UI can display to the user.
"""

import re
import statistics
from dataclasses import dataclass, field

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ---- Tunable thresholds -----------------------------------------------

# Below this many characters per page, we suspect the page is scanned
# (i.e. PyPDFLoader found little/no embedded text layer).
MIN_CHARS_FOR_DIGITAL_PAGE = 40

# OCR render resolution. Higher = more accurate OCR, slower, more RAM.
OCR_DPI = 300

# Below this quality score (0-100), a page is flagged as "low confidence"
# in the UI rather than silently trusted.
QUALITY_FLAG_THRESHOLD = 55


@dataclass
class PageQuality:
    page_number: int
    source: str          # "digital" or "ocr"
    char_count: int
    quality_score: float  # 0-100
    flagged: bool
    note: str = ""


@dataclass
class DocumentQualityReport:
    total_pages: int = 0
    ocr_pages: int = 0
    flagged_pages: list = field(default_factory=list)  # list[PageQuality]
    page_details: list = field(default_factory=list)   # list[PageQuality]
    ocr_errors: list = field(default_factory=list)      # list[str], hard OCR failures

    @property
    def has_warnings(self) -> bool:
        return len(self.flagged_pages) > 0

    def summary_text(self) -> str:
        if self.total_pages == 0:
            return "No pages processed."
        lines = [
            f"{self.total_pages} page(s) processed, "
            f"{self.ocr_pages} required OCR (scanned/image-based)."
        ]
        if self.ocr_errors:
            lines.append(
                f"❌ OCR failed on {len(self.ocr_errors)} page(s) — "
                f"check Tesseract/poppler installation."
            )
        if self.flagged_pages:
            pages = ", ".join(str(p.page_number) for p in self.flagged_pages)
            lines.append(
                f"⚠️ {len(self.flagged_pages)} page(s) have low text-quality "
                f"confidence and may produce less reliable answers: page(s) {pages}."
            )
        else:
            lines.append("All pages extracted with high confidence.")
        return "\n".join(lines)


def _quality_score(text: str) -> float:
    """
    Heuristic 0-100 score for how trustworthy extracted text is, whether it
    came from the digital text layer or from OCR.

    Penalizes:
    - very short text (likely extraction failure)
    - low ratio of alphanumeric characters (OCR noise / garbled symbols)
    - excessive repeated whitespace or control characters
    """
    text = text.strip()
    if not text:
        return 0.0

    length = len(text)
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    alnum_ratio = alnum / length

    # Garbage OCR often produces long runs of odd punctuation/symbols
    weird_chars = len(re.findall(r"[^\w\s.,;:!?'\"()\-–—/€$%]", text))
    weird_ratio = weird_chars / length

    length_score = min(length / 200, 1.0)  # saturate around ~200 chars/page
    score = 100 * (0.5 * length_score + 0.4 * alnum_ratio + 0.1 * (1 - weird_ratio))
    return max(0.0, min(100.0, round(score, 1)))


def _ocr_page(pdf_path: str, page_index: int, lang: str = None) -> str:
    """Render a single PDF page to an image and run Tesseract on it."""
    images = convert_from_path(
        pdf_path,
        dpi=OCR_DPI,
        first_page=page_index + 1,
        last_page=page_index + 1,
    )
    if not images:
        return ""
    # lang=None lets Tesseract use its default; pass e.g. "eng+deu+dan"
    # if you want to constrain/boost specific languages.
    kwargs = {"lang": lang} if lang else {}
    return pytesseract.image_to_string(images[0], **kwargs)


def load_and_score_document(
    pdf_path: str,
    ocr_lang: str = None,
) -> tuple[list[Document], DocumentQualityReport]:
    """
    Main entry point: loads a PDF, detects scanned pages, OCRs them,
    scores every page, and returns (documents, quality_report).

    `ocr_lang` lets you pass Tesseract language codes (e.g. "eng+deu+dan")
    if you know the document set's languages in advance. Left as None,
    Tesseract uses its installed default model.
    """
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()  # one Document per page, in order

    report = DocumentQualityReport(total_pages=len(raw_docs))
    enriched_docs: list[Document] = []

    # Track hard OCR failures separately so they can be surfaced loudly —
    # a silently-empty page is far more dangerous than a visibly failed one.
    ocr_errors: list[str] = []

    for i, doc in enumerate(raw_docs):
        digital_text = doc.page_content or ""
        source = "digital"
        used_ocr = False

        if len(digital_text.strip()) < MIN_CHARS_FOR_DIGITAL_PAGE:
            # Likely a scanned/image-only page -> fall back to OCR
            if OCR_AVAILABLE:
                try:
                    ocr_text = _ocr_page(pdf_path, i, lang=ocr_lang)
                except Exception as e:
                    ocr_text = ""
                    doc.metadata["ocr_error"] = str(e)
                    ocr_errors.append(f"Page {i + 1}: {e}")
                final_text = ocr_text
                source = "ocr"
                used_ocr = True
            else:
                final_text = digital_text  # nothing better available
        else:
            final_text = digital_text

        score = _quality_score(final_text)
        flagged = score < QUALITY_FLAG_THRESHOLD

        page_num = i + 1
        pq = PageQuality(
            page_number=page_num,
            source=source,
            char_count=len(final_text.strip()),
            quality_score=score,
            flagged=flagged,
            note="Low-confidence extraction — verify against original document"
                 if flagged else "",
        )
        report.page_details.append(pq)
        if used_ocr:
            report.ocr_pages += 1
        if flagged:
            report.flagged_pages.append(pq)

        # Enrich the document's metadata so the RAG chain / UI can trace
        # every retrieved chunk back to its confidence level.
        doc.page_content = final_text
        doc.metadata.update({
            "page_number": page_num,
            "extraction_source": source,       # "digital" or "ocr"
            "quality_score": score,
            "low_confidence": flagged,
        })
        enriched_docs.append(doc)

    report.ocr_errors = ocr_errors

    # Hard guard: if every single page ended up with empty/near-empty text,
    # don't let this silently propagate into an empty Chroma upsert (which
    # fails with a cryptic "Expected Embeddings to be non-empty" error far
    # away from the real cause). Fail loudly, here, with the actual reason.
    total_chars = sum(len(d.page_content.strip()) for d in enriched_docs)
    if total_chars == 0:
        if ocr_errors:
            detail = " | ".join(ocr_errors[:5])
            raise RuntimeError(
                "No usable text could be extracted from this PDF, and OCR "
                f"failed on every scanned page. First error(s): {detail}\n"
                "This usually means Tesseract or poppler (used by pdf2image) "
                "is not correctly installed/on PATH in this Python environment. "
                "Verify with: `python -c \"import pytesseract; "
                "print(pytesseract.get_tesseract_version())\"` and "
                "`python -c \"from pdf2image import convert_from_path; "
                "convert_from_path('your.pdf')\"`."
            )
        elif not OCR_AVAILABLE:
            raise RuntimeError(
                "No usable text could be extracted from this PDF (it appears "
                "to be scanned/image-based), and OCR support is not available "
                "because pytesseract/pdf2image failed to import. Install them "
                "with: pip install pytesseract pdf2image"
            )
        else:
            raise RuntimeError(
                "No usable text could be extracted from this PDF, even after "
                "attempting OCR. The document may be empty, corrupted, or "
                "contain unsupported content."
            )

    return enriched_docs, report
