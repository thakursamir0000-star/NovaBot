import PyPDF2
import re


def extract_text_by_page(pdf_path: str) -> dict:
    """
    Reads every page of the PDF.
    Returns {page_number (int): page_text (str)}
    """
    pages = {}
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        total = len(reader.pages)
        for i in range(total):
            text = reader.pages[i].extract_text()
            pages[i] = text if text else ""
    print(f"[ingest] Extracted {total} pages.")
    return pages


def extract_book_title(pdf_path: str) -> str:
    """
    Try to extract book title from PDF metadata or first page content.
    Returns a human-readable title string.
    """
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # Try PDF metadata first
            metadata = reader.metadata
            if metadata and metadata.title and len(metadata.title.strip()) > 3:
                return metadata.title.strip()
            # Fall back to first page — take first substantial line
            first_page = reader.pages[0].extract_text() or ""
            lines = [l.strip() for l in first_page.split('\n') if l.strip()]
            for line in lines[:8]:
                if 10 < len(line) < 120 and not line.startswith(('http', 'www')):
                    return line
    except Exception as e:
        print(f"[ingest] Could not extract title: {e}")
    return "the uploaded book"


def chunk_with_overlap(pages: dict, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Sliding-window chunking with configurable overlap.

    Improvements over the old regex-based approach:
      1. Works with ANY PDF, not just one book format
      2. Overlapping windows prevent context loss at chunk boundaries
      3. Tracks page spans accurately per chunk
      4. Detects chapter/section headings as metadata (not for splitting)

    Args:
        pages: {page_num: text} from extract_text_by_page
        chunk_size: number of words per chunk
        overlap: number of overlapping words between consecutive chunks

    Returns:
        list of dicts: {heading, text, start_page, end_page}
    """
    # Build a list of (word, page_num) to track page boundaries
    word_pages = []
    for page_num in sorted(pages.keys()):
        text = pages[page_num]
        if not text:
            continue
        words = text.split()
        for word in words:
            word_pages.append((word, page_num))

    if not word_pages:
        return []

    # Heading patterns — used for labeling, NOT for splitting
    heading_pattern = re.compile(
        r'(?:^|\s)(Chapter\s+\d+[^a-z]{0,60}'
        r'|Part\s+[IVX\d]+[^a-z]{0,60}'
        r'|Appendix\s+[A-Z][^a-z]{0,60}'
        r'|CHAPTER\s+\d+[^a-z]{0,60})',
        re.IGNORECASE
    )

    # Build chunks with sliding window
    chunks = []
    total_words = len(word_pages)
    step = max(chunk_size - overlap, 1)

    i = 0
    while i < total_words:
        end = min(i + chunk_size, total_words)
        chunk_words = word_pages[i:end]

        text = " ".join(w for w, _ in chunk_words)
        start_page = chunk_words[0][1]
        end_page = chunk_words[-1][1]

        # Try to find a heading in this chunk for context
        heading_match = heading_pattern.search(text)
        if heading_match:
            heading = heading_match.group(0).strip()[:80]
        else:
            heading = f"Section (Pages {start_page + 1}\u2013{end_page + 1})"

        # Skip tiny fragments
        if len(text.split()) >= 20:
            chunks.append({
                "heading": heading,
                "text": text,
                "start_page": start_page,
                "end_page": end_page,
            })

        if end >= total_words:
            break
        i += step

    print(f"[ingest] Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return chunks
