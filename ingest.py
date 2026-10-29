"""PDF ingestion: page text, book title, and sliding-window chunks."""
import hashlib
import re

from pypdf import PdfReader

# A heading is a short line that starts with Chapter/Part/Section/Appendix + a number.
# Checked per line, so in-text mentions like "see Chapter 5" don't count.
_HEADING = re.compile(r"^(chapter|part|section|appendix)\s+([0-9]+|[ivxlc]+|[a-z])\b.{0,70}$", re.IGNORECASE)


def document_id(pdf_bytes, chunk_size, overlap):
    """Stable ID for one PDF *and* one chunking setup.

    Including the chunk settings means changing a slider builds a new index
    instead of reusing vectors that belong to differently cut chunks.
    """
    digest = hashlib.sha1(pdf_bytes).hexdigest()[:12]
    return f"{digest}-{chunk_size}-{overlap}"


def extract_text_by_page(pdf_path):
    """Return {page_number (0-based): page_text}."""
    reader = PdfReader(pdf_path)
    pages = {}
    for i, page in enumerate(reader.pages):
        try:
            pages[i] = page.extract_text() or ""
        except Exception as e:  # one broken page shouldn't sink the whole book
            print(f"[ingest] Could not read page {i + 1}: {e}")
            pages[i] = ""
    print(f"[ingest] Extracted {len(pages)} pages.")
    return pages


def extract_book_title(pdf_path):
    """Title from PDF metadata, else the first substantial line of page 1."""
    try:
        reader = PdfReader(pdf_path)
        metadata = reader.metadata
        if metadata and metadata.title and len(metadata.title.strip()) > 3:
            return metadata.title.strip()
        first_page = reader.pages[0].extract_text() or ""
        lines = [line.strip() for line in first_page.split("\n") if line.strip()]
        for line in lines[:8]:
            if 10 < len(line) < 120 and not line.startswith(("http", "www")):
                return line
    except Exception as e:
        print(f"[ingest] Could not extract title: {e}")
    return "the uploaded book"


def safe_overlap(chunk_size, overlap):
    """Overlap must stay well below chunk size, or the window barely moves and
    the chunk count explodes (overlap >= size used to mean a chunk per word)."""
    return max(0, min(int(overlap), int(chunk_size) // 2))


def chunk_with_overlap(pages, chunk_size=500, overlap=100):
    """Sliding-window chunks of `chunk_size` words with `overlap` words shared.

    Each chunk records its page span and the most recent heading seen before it,
    so sources can show "Chapter 3: ..." instead of only page numbers.

    Returns a list of {heading, text, start_page, end_page}.
    """
    overlap = safe_overlap(chunk_size, overlap)
    words = []  # (word, page, heading)
    heading = None
    for page_num in sorted(pages):
        for line in (pages[page_num] or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if _HEADING.match(line):
                heading = line[:80]
            words.extend((w, page_num, heading) for w in line.split())

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        first_page, last_page = window[0][1], window[-1][1]
        if len(window) < 20 and chunks:
            # A tiny tail: fold the words the previous chunk doesn't have into it.
            new_words = words[max(start, chunks[-1]["_end"]):start + chunk_size]
            chunks[-1]["text"] += " " + " ".join(w for w, _, _ in new_words)
            chunks[-1]["end_page"] = last_page
        else:
            chunks.append({
                "heading": window[0][2] or f"Pages {first_page + 1}–{last_page + 1}",
                "text": " ".join(w for w, _, _ in window),
                "start_page": first_page,
                "end_page": last_page,
                "_end": start + len(window),
            })
        if start + chunk_size >= len(words):
            break

    for chunk in chunks:
        chunk.pop("_end", None)
    print(f"[ingest] Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return chunks
