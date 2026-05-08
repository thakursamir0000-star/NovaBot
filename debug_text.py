"""Quick diagnostic: find what's in the chunk text that breaks the tokenizer."""
import sys, os, tempfile
from ingest import extract_text_by_page, chunk_into_sections
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Get PDF path from command line
pdf_path = sys.argv[1]

pages = extract_text_by_page(pdf_path)
chunks = chunk_into_sections(pages)

print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    text = chunk['text']
    text_type = type(text).__name__
    text_repr = repr(text[:100]) if text else repr(text)
    
    # Check for problematic characters
    has_null = '\x00' in str(text) if text else False
    has_surrogates = False
    if isinstance(text, str):
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            has_surrogates = True
    
    if not text or not isinstance(text, str) or has_null or has_surrogates or not text.strip():
        print(f"  PROBLEM chunk {i}: type={text_type}, null={has_null}, surrogates={has_surrogates}, empty={not text}, repr={text_repr}")
    
    # Try encoding
    try:
        clean = str(text).encode('utf-8', errors='replace').decode('utf-8').strip()
        if clean:
            embedder.encode([clean], show_progress_bar=False)
    except Exception as e:
        print(f"  ENCODE FAIL chunk {i}: {e}")
        print(f"    type={text_type}, len={len(text) if text else 0}, repr={text_repr}")

print("Done - all chunks tested.")
