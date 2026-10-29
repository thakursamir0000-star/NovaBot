from ingest import chunk_with_overlap, document_id, safe_overlap


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


def test_overlap_larger_than_chunk_size_does_not_explode():
    pages = {0: words(1000)}
    chunks = chunk_with_overlap(pages, chunk_size=200, overlap=300)
    assert safe_overlap(200, 300) == 100
    assert len(chunks) == 9  # used to be ~800 one-word steps


def test_every_word_is_kept_including_the_tail():
    pages = {0: words(205)}
    chunks = chunk_with_overlap(pages, chunk_size=100, overlap=0)
    seen = set(" ".join(c["text"] for c in chunks).split())
    assert seen == set(words(205).split())


def test_page_spans_are_tracked():
    pages = {0: words(60), 1: words(60, 60)}
    chunks = chunk_with_overlap(pages, chunk_size=80, overlap=10)
    assert chunks[0]["start_page"] == 0 and chunks[0]["end_page"] == 1


def test_headings_carry_forward_but_mentions_do_not_count():
    pages = {
        0: "Chapter 1: Getting Started\n" + words(100) + "\nAs we saw, see Chapter 5 for details " + words(30, 100),
        1: "Chapter 2: Field Work\n" + words(100, 200),
    }
    chunks = chunk_with_overlap(pages, chunk_size=50, overlap=10)
    headings = [c["heading"] for c in chunks]
    assert headings[0] == "Chapter 1: Getting Started"
    assert "Chapter 2: Field Work" in headings
    assert not any("see Chapter 5" in h for h in headings)


def test_document_id_changes_with_chunk_settings():
    pdf = b"%PDF-1.4 example"
    assert document_id(pdf, 500, 100) == document_id(pdf, 500, 100)
    assert document_id(pdf, 500, 100) != document_id(pdf, 400, 100)
    assert document_id(pdf, 500, 100) != document_id(pdf, 500, 50)
