import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings import HashingEmbedder  # noqa: E402
from vector_store import InMemoryStore  # noqa: E402


def make_chunk(text, page=0, heading="Chapter 1: Test"):
    return {"heading": heading, "text": text, "start_page": page, "end_page": page}


@pytest.fixture
def chunks():
    return [
        make_chunk("The sediment cartridges are replaced every 90 days in the pump house.", 0),
        make_chunk("Drones may not fly higher than 120 m above the ground near the colony.", 1),
        make_chunk("Black bears are active from April to November; use the steel bear lockers.", 2),
        make_chunk("The emergency radio frequency is 151.625 MHz, programmed as channel 3.", 3),
        make_chunk("Breakfast is served from 06:30 to 08:00 and dinner starts at 18:30.", 4),
    ]


@pytest.fixture
def index(chunks):
    from retrieval import build_index

    return build_index(chunks, "doc-1", InMemoryStore(), embedder=HashingEmbedder())


class FakeStream:
    def __init__(self, pieces, fail_after=None):
        self.pieces, self.fail_after = pieces, fail_after

    def __iter__(self):
        for i, piece in enumerate(self.pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("connection reset")
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])


class FakeClient:
    """Stands in for groq.Groq: `responder(kwargs)` returns text, a FakeStream, or an Exception."""

    def __init__(self, responder):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._responder = responder

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responder(kwargs)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, FakeStream):
            return result
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result))],
            usage=SimpleNamespace(total_tokens=42),
        )


@pytest.fixture(autouse=True)
def _reset_models():
    import llm

    llm.reset_unavailable_models()
