import pytest

import llm
from tests.conftest import FakeClient, FakeStream, make_chunk


class Missing(Exception):
    status_code = 404


def test_check_citations():
    result = llm.check_citations("Every 90 days [Source 1][source 3], see [Source 7].", n_sources=5)
    assert result == {"cited": [1, 3, 7], "invalid": [7], "abstained": False}


@pytest.mark.parametrize("answer, cited", [
    ("Every 90 days [Source 2].", [2]),
    ("Every 90 days 【Source 2】.", [2]),
    ("Every 90 days (Source 2).", [2]),
    ("Every 90 days [Source 1, 3].", [1, 3]),
    ("Every 90 days [Sources 2 and 4].", [2, 4]),
    ("Every 90 days [Sources 1-3].", [1, 2, 3]),
    ("Every 90 days 【1†source】 and 【4†L2-L3】.", [1, 4]),
    ("Every 90 days, see Source 5 | Pages 3–4.", [5]),
    ("Every 90 days.", []),
    ("Replaced 2-3 times a year (no source).", []),
])
def test_citation_formats(answer, cited):
    assert llm.check_citations(answer, n_sources=5)["cited"] == cited


def test_abstention_detection():
    assert llm.is_abstention("I couldn't find this in the book.")
    assert llm.is_abstention("  i couldn't find this in the book")
    assert llm.is_abstention("I couldn’t find this in the book.")  # curly apostrophe
    assert not llm.is_abstention("The filter is replaced every 90 days [Source 1].")


def test_build_messages_numbers_sources_and_explains_abstention():
    chunks = [make_chunk("alpha text here", 0), make_chunk("beta text here", 4, heading="Chapter 2: B")]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    messages = llm.build_messages("What is beta?", chunks, history, book_title="Test Book")
    assert messages[0]["role"] == "system" and llm.NOT_FOUND in messages[0]["content"]
    assert "citation after every sentence" in messages[0]["content"]
    assert messages[1:3] == history
    assert "[Source 2 | Pages 5–5 | Chapter 2: B]" in messages[-1]["content"]


def test_condense_question_uses_history():
    client = FakeClient(lambda kwargs: "How often are the sediment cartridges replaced?")
    history = [{"role": "user", "content": "Tell me about the water filter"},
               {"role": "assistant", "content": "It has three stages."}]
    assert llm.condense_question(client, "how often is it changed?", history) == \
        "How often are the sediment cartridges replaced?"


def test_condense_question_without_history_makes_no_call():
    client = FakeClient(lambda kwargs: "should not be used")
    assert llm.condense_question(client, "What is Polaris?", []) == "What is Polaris?"
    assert client.calls == []


def test_condense_question_falls_back_on_failure():
    client = FakeClient(lambda kwargs: RuntimeError("down"))
    history = [{"role": "user", "content": "x"}]
    assert llm.condense_question(client, "and then?", history) == "and then?"


def test_stream_fails_over_before_first_token():
    def responder(kwargs):
        if kwargs["model"] == llm.PRIMARY_MODEL:
            return Missing("model_not_found")
        return FakeStream(["Every ", "90 days."])

    client = FakeClient(responder)
    assert "".join(llm.stream_answer(client, [])) == "Every 90 days."
    # The missing model is skipped next time.
    "".join(llm.stream_answer(client, []))
    assert [c["model"] for c in client.calls].count(llm.PRIMARY_MODEL) == 1


def test_stream_failure_mid_answer_adds_a_note():
    client = FakeClient(lambda kwargs: FakeStream(["Every ", "90 ", "days"], fail_after=2))
    text = "".join(llm.stream_answer(client, []))
    assert text.startswith("Every 90 ") and "cut off" in text
    assert len(client.calls) == 1


def test_stream_when_everything_is_down():
    client = FakeClient(lambda kwargs: RuntimeError("down"))
    assert "unavailable" in "".join(llm.stream_answer(client, []))


def test_gpt_oss_uses_low_reasoning_effort():
    client = FakeClient(lambda kwargs: "ok")
    text, usage = llm.complete(client, [{"role": "user", "content": "hi"}])
    assert text == "ok" and usage.total_tokens == 42
    assert client.calls[0]["reasoning_effort"] == "low"


@pytest.fixture
def no_llm_env(monkeypatch):
    for var in ("LLM_PROVIDER", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_setting", lambda name: llm.os.environ.get(name, ""))  # no Streamlit secrets
    return monkeypatch


def test_provider_follows_the_configured_key(no_llm_env):
    assert llm.provider_name() == "groq"
    assert llm.get_client() is None
    no_llm_env.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert llm.provider_name() == "openrouter"
    client = llm.get_client()
    assert llm.is_openrouter(client) and client.timeout == 60
    no_llm_env.setenv("GROQ_API_KEY", "gsk-test")
    assert llm.provider_name() == "groq"  # both keys: Groq unless LLM_PROVIDER says otherwise
    no_llm_env.setenv("LLM_PROVIDER", "openrouter")
    assert llm.provider_name() == "openrouter"
    no_llm_env.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        llm.provider_name()


def openrouter_fake(responder):
    client = FakeClient(responder)
    client.base_url = "https://openrouter.ai/api/v1/"
    return client


def test_openrouter_requests_carry_routing_and_reasoning_in_the_body():
    client = openrouter_fake(lambda kwargs: FakeStream(["Hi ", "there"]))
    assert "".join(llm.stream_answer(client, [])) == "Hi there"
    body = client.calls[0]["extra_body"]
    assert "reasoning_effort" not in client.calls[0]
    assert body == {"provider": {"data_collection": "deny", "ignore": ["google-vertex"]},
                    "reasoning": {"effort": "low"}}


def test_openrouter_unknown_model_is_skipped():
    class InvalidModel(Exception):
        status_code = 400

    def responder(kwargs):
        if kwargs["model"] == llm.PRIMARY_MODEL:
            return InvalidModel(f"Error code: 400 - {llm.PRIMARY_MODEL} is not a valid model ID")
        return "ok"

    client = openrouter_fake(responder)
    assert llm.complete(client, [{"role": "user", "content": "hi"}])[0] == "ok"
    assert llm.PRIMARY_MODEL not in llm.candidate_models()


# A review showed answers near the end of a correctly retrieved passage were cut
# off: only the first 1,200 characters of each ~3,000-character chunk were sent.
FILLER = " ".join(f"Section {i} describes routine office procedures and general guidance for staff." for i in range(22))


def test_excerpt_keeps_short_passages_whole():
    assert llm.excerpt("A short passage.", "anything") == "A short passage."


def test_excerpt_finds_an_answer_near_the_end_of_a_long_passage():
    passage = FILLER + " The launch password is ORCHID."
    assert len(passage) > llm.EXCERPT_CHARS
    piece = llm.excerpt(passage, "What is the launch password?")
    assert "ORCHID" in piece and len(piece) <= llm.EXCERPT_CHARS + 4 and piece.startswith("…")
    messages = llm.build_messages("What is the launch password?", [make_chunk(passage)])
    assert "ORCHID" in messages[-1]["content"]


def test_excerpt_without_matching_words_keeps_the_beginning():
    passage = FILLER + " The launch password is ORCHID."
    assert llm.excerpt(passage, "zzz qqq").startswith("Section 0")
