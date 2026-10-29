import llm
from tests.conftest import FakeClient, FakeStream, make_chunk


class Missing(Exception):
    status_code = 404


def test_check_citations():
    result = llm.check_citations("Every 90 days [Source 1][source 3], see [Source 7].", n_sources=5)
    assert result == {"cited": [1, 3, 7], "invalid": [7], "abstained": False}


def test_abstention_detection():
    assert llm.is_abstention("I couldn't find this in the book.")
    assert llm.is_abstention("  i couldn't find this in the book")
    assert not llm.is_abstention("The filter is replaced every 90 days [Source 1].")


def test_build_messages_numbers_sources_and_explains_abstention():
    chunks = [make_chunk("alpha text here", 0), make_chunk("beta text here", 4, heading="Chapter 2: B")]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    messages = llm.build_messages("What is beta?", chunks, history, book_title="Test Book")
    assert messages[0]["role"] == "system" and llm.NOT_FOUND in messages[0]["content"]
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
