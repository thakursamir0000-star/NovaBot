"""Groq LLM calls: follow-up rewriting, grounded answers with citations, and citation checks."""
import logging
import os
import re

logger = logging.getLogger(__name__)

PRIMARY_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",") if m.strip()]
HISTORY_TURNS = 6
EXCERPT_CHARS = 1200
NOT_FOUND = "I couldn't find this in the book."

_unavailable_models = set()

SYSTEM_PROMPT = """You are an expert assistant helping users understand {book}.

Rules:
- Answer ONLY using the numbered context passages.
- Cite the passages you use, e.g. [Source 1] or [Source 2][Source 3].
- If the passages do not contain the answer, reply with exactly: "{not_found}" and nothing else.
- Be concise and accurate; use bullet points for lists and steps.
- Use the conversation history only to understand what the user is referring to."""

CONDENSE_PROMPT = """Rewrite the user's latest message as a standalone search query for a book, \
resolving words like "it", "that" or "more" from the conversation. Keep names and technical terms. \
Reply with the query only.

Conversation:
{history}

Latest message: {question}

Standalone query:"""


def get_client():
    """Groq client from GROQ_API_KEY (env or Streamlit secrets), or None."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st

            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            api_key = ""
    api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        return None
    from groq import Groq

    return Groq(api_key=api_key)


def candidate_models():
    models = []
    for m in [PRIMARY_MODEL, *FALLBACK_MODELS]:
        if m not in models and m not in _unavailable_models:
            models.append(m)
    return models


def _record_failure(model, error):
    if getattr(error, "status_code", None) == 404 or "model_not_found" in str(error):
        _unavailable_models.add(model)
        logger.warning("Model %s is not available on this account; skipping it from now on", model)
    else:
        logger.warning("Model %s failed: %s", model, error)


def _kwargs(model, **extra):
    if model.startswith("openai/gpt-oss"):
        extra["reasoning_effort"] = "low"
    return {"model": model, **extra}


def complete(client, messages, temperature=0.2, max_tokens=1024):
    """Non-streaming completion with failover. Returns (text, usage) or (None, None)."""
    for model in candidate_models():
        try:
            response = client.chat.completions.create(
                **_kwargs(model, messages=messages, temperature=temperature, max_tokens=max_tokens))
            return response.choices[0].message.content or "", getattr(response, "usage", None)
        except Exception as e:
            _record_failure(model, e)
    return None, None


def stream_answer(client, messages, temperature=0.2, max_tokens=1024):
    """Yield answer text pieces. Falls back to the next model if a model fails before
    producing any text; a failure mid-answer ends the stream with a short note."""
    for model in candidate_models():
        started = False
        try:
            stream = client.chat.completions.create(
                **_kwargs(model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True))
            for part in stream:
                delta = part.choices[0].delta.content if part.choices else None
                if delta:
                    started = True
                    yield delta
            return
        except Exception as e:
            _record_failure(model, e)
            if started:
                yield "\n\n*(The answer was cut off by a model error — please ask again.)*"
                return
    yield "Sorry — the language model is unavailable right now. Please try again in a minute."


def format_history(chat_history, turns=HISTORY_TURNS):
    return [{"role": m["role"], "content": m["content"]} for m in (chat_history or [])[-turns:]]


def condense_question(client, question, chat_history):
    """Turn a follow-up ("explain that more") into a standalone search query.

    Returns the original question when there is no history or the call fails.
    """
    history = format_history(chat_history)
    if not history or client is None:
        return question
    transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history)
    text, _ = complete(client, [{"role": "user", "content": CONDENSE_PROMPT.format(history=transcript, question=question)}],
                       temperature=0.0, max_tokens=400)
    text = (text or "").strip().strip('"')
    return text if 3 <= len(text) <= 400 else question


def build_messages(question, chunks, chat_history=None, book_title=None):
    """System prompt + recent history + the question with numbered context passages."""
    book = f'"{book_title}"' if book_title else "the uploaded book"
    context = "\n\n---\n\n".join(
        f"[Source {i} | Pages {c['start_page'] + 1}–{c['end_page'] + 1} | {c['heading']}]\n{c['text'][:EXCERPT_CHARS]}"
        for i, c in enumerate(chunks, 1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(book=book, not_found=NOT_FOUND)}]
    messages += format_history(chat_history)
    messages.append({"role": "user", "content": f"Context from {book}:\n{context}\n\nQuestion: {question}\n\nAnswer:"})
    return messages


# "Source 1", "Sources 2 and 4", "Source 1, 3", "Sources 1-3", in any brackets ([...], (...), 【...】).
_CITATION = re.compile(r"\bsources?\s*#?\s*(\d+(?:\s*(?:,|&|and|-|–)\s*\d+)*)", re.IGNORECASE)
# gpt-oss style "【1†source】" / "【2†L3-L5】".
_DAGGER_CITATION = re.compile(r"【\s*(\d+)\s*†")
_NUMBER_OR_RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)|(\d+)")


def _cited_numbers(answer):
    cited = {int(n) for n in _DAGGER_CITATION.findall(answer)}
    for group in _CITATION.findall(answer):
        for start, end, single in _NUMBER_OR_RANGE.findall(group):
            if single:
                cited.add(int(single))
            elif int(end) - int(start) < 20:
                cited.update(range(int(start), int(end) + 1))
    return cited


def check_citations(answer, n_sources):
    """Which sources an answer cites, and whether any citation points past the provided passages."""
    cited = sorted(_cited_numbers(answer or ""))
    return {
        "cited": cited,
        "invalid": [n for n in cited if not 1 <= n <= n_sources],
        "abstained": is_abstention(answer),
    }


def is_abstention(answer):
    text = (answer or "").strip().lower()
    return text.startswith(NOT_FOUND.lower().rstrip(".")) or "couldn't find this in the book" in text


def reset_unavailable_models():
    _unavailable_models.clear()
