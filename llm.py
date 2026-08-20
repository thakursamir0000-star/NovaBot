import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def _get_client():
    """Lazy client — reads key at call time, strips whitespace."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not set. Add it in Settings → Secrets.")
        st.stop()
    return Groq(api_key=api_key)


def generate_answer(query: str, context_chunks: list,
                    chat_history: list = None, book_title: str = None):
    """
    Sends query + context to GROQ openai/gpt-oss-120b.
    Returns a streaming response object.
    """
    client = _get_client()

    # ── Build context string from retrieved chunks ───────────────────────
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        topics = ", ".join(chunk['topic_words']) if chunk['topic_words'] else "general"
        page_info = f"Pages {chunk['start_page']+1}\u2013{chunk['end_page']+1}"
        excerpt = chunk['text'][:1200]
        context_parts.append(
            f"[Source {i} | {page_info} | Topics: {topics}]\n{excerpt}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    # ── Dynamic system prompt ────────────────────────────────────────────
    book_ref = f'"{book_title}"' if book_title else "the uploaded book"

    system_prompt = f"""You are an expert AI assistant helping users understand {book_ref}.

Rules:
- Answer ONLY using the provided context passages.
- Always cite source numbers, e.g. [Source 1], [Source 3].
- If the context does not contain enough information, say so clearly and suggest what section might help.
- Be concise, accurate, and technical when appropriate.
- Use bullet points for lists and steps.
- When explaining concepts, provide clear examples where possible.
- If the user asks a follow-up question, use the conversation history for context."""

    # ── Build messages with conversation memory ──────────────────────────
    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        recent_history = chat_history[-6:]
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    user_prompt = f"""Context from {book_ref}:
{context_str}

Question: {query}

Answer:"""

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        stream=True
    )

    return response


def stream_to_string(stream_response) -> str:
    """Helper to collect a full streamed response as a plain string."""
    result = ""
    for chunk in stream_response:
        delta = chunk.choices[0].delta.content
        if delta:
            result += delta
    return result
