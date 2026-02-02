# streamlit_reel_writer_app.py
"""
Streamlit UI for Instagram-style script generation with optional fact checking,
ElevenLabs voiceover generation (preview, regenerate, download), and HeyGen avatar video
rendering using a standard **video avatar** (NO Avatar IV / talking-photo).

Run:
    streamlit run streamlit_reel_writer_app.py

Requirements:
    pip install streamlit openai requests numpy

Environment (can also be entered in the sidebar at runtime):
    - OPENAI_API_KEY
    - TAVILY_API_KEY   (optional; required only if Fact Checking is enabled)
    - ELEVENLABS_API_KEY (for voiceover)
    - HEYGEN_API_KEY     (for HeyGen assets + video)

Auth:
    - APP_USERNAME (in .streamlit/secrets.toml preferred)
    - APP_PASSWORD (in .streamlit/secrets.toml preferred)

Notes:
- For GPT-5 models ("gpt-5", "gpt-5-pro"), we use the Responses API and NEVER send `temperature`.
- HeyGen integration now supports only a **video avatar**:
    * character = {"type": "avatar", "avatar_id": "..."}
- This version prefers the **HeyGen audio-asset flow**:
    1) Generate MP3 with ElevenLabs (in-memory).
    2) Upload MP3 to HeyGen Assets (/v1/asset) → get `asset_id`.
    3) Create video with voice `{ "type": "audio", "audio_asset_id": "<asset_id>" }`.
"""

from __future__ import annotations

import os
import io
import re
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import timedelta

import numpy as np
import requests
import streamlit as st
from openai import OpenAI


# -------------------------------
# Simple login/auth (via st.secrets)
# -------------------------------
def _get_auth_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Fetch login username/password from st.secrets or env.

    Preferred (Streamlit Cloud / local):
        .streamlit/secrets.toml
            APP_USERNAME = "myuser"
            APP_PASSWORD = "supersecret"

    Fallback:
        Environment variables APP_USERNAME / APP_PASSWORD
    """
    user = None
    pwd = None

    # Prefer secrets
    try:
        user = st.secrets.get("APP_USERNAME")  # type: ignore[attr-defined]
        pwd = st.secrets.get("APP_PASSWORD")  # type: ignore[attr-defined]
    except Exception:
        # st.secrets may not be configured (e.g., local without secrets)
        pass

    # Fallback to env if secrets not set
    if not user:
        user = os.getenv("APP_USERNAME")
    if not pwd:
        pwd = os.getenv("APP_PASSWORD")

    return user, pwd


def require_login() -> None:
    """
    Simple username/password gate using Streamlit session_state.

    Call this once near the top of the app; if credentials aren't set
    or login fails, the rest of the app will not run.
    """
    expected_user, expected_pw = _get_auth_credentials()

    if not expected_user or not expected_pw:
        st.error(
            "Authentication is enabled but APP_USERNAME / APP_PASSWORD "
            "are not configured in `.streamlit/secrets.toml` or environment variables."
        )
        st.stop()

    # Already authenticated in this session
    if st.session_state.get("auth_ok"):
        return

    st.title("🔐 Login required")
    st.caption("Please sign in to use the Reel Script Generator.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username == expected_user and password == expected_pw:
            st.session_state["auth_ok"] = True
            # For modern Streamlit:
            st.rerun()
        else:
            st.error("Invalid username or password.")
            st.stop()

    else:
        # Don't show the rest of the app until they submit valid creds
        st.stop()


# -------------------------------
# Defaults / Config
# -------------------------------
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
DEFAULT_GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5")
DEFAULT_VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", ".vector_store"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

TAVILY_API_KEY_ENV = os.getenv("TAVILY_API_KEY")
TAVILY_ENDPOINT = "https://api.tavily.com/search"

# --- ElevenLabs ---
ELEVEN_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY", "")
ELEVEN_BASE = "https://api.elevenlabs.io/v1"

# --- HeyGen ---
HEYGEN_API_KEY_ENV = os.getenv("HEYGEN_API_KEY", "")
HEYGEN_BASE_V2 = "https://api.heygen.com/v2"
HEYGEN_STATUS_V1 = "https://api.heygen.com/v1/video_status.get"


# -------------------------------
# Utilities
# -------------------------------
def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "output"


def is_gpt5(model: str) -> bool:
    return model and model.strip().lower().startswith("gpt-5")


# -------------------------------
# Corpus chunking
# -------------------------------
def load_and_chunk_corpus_text(raw_text: str) -> List[str]:
    """
    Splits the corpus into semantically meaningful chunks:
    - Break on double newlines and "Script X:" markers
    - Fuse tiny fragments to keep chunk size reasonable (200–1200 chars)
    """
    raw = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?:\n\s*\n)|(?:^|\n)(?:Script\s*\d+\s*:)", raw)
    parts = [normalize_text(p) for p in parts if normalize_text(p)]

    fused: List[str] = []
    buf: List[str] = []
    char_target_min = 400
    char_target_max = 1200

    for p in parts:
        if sum(len(x) for x in buf) + len(p) < char_target_min:
            buf.append(p)
        else:
            if buf:
                fused.append(" ".join(buf))
                buf = []
            if len(p) > char_target_max:
                sentences = re.split(r"(?<=[.!?])\s+", p)
                temp: List[str] = []
                cur = ""
                for s in sentences:
                    if len(cur) + len(s) < char_target_max:
                        cur = (cur + " " + s).strip()
                    else:
                        if cur:
                            temp.append(cur)
                        cur = s
                if cur:
                    temp.append(cur)
                fused.extend(temp)
            else:
                fused.append(p)

    if buf:
        fused.append(" ".join(buf))

    chunks = [c for c in fused if len(c) > 150]
    return chunks


# -------------------------------
# Embeddings
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str | None) -> OpenAI:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    try:
        return OpenAI()
    except Exception as e:
        raise RuntimeError("Could not initialize OpenAI client. Provide a valid OPENAI_API_KEY.") from e


def embed_texts(client: OpenAI, texts: List[str], embed_model: str) -> np.ndarray:
    embeddings: List[List[float]] = []
    batch = 128
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=embed_model, input=texts[i : i + batch])
        embeddings.extend([d.embedding for d in resp.data])
    return np.array(embeddings, dtype=np.float32)


@st.cache_data(show_spinner=False)
def build_or_load_index_cached(
    *,
    file_hash: str,
    embed_model: str,
    raw_text: str,
) -> Dict[str, Any]:
    client = get_openai_client(st.session_state.get("openai_api_key"))
    chunks = load_and_chunk_corpus_text(raw_text)
    embeddings = embed_texts(client, chunks, embed_model)
    meta = {
        "file_hash": file_hash,
        "model": embed_model,
        "chunks": chunks,
        "embeddings": embeddings,
        "built_at": datetime.utcnow().isoformat() + "Z",
    }
    return meta


# -------------------------------
# Retrieval & Generation
# -------------------------------
STYLE_SYSTEM_MSG = (
    "You are a concise scriptwriter for short, informative Instagram reels.\n"
    "Your style based on prior examples:\n"
    "- punchy, journalistic lines (1–2 sentences per beat)\n"
    "- make the first line a strog catchy hook for an instagram reel/vide\n"
    "- Pakistan-centric business, culture, tech examples\n"
    "- Stay neutral and positive — never take a stance\n"
    "- dont include lines like \"this info is not in the sources\" in the final script  \n"
    "- no emojis, no hashtags, no clickbait\n"
    "- keep to ~200 words\n"
    "Produce ONE script only, no extra commentary or titles, no markdown.\n"
)


def make_prompt(topic: str, retrieved_chunks: List[str]) -> List[Dict[str, str]]:
    context_block = "\n\n--- EXCERPTED STYLE/CONTENT SAMPLES ---\n" + "\n\n".join(retrieved_chunks)
    user_msg = f"""Topic: {topic}

Using the style learned from the samples below (structure, cadence, sentence length, and tone), write a new script tailored to the topic. Avoid copying lines verbatim. Prefer locally relevant angles (Pakistan when applicable), concrete numbers, and short paragraph beats.

{context_block}
"""
    return [
        {"role": "system", "content": STYLE_SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]


def retrieve_top_k(topic: str, meta: Dict[str, Any], client: OpenAI, embed_model: str, k: int = 15) -> List[int]:
    topic_vec = embed_texts(client, [normalize_text(topic)], embed_model)[0]
    matrix = meta["embeddings"]
    sims = np.dot(matrix, topic_vec) / (np.linalg.norm(matrix, axis=1) * (np.linalg.norm(topic_vec) or 1e-8))
    idxs = np.argsort(-sims)[:k]
    return [int(i) for i in idxs]


def _extract_text_from_responses(resp: Any) -> str:
    """
    Robustly extract assistant text from a Responses API result.

    Supports both Pydantic objects (new SDK) and plain dicts.
    Looks for output_text blocks and pulls their `text.value` or `text`.
    """
    # Helper to pull text from a single content item (object or dict)
    def _content_to_text(c: Any) -> str:
        # dict shape
        if isinstance(c, dict):
            if c.get("type") != "output_text":
                return ""
            text_obj = c.get("text")
            if isinstance(text_obj, dict):
                # typical: {"value": "...", "annotations": [...]}
                val = text_obj.get("value")
                if isinstance(val, str):
                    return val
                return str(val) if val is not None else ""
            if isinstance(text_obj, str):
                return text_obj
            return str(text_obj) if text_obj is not None else ""

        # object shape
        c_type = getattr(c, "type", "")
        if c_type != "output_text":
            return ""
        text_obj = getattr(c, "text", None)
        # text_obj might itself have .value
        if hasattr(text_obj, "value"):
            val = getattr(text_obj, "value")
            if isinstance(val, str):
                return val
            return str(val) if val is not None else ""
        if isinstance(text_obj, str):
            return text_obj
        if text_obj is not None:
            return str(text_obj)
        return ""

    # 1) Try direct attr access: resp.output -> out.content -> content.text
    try:
        output = getattr(resp, "output", None)
        chunks: List[str] = []
        if output:
            for out in output:
                # handle object vs dict for `out`
                if hasattr(out, "content"):
                    content_list = getattr(out, "content") or []
                elif isinstance(out, dict):
                    content_list = out.get("content", []) or []
                else:
                    content_list = []
                for c in content_list:
                    txt = _content_to_text(c)
                    if txt:
                        chunks.append(txt)
        if chunks:
            return "\n".join(chunks).strip()
    except Exception:
        pass

    # 2) Try model_dump() if available (Pydantic models)
    try:
        d = resp.model_dump()  # type: ignore[attr-defined]
    except Exception:
        d = resp if isinstance(resp, dict) else None

    if isinstance(d, dict):
        chunks: List[str] = []
        output = d.get("output") or []
        for out in output:
            content_list = out.get("content", []) or []
            for c in content_list:
                txt = _content_to_text(c)
                if txt:
                    chunks.append(txt)
        if chunks:
            return "\n".join(chunks).strip()

    # 3) Other fallbacks: output_text, text, content
    # Some SDK versions expose a simpler aggregate
    try:
        ot = getattr(resp, "output_text", None)
        if ot:
            return str(ot).strip()
    except Exception:
        pass

    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    if hasattr(resp, "content") and resp.content:
        return str(resp.content).strip()

    return ""




def generate_script(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    if is_gpt5(model):
        resp = client.responses.create(
            model=model,
            input=messages,
            text={"format": {"type": "text"}, "verbosity": "medium"},
            reasoning={"effort": "medium", "summary": "auto"},
            tools=[{
                "type": "web_search",
                "user_location": {"type": "approximate"},
                "search_context_size": "medium",
            }],
            store=True,
            include=["reasoning.encrypted_content", "web_search_call.action.sources"],
        )
        text = _extract_text_from_responses(resp)
        if not text:
            raise RuntimeError("No text content returned by GPT-5 response.")
        return text

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        resp = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=600,
        )
        text = _extract_text_from_responses(resp)
        if text:
            return text
        raise RuntimeError("No text content returned by Responses fallback.")


def fetch_pakistan_people_news_from_web(
    client: OpenAI,
    *,
    kind: str,  # "creators" or "founders"
    days_back: int,
    name_filter: str = "",
    max_items: int = 8,
) -> List[Dict[str, str]]:
    kind_norm = (kind or "").strip().lower()
    if kind_norm not in {"creators", "founders"}:
        raise ValueError("kind must be 'creators' or 'founders'")

    now_utc = datetime.utcnow()
    cutoff = now_utc - timedelta(days=max(1, int(days_back)))
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    name_filter = (name_filter or "").strip()

    if kind_norm == "creators":
        system_msg = (
            "You are a careful Pakistan news researcher.\n"
            "Search the web for recent news on Pakistani YouTubers and Internet personalities.\n"
            "ONLY include positive, educational, inspiring, or uplifting news.\n"
            "Skip scandals/controversy.\n"
            "Return ONLY valid JSON (no markdown, no commentary) in EXACTLY this schema:\n"
            "{\"items\": [{\"name\":\"...\",\"headline\":\"...\",\"summary\":\"...\",\"url\":\"...\",\"published\":\"...\"}]}\n"
        )
        user_msg = (
            f"Find up to {max_items} items from the last {days_back} days (cutoff: {cutoff_str}).\n"
            f"{'Focus on this name: ' + name_filter if name_filter else 'No name filter.'}\n"
            "Prefer reputable sources and official announcements."
        )
    else:
        system_msg = (
            "You are a careful Pakistan business/startup news researcher.\n"
            "Search the web for recent news on Pakistani founders.\n"
            "Return ONLY valid JSON (no markdown, no commentary) in EXACTLY this schema:\n"
            "{\"items\": [{\"name\":\"...\",\"headline\":\"...\",\"summary\":\"...\",\"url\":\"...\",\"published\":\"...\"}]}\n"
        )
        user_msg = (
            f"Find up to {max_items} items from the last {days_back} days (cutoff: {cutoff_str}).\n"
            f"{'Focus on this name: ' + name_filter if name_filter else 'No name filter.'}\n"
            "Prefer credible business/tech sources."
        )

    # ✅ IMPORTANT: web_search cannot be used with JSON mode, so keep output as TEXT
    resp = client.responses.create(
        model="o4-mini",
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        text={"format": {"type": "text"}, "verbosity": "medium"},
        tools=[
            {
                "type": "web_search",
                "user_location": {"type": "approximate", "country": "PK"},
                "search_context_size": "medium",
            }
        ],
        max_output_tokens=900,
    )

    content = _extract_text_from_responses(resp)
    if not content:
        return []

    # Parse JSON from the returned text
    data = None
    try:
        data = json.loads(content)
    except Exception:
        # try to extract the first JSON object from any extra text
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return []

    items = data.get("items") or []
    out: List[Dict[str, str]] = []

    for it in items:
        if not isinstance(it, dict):
            continue

        name = str(it.get("name") or "").strip()
        headline = str(it.get("headline") or "").strip()
        summary = str(it.get("summary") or "").strip()
        url = str(it.get("url") or "").strip()
        published = str(it.get("published") or "").strip()

        if not (name and headline and summary):
            continue

        out.append(
            {"name": name, "headline": headline, "summary": summary, "url": url, "published": published}
        )

    return out[:max_items]



# -------------------------------
# Fact checking (optional)
# -------------------------------
FACT_EXTRACT_SYS = (
    "You extract atomic, checkable claims from short newsy scripts.\n"
    "Rules:\n"
    "- Include claims that can be verified online (names, titles, dates, counts, money, rankings, places), but dont mention things like 'this claim cant be verified from sources' in the final script\n"
    "- Make claims as self-contained sentences.\n"
    "- 3-10 claims total, focus on the most material ones.\n"
    'Output JSON: {"claims": ["...", "...", ...]}'
)

def extract_claims(client: OpenAI, model: str, script: str) -> List[str]:
    msg = [{"role": "system", "content": FACT_EXTRACT_SYS}, {"role": "user", "content": script}]
    if is_gpt5(model):
        resp = client.responses.create(
            model=model,
            input=msg,
            text={"format": {"type": "json_object"}, "verbosity": "medium"},
            reasoning={"effort": "medium", "summary": "auto"},
        )
        content = _extract_text_from_responses(resp)
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=600,
        )
        content = resp.choices[0].message.content

    try:
        data = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0)) if m else {"claims": []}

    claims = [c.strip() for c in data.get("claims", []) if isinstance(c, str) and c.strip()]
    return claims[:10]


FACT_VERIFY_SYS = (
    "You are a meticulous fact checker. Given (1) a claim and (2) web snippets with sources, decide:\n"
    "- VERDICT: \"supported\", \"refuted\", or \"uncertain\"\n"
    "- CORRECTION: if refuted or uncertain, suggest a corrected, precise statement\n"
    "- CONFIDENCE: 0-1\n"
    "- CITES: up to 3 URLs\n"
    'Return JSON: {"verdict":"...", "correction":"...", "confidence":0.0, "citations":["..."]}'
)

def verify_claim(client: OpenAI, model: str, claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    ev_text = "\n\n".join(
        f"- {e.get('title') or ''}\n{e.get('snippet') or ''}\nSource: {e.get('url') or ''}"
        + (f" | Published: {e.get('published')}" if e.get("published") else "")
        for e in evidence[:6]
    )
    messages = [
        {"role": "system", "content": FACT_VERIFY_SYS},
        {"role": "user", "content": f"CLAIM:\n{claim}\n\nEVIDENCE SNIPPETS:\n{ev_text}"},
    ]
    if is_gpt5(model):
        resp = client.responses.create(
            model=model,
            input=messages,
            text={"format": {"type": "json_object"}, "verbosity": "medium"},
            reasoning={"effort": "medium", "summary": "auto"},
        )
        content = _extract_text_from_responses(resp)
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content

    try:
        out = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        out = json.loads(m.group(0)) if m else {}

    out["citations"] = [c for c in out.get("citations", []) if isinstance(c, str) and c.startswith("http")][:3]
    out["verdict"] = (out.get("verdict") or "").lower()
    try:
        out["confidence"] = float(out.get("confidence") or 0.0)
    except Exception:
        out["confidence"] = 0.0
    out["correction"] = (out.get("correction") or "").strip()
    out["claim"] = claim
    return out


def apply_corrections_and_citations(
    client: OpenAI,
    model: str,
    script: str,
    decisions: List[Dict[str, Any]],
    add_inline_citations: bool,
) -> Tuple[str, List[str]]:
    guidance = {
        "decisions": decisions,
        "inline_citations": add_inline_citations,
        "rules": [
            "Preserve cadence.",
            "Change only refuted/uncertain facts.",
            "If inline_citations=true, add [n] markers.",
            "Return only the corrected script.",
        ],
    }
    messages = [
        {"role": "system", "content": "You revise scripts using verified facts and add compact citations."},
        {"role": "user", "content": json.dumps({"script": script, "guidance": guidance})},
    ]

    if is_gpt5(model):
        resp = client.responses.create(
            model=model,
            input=messages,
            text={"format": {"type": "text"}, "verbosity": "medium"},
            reasoning={"effort": "medium", "summary": "auto"},
        )
        revised = _extract_text_from_responses(resp)
    else:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=800)
        revised = resp.choices[0].message.content.strip()

    ordered_urls: List[str] = []
    for d in decisions:
        for u in d.get("citations", []):
            if u not in ordered_urls:
                ordered_urls.append(u)

    if add_inline_citations and ordered_urls:
        biblio_lines = ["", "", "References:"]
        for i, url in enumerate(ordered_urls, start=1):
            biblio_lines.append(f"[{i}] {url}")
        revised = revised.rstrip() + "\n" + "\n".join(biblio_lines) + "\n"

    return revised, ordered_urls


# -------------------------------
# ElevenLabs TTS helpers
# -------------------------------
@st.cache_data(show_spinner=False)
def eleven_list_voices(api_key: str) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    headers = {"xi-api-key": api_key}
    try:
        r = requests.get(f"{ELEVEN_BASE}/voices", headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        voices = data.get("voices", []) or []
        return [
            {
                "id": v.get("voice_id"),
                "name": v.get("name"),
                "category": v.get("category"),
                "labels": v.get("labels") or {},
                "description": v.get("description") or "",
            }
            for v in voices
            if v.get("voice_id") and v.get("name")
        ]
    except Exception as e:
        st.warning(f"Could not load ElevenLabs voices: {e}")
        return []


def eleven_tts(
    api_key: str,
    voice_id: str,
    text: str,
    *,
    model_id: str = "eleven_multilingual_v2",
    stability: float | None = None,
    similarity_boost: float | None = None,
    style: float | None = None,
    use_speaker_boost: bool | None = None,
) -> bytes:
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set.")
    if not voice_id:
        raise RuntimeError("Please select a voice.")

    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }

    payload: Dict[str, Any] = {
        "text": text,
        "model_id": model_id,
    }
    voice_settings: Dict[str, Any] = {}
    if stability is not None:
        voice_settings["stability"] = float(stability)
    if similarity_boost is not None:
        voice_settings["similarity_boost"] = float(similarity_boost)
    if style is not None:
        voice_settings["style"] = float(style)
    if use_speaker_boost is not None:
        voice_settings["use_speaker_boost"] = bool(use_speaker_boost)
    if voice_settings:
        payload["voice_settings"] = voice_settings

    url = f"{ELEVEN_BASE}/text-to-speech/{voice_id}"
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.content


# -------------------------------
# HeyGen API helpers (assets + video)
# -------------------------------
def _heygen_headers_json(api_key: str) -> Dict[str, str]:
    return {
        "X-Api-Key": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _heygen_headers_get(api_key: str) -> Dict[str, str]:
    return {
        "X-Api-Key": api_key,
        "Accept": "application/json",
    }


def heygen_upload_audio_asset(api_key: str, filename: str, audio_bytes: bytes) -> str:
    """
    Upload an MP3 to HeyGen Assets.

    Correct endpoint & format:
      POST https://upload.heygen.com/v1/asset
      Headers: X-Api-Key, Content-Type: audio/mpeg
      Body: raw bytes (NOT multipart)
    Returns the created audio asset id.
    """
    url = "https://upload.heygen.com/v1/asset"
    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "audio/mpeg",
        "Accept": "application/json",
    }

    r = requests.post(url, headers=headers, data=audio_bytes, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"HeyGen audio upload failed: {r.status_code}\n\n{r.text}")

    data = r.json() or {}
    asset_id = (data.get("data") or {}).get("id") or data.get("id")
    if not asset_id:
        raise RuntimeError(f"Upload succeeded but no asset id in response: {data}")
    return asset_id


def heygen_get_avatar_details(api_key: str, avatar_id: str) -> dict:
    """
    Optional preflight to validate a VIDEO avatar id is accessible in this account/workspace.
    """
    r = requests.get(
        f"{HEYGEN_BASE_V2}/avatar/{avatar_id}/details",
        headers=_heygen_headers_get(api_key),
        timeout=20,
    )
    if r.status_code != 404:
        r.raise_for_status()
    else:
        raise RuntimeError(f"HeyGen avatar not found or not accessible: {avatar_id}")
    return r.json() or {}


def heygen_generate_video(
    api_key: str,
    *,
    avatar_id: Optional[str],
    audio_asset_id: Optional[str] = None,
    audio_url: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    background_color: Optional[str] = None,
) -> str:
    """
    Generate a HeyGen video using ONLY a standard video avatar (Avatar IV disabled).
    """
    if not (audio_asset_id or audio_url):
        raise ValueError("Provide either audio_asset_id or audio_url for HeyGen voice.")

    if not avatar_id:
        raise ValueError("Provide avatar_id (video avatar). Talking-photo (Avatar IV) is disabled in this build.")

    character = {"type": "avatar", "avatar_id": avatar_id, "avatar_style": "normal"}

    voice: Dict[str, Any] = {"type": "audio"}
    if audio_asset_id:
        voice["audio_asset_id"] = audio_asset_id
    if audio_url:
        voice["audio_url"] = audio_url

    video_input: Dict[str, Any] = {"character": character, "voice": voice}
    if background_color:
        video_input["background"] = {"type": "color", "value": background_color}

    payload = {"video_inputs": [video_input], "dimension": {"width": width, "height": height}}

    r = requests.post(f"{HEYGEN_BASE_V2}/video/generate", headers=_heygen_headers_json(api_key), data=json.dumps(payload), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HeyGen video generation failed: {r.status_code}\n{r.text}")
    data = r.json() or {}
    vid = (data.get("data", {}) or {}).get("video_id") or data.get("video_id")
    if not vid:
        raise RuntimeError(f"No video_id in HeyGen response: {data}")
    return vid


def heygen_wait_for_video(api_key: str, video_id: str, poll_seconds: int = 5, timeout_seconds: int = 900) -> Dict[str, Any]:
    start = time.time()
    while True:
        r = requests.get(f"{HEYGEN_STATUS_V1}?video_id={video_id}", headers=_heygen_headers_get(api_key), timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HeyGen status check failed: {r.status_code}\n{r.text}")
        data = r.json() or {}
        status = (data.get("data", {}) or {}).get("status") or data.get("status")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"HeyGen reported failure: {json.dumps(data, indent=2)}")
        if (time.time() - start) > timeout_seconds:
            raise TimeoutError("Timed out waiting for HeyGen video to complete.")
        time.sleep(poll_seconds)


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Reel Script Generator", page_icon="🎬", layout="wide")

# 🔐 Require login BEFORE showing the rest of the app
require_login()

st.title("🎬 Reel Script Generator")
st.caption("Generate scripts, voice them with ElevenLabs, then upload MP3 to HeyGen Assets and render the avatar video (video avatars only — Avatar IV disabled).")

# Sidebar: keys & settings
with st.sidebar:
    st.header("Settings")

    # Optional logout
    if st.session_state.get("auth_ok"):
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

    openai_api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    st.session_state["openai_api_key"] = openai_api_key

    tavily_api_key = st.text_input("TAVILY_API_KEY (optional; for Fact Check)", value=TAVILY_API_KEY_ENV or "", type="password")
    st.session_state["tavily_api_key"] = tavily_api_key

    st.divider()
    embed_model = st.selectbox("Embedding model", [DEFAULT_EMBED_MODEL, "text-embedding-3-large"], index=0)
    gen_model = st.selectbox(
        "Generation model",
        [DEFAULT_GEN_MODEL, "gpt-4o", "gpt-4o-mini", "o4-mini", "gpt-5", "gpt-5-pro"],
        index=0,
    )
    is_g5 = is_gpt5(gen_model)
    temperature = st.slider("Temperature", 0.0, 1.2, 0.7, 0.1, disabled=is_g5)
    if is_g5:
        st.caption("Note: GPT-5 ignores temperature. Using Responses API with reasoning & web_search tool.")

    st.divider()
    k = st.slider("Top-k retrieved chunks", 1, 20, 15)

    max_ctx_chars = st.number_input("Max context characters", min_value=500, max_value=20000, value=6000, step=100)

    st.divider()
    fact_check = st.toggle("Enable Fact Checking", value=False)
    citations = st.toggle("Add inline citations & references", value=True, disabled=not fact_check)
    max_claims = st.slider("Max claims to check", 1, 10, 8, disabled=not fact_check)
    force_search_refresh = st.toggle("Force fresh web search", value=False, disabled=not fact_check)

    st.divider()
    # --- ElevenLabs UI ---
    st.subheader("Voiceover (ElevenLabs)")
    eleven_api_key = st.text_input("ELEVENLABS_API_KEY", value=ELEVEN_API_KEY_ENV, type="password")
    st.session_state["eleven_api_key"] = eleven_api_key

    stability = 0.5
    similarity_boost = 0.75
    style = 0.0
    use_speaker_boost = True

    voices = eleven_list_voices(eleven_api_key) if eleven_api_key else []
    voice_names = [f"{v['name']}  ·  {v.get('category') or ''}".strip() for v in voices] if voices else []
    voice_idx = st.selectbox(
        "Voice",
        list(range(len(voice_names))) if voice_names else [],
        format_func=lambda i: voice_names[i],
        index=0 if voice_names else None,
        placeholder="Login key first to load voices…",
        disabled=not voices,
    )
    st.caption("Tip: Use multilingual v2 for Urdu/English mixed content.")
    with st.popover("Advanced voice settings"):
        stability = st.slider("Stability", 0.0, 1.0, stability, 0.05)
        similarity_boost = st.slider("Similarity Boost", 0.0, 1.0, similarity_boost, 0.05)
        style = st.slider("Style", 0.0, 1.0, style, 0.05)
        use_speaker_boost = st.toggle("Use Speaker Boost", value=use_speaker_boost)

    st.session_state["eleven_voice_id"] = voices[voice_idx]["id"] if voices and voice_idx is not None else None
    st.session_state["eleven_settings"] = {
        "stability": stability if voices else None,
        "similarity_boost": similarity_boost if voices else None,
        "style": style if voices else None,
        "use_speaker_boost": use_speaker_boost if voices else None,
    }

    st.divider()
    # --- HeyGen UI ---
    st.subheader("HeyGen Video")
    heygen_api_key = st.text_input("HEYGEN_API_KEY", value=HEYGEN_API_KEY_ENV, type="password")
    st.session_state["heygen_api_key"] = heygen_api_key

    st.caption("Select a VIDEO avatar (avatar_id). Talking-photo (Avatar IV) is disabled in this build.")
    avatar_id_input = st.text_input("HeyGen AVATAR ID (video avatar)", value="d53434113ef240969f03d1fd376176e0", placeholder="e.g. 920e0e2ea96e4fcf8d3cc9a7457840bf")
    st.session_state["avatar_id_input"] = avatar_id_input.strip()

    engine = "Video Avatar (Avatar IV disabled)"
    st.caption(f"Engine: {engine}")

    heygen_width = st.number_input("Video Width", min_value=320, max_value=1920, value=1280, step=10)
    heygen_height = st.number_input("Video Height", min_value=240, max_value=1080, value=720, step=10)
    heygen_bg = st.text_input("Background color (hex, optional)", value="#000000")

    # (Optional) If you already have an audio asset id or a public URL, you can paste it here.
    audio_asset_id_input = st.text_input("Use existing HeyGen audio_asset_id (optional)", value="")
    audio_url_input = st.text_input("Or use a public audio_url (optional)", value="", placeholder="https://.../voiceover.mp3")

    # Persist UI state
    st.session_state.update(
        {
            "heygen_width": heygen_width,
            "heygen_height": heygen_height,
            "heygen_bg": heygen_bg,
            "audio_asset_id_input": audio_asset_id_input.strip(),
            "audio_url_input": audio_url_input.strip(),
        }
    )

# Early key tip
if not st.session_state.get("openai_api_key"):
    st.info("Add your OPENAI_API_KEY in the sidebar to begin.")

# -------------------------------
# 1) Corpus from repo (scripts.txt)
# -------------------------------
st.subheader("1) Corpus (loaded from repo)")

raw_text = ""
source_label = ""
corpus_path = Path(__file__).parent / "scripts.txt"

if corpus_path.exists():
    try:
        raw_text = corpus_path.read_text(encoding="utf-8", errors="ignore")
        source_label = f"repo:{corpus_path.name}"
        file_hash = sha256_bytes(raw_text.encode("utf-8"))
        st.success(
            f"Loaded corpus from `{corpus_path.name}` • {len(raw_text):,} characters • hash {file_hash[:8]}…"
        )
        with st.spinner("Building embeddings index (cached by corpus+model)…"):
            try:
                meta = build_or_load_index_cached(file_hash=file_hash, embed_model=embed_model, raw_text=raw_text)
                st.session_state["meta"] = meta
                st.caption(
                    f"Indexed {len(meta['chunks'])} chunks • model {meta['model']} • built {meta['built_at']}"
                )
            except Exception as e:
                st.error(f"Failed to build/load index: {e}")
    except Exception as e:
        st.error(f"Error reading scripts.txt from repo: {e}")
else:
    st.error("Could not find `scripts.txt` in the app directory. Add it to the repo to proceed.")

# -------------------------------
# 2) Generate a script
# -------------------------------
st.subheader("2) Generate a script")

# Mode selector
mode = st.radio(
    "Choose input for script",
    ["Type a topic", "Find news"],
    horizontal=True,
    key="script_input_mode",
)

topic = ""

# ---- MODE A: manual topic ----
if mode == "Type a topic":
    topic = st.text_input(
        "Topic",
        placeholder="e.g., Pakistan's fintech wave in 2025",
        key="topic_input",
    )

# ---- MODE B: news -> select one ----
else:
    st.caption("Search recent Pakistani founders/creators news, select one item, then generate a script from it.")

    # Controls for web news finder
    cA, cB, cC = st.columns([1, 1, 2])
    st.session_state["news_kind"] = cA.radio(
        "Type",
        ["Creators", "Founders"],
        index=0 if st.session_state.get("news_kind", "Creators") == "Creators" else 1,
        key="news_kind_radio",
    )

    st.session_state["news_range"] = cB.selectbox(
        "Date range",
        ["One day", "One week", "One month", "3 months", "6 months", "One year"],
        index=["One day", "One week", "One month", "3 months", "6 months", "One year"].index(
            st.session_state.get("news_range", "One week")
        ),
        key="news_range_select",
    )

    st.session_state["news_name_filter"] = cC.text_input(
        "Name filter (optional)",
        value=st.session_state.get("news_name_filter", ""),
        placeholder="e.g., 'Mooroo' or 'Monis Rahman'",
        key="news_name_filter_input",
    )

    # Buttons row
    colg = st.columns([1, 1, 3])
    suggest_btn = colg[0].button("Find news", use_container_width=True)
    refresh_note = colg[1].toggle("Auto-refresh when changing filters", value=False, key="news_autorefresh")

    # Optional auto-refresh: rerun search when filters change
    if refresh_note:
        # create a simple signature of filters
        sig = f"{st.session_state['news_kind']}|{st.session_state['news_range']}|{st.session_state['news_name_filter']}"
        if st.session_state.get("news_filter_sig") != sig:
            st.session_state["news_filter_sig"] = sig
            suggest_btn = True  # trigger fetch

    if suggest_btn:
        if not st.session_state.get("openai_api_key"):
            st.error("Add your OPENAI_API_KEY in the sidebar to fetch news.")
        else:
            try:
                client = get_openai_client(st.session_state.get("openai_api_key"))

                days_map = {
                    "One day": 1,
                    "One week": 7,
                    "One month": 30,
                    "3 months": 90,
                    "6 months": 180,
                    "One year": 365,
                }
                days_back = days_map.get(st.session_state["news_range"], 7)
                kind_key = "creators" if st.session_state["news_kind"].lower().startswith("creator") else "founders"

                with st.spinner("Searching the web…"):
                    items = fetch_pakistan_people_news_from_web(
                        client,
                        kind=kind_key,
                        days_back=days_back,
                        name_filter=st.session_state.get("news_name_filter", ""),
                        max_items=8,
                    )

                if not items:
                    st.warning("No items found. Try widening the date range or removing the name filter.")
                st.session_state["web_news_items"] = items
                st.toast("Loaded news from web", icon="📰")

            except Exception as e:
                st.error(f"Fetching news failed: {e}")

    items = st.session_state.get("web_news_items") or []
    if not items:
        st.info("Click **Find news** to load results.")
    else:
        labels = [f"{it.get('name','')} — {it.get('headline','')}"[:160] for it in items]
        selected_idx = st.radio(
            "Select one story",
            list(range(len(labels))),
            format_func=lambda i: labels[i],
            key="web_news_pick_radio",
        )
        chosen = items[selected_idx] if 0 <= selected_idx < len(items) else None

        if chosen:
            st.markdown(f"**Name:** {chosen.get('name','')}")
            st.markdown(f"**Headline:** {chosen.get('headline','')}")
            st.markdown(f"**Summary:** {chosen.get('summary','')}")
            if chosen.get("published"):
                st.markdown(f"**Published:** {chosen.get('published')}")
            if chosen.get("url"):
                st.markdown(f"**Source:** {chosen.get('url')}")

            # This becomes the "topic" used for generation
            headline = (chosen.get("headline") or "").strip()
            summary = (chosen.get("summary") or "").strip()
            url = (chosen.get("url") or "").strip()

            # Make topic richer than just headline (helps model stay anchored)
            topic = headline
            if summary:
                topic += f" — {summary}"
            if url:
                topic += f" (Source: {url})"

# Shared Generate button (works for BOTH modes)
colg2 = st.columns([1, 4])
clicked = colg2[0].button("Generate script", type="primary", use_container_width=True)

# Output containers
script_box = st.empty()
facts_expander = st.expander("Fact-check results", expanded=False)



def download_buttons_area(text: str, topic: str, facts_payload: Dict[str, Any] | None):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{ts}-{slugify(topic)}"
    txt_name = f"{base}.txt"
    st.download_button("Download script (.txt)", text, file_name=txt_name, mime="text/plain")
    if facts_payload is not None:
        json_name = f"{base}-facts.json"
        st.download_button(
            "Download fact report (.json)",
            json.dumps(facts_payload, ensure_ascii=False, indent=2),
            file_name=json_name,
            mime="application/json",
        )

if clicked and not topic:
    st.warning("Please enter a topic or select a news item first.")
    st.stop()


if clicked and topic and st.session_state.get("meta"):
    # Retrieval
    client = get_openai_client(st.session_state.get("openai_api_key"))
    meta = st.session_state["meta"]

    with st.spinner("Retrieving top chunks…"):
        top = retrieve_top_k(topic, meta, client, embed_model, k=k)
        retrieved: List[str] = []
        running = 0
        for idx in top:
            chunk = meta["chunks"][idx]
            if running + len(chunk) <= max_ctx_chars:
                retrieved.append(chunk)
                running += len(chunk)
        if not retrieved and top:
            retrieved.append(meta["chunks"][top[0]])


    with st.spinner("Generating script…"):
        messages = make_prompt(topic, retrieved)
        try:
            script = generate_script(client, gen_model, messages, temperature=temperature)
            st.session_state.pop("audio_bytes", None)
            st.session_state.pop("heygen_video_url", None)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    # Optional fact checking
    bibliography: List[str] = []
    facts_payload: Dict[str, Any] | None = None
    if fact_check:
        if not (st.session_state.get("tavily_api_key") or TAVILY_API_KEY_ENV):
            st.warning("Enable Fact Checking requires TAVILY_API_KEY in the sidebar.")
        else:
            st.toast("Fact checking enabled", icon="🔎")
            # Claim extraction
            try:
                claims = extract_claims(client, gen_model, script)[: max_claims]
            except Exception as e:
                st.error(f"Claim extraction failed: {e}")
                claims = []
            if claims:
                facts_table_rows = []
                progress = st.progress(0.0, text="Verifying claims…")

                def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
                    key = st.session_state.get("tavily_api_key") or TAVILY_API_KEY_ENV
                    payload = {
                        "api_key": key,
                        "query": f"{query} Pakistan",
                        "max_results": max_results,
                        "search_depth": "advanced",
                        "include_answer": False,
                        "include_images": False,
                        "include_raw_content": False,
                    }
                    r = requests.post(TAVILY_ENDPOINT, json=payload, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    results = data.get("results", [])
                    return [
                        {
                            "title": it.get("title"),
                            "url": it.get("url"),
                            "snippet": it.get("content"),
                            "published": it.get("published_date"),
                        }
                        for it in results
                    ]

                decisions: List[Dict[str, Any]] = []
                for i, c in enumerate(claims, start=1):
                    try:
                        evidence = tavily_search(c)
                        decision = verify_claim(client, gen_model, c, evidence)
                    except Exception as e:
                        decision = {"claim": c, "verdict": "error", "correction": str(e), "confidence": 0.0, "citations": []}
                    decisions.append(decision)
                    cites = ", ".join(decision.get("citations", [])[:2]) or "—"
                    facts_table_rows.append({"#": i, "Claim": c, "Verdict": decision.get("verdict", ""), "Confidence": f"{decision.get('confidence', 0):.2f}", "Citations": cites})
                    progress.progress(i / len(claims))
                progress.empty()

                try:
                    script, bibliography = apply_corrections_and_citations(client, gen_model, script, decisions, add_inline_citations=citations)
                except Exception as e:
                    st.warning(f"Could not auto-apply corrections: {e}")

                facts_payload = {
                    "topic": topic,
                    "claims_checked": claims,
                    "decisions": decisions,
                    "citations_flat": bibliography,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "source_corpus": source_label,
                }

                with facts_expander:
                    st.write("Claim verification summary:")
                    st.dataframe(facts_table_rows, use_container_width=True, hide_index=True)
                    if bibliography:
                        st.markdown("**References**")
                        for i, url in enumerate(bibliography, start=1):
                            st.markdown(f"[{i}] {url}")

    st.session_state.update(
        {"has_script": True, "last_script": script, "last_topic": topic, "last_facts_payload": facts_payload}
    )

    current_script_text = script_box.text_area("Generated script", value=script, height=320, key="generated_script_text")
    st.caption("Tip: Edit above before generating audio or downloading.")
    download_buttons_area(current_script_text, topic, facts_payload)

    # -------------------------------
    # 3) Voiceover (ElevenLabs)
    # -------------------------------
    st.subheader("3) Voiceover (ElevenLabs)")
    if not st.session_state.get("eleven_api_key"):
        st.info("Add ELEVENLABS_API_KEY in the sidebar to enable audio generation.")
    elif not st.session_state.get("eleven_voice_id"):
        st.warning("Select a voice in the sidebar.")
    else:
        c1, c2, _ = st.columns([1, 1, 2])
        make_audio = c1.button("Generate Audio", type="primary", use_container_width=True)
        regen_audio = c2.button("Regenerate", use_container_width=True)
        do_tts = make_audio or regen_audio
        tts_text = st.session_state.get("generated_script_text") or current_script_text or script

        if do_tts:
            try:
                with st.spinner("Generating voiceover…"):
                    audio_bytes = eleven_tts(
                        st.session_state["eleven_api_key"],
                        st.session_state["eleven_voice_id"],
                        tts_text,
                        model_id="eleven_multilingual_v2",
                        **(st.session_state.get("eleven_settings") or {}),
                    )
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state.pop("heygen_video_url", None)
                st.toast("Audio ready", icon="🔊")
            except Exception as e:
                st.error(f"Audio generation failed: {e}")

        if st.session_state.get("audio_bytes"):
            st.audio(st.session_state["audio_bytes"], format="audio/mpeg")
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = f"{ts}-{slugify(topic)}"
            mp3_name = f"{base}.mp3"
            st.download_button("Download audio (.mp3)", st.session_state["audio_bytes"], file_name=mp3_name, mime="audio/mpeg", use_container_width=True)
        else:
            st.caption("No audio yet. Generate to preview.")

    # -------------------------------
    # 4) Video (HeyGen) — upload MP3 to HeyGen, then render with audio_asset_id
    # -------------------------------
    st.subheader("4) Render Video (HeyGen)")
    if not st.session_state.get("heygen_api_key"):
        st.info("Add HEYGEN_API_KEY in the sidebar to enable video rendering.")
    else:
        colv1, colv2, _ = st.columns([1, 1, 2])
        btn_render = colv1.button("Upload MP3 to HeyGen & Make Video", type="primary", use_container_width=True)
        btn_rerender = colv2.button("Re-render (reuse inputs)", use_container_width=True)

        if btn_render or btn_rerender:
            # Priority order for audio:
            # 1) user-provided audio_asset_id
            # 2) user-provided audio_url
            # 3) generated ElevenLabs audio_bytes -> upload to HeyGen Assets
            chosen_asset_id = (st.session_state.get("audio_asset_id_input") or "").strip()
            chosen_audio_url = (st.session_state.get("audio_url_input") or "").strip()

            if not chosen_asset_id and not chosen_audio_url:
                if not st.session_state.get("audio_bytes"):
                    st.error("No ElevenLabs audio found. Generate audio first, or paste an existing HeyGen audio_asset_id / public audio_url.")
                else:
                    try:
                        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                        filename = f"{slugify(topic) or 'voiceover'}-{ts}.mp3"
                        with st.spinner("Uploading MP3 to HeyGen Assets…"):
                            chosen_asset_id = heygen_upload_audio_asset(
                                st.session_state["heygen_api_key"],
                                filename,
                                st.session_state["audio_bytes"],
                            )
                        st.toast(f"Uploaded to HeyGen Assets (asset_id: {chosen_asset_id[:8]}…)", icon="⬆️")
                    except Exception as e:
                        st.error(f"HeyGen audio upload failed: {e}")

            # Character selection (video avatar only)
            avatar_id = st.session_state.get("avatar_id_input")

            try:
                # Preflight for VIDEO avatars
                if avatar_id:
                    heygen_get_avatar_details(st.session_state["heygen_api_key"], avatar_id)
                else:
                    st.error("Please provide a HeyGen AVATAR ID (video avatar).")
                    st.stop()

                with st.spinner("Starting HeyGen video generation…"):
                    video_id = heygen_generate_video(
                        st.session_state["heygen_api_key"],
                        avatar_id=avatar_id or None,
                        audio_asset_id=chosen_asset_id or None,
                        audio_url=chosen_audio_url or None,
                        width=int(st.session_state["heygen_width"]),
                        height=int(st.session_state["heygen_height"]),
                        background_color=(st.session_state["heygen_bg"] or "").strip() or None,
                    )

                with st.spinner("Waiting for HeyGen to finish…"):
                    final_status = heygen_wait_for_video(
                        st.session_state["heygen_api_key"],
                        video_id,
                        poll_seconds=5,
                        timeout_seconds=900,
                    )
                video_url = (final_status.get("data", {}) or {}).get("video_url") or final_status.get("video_url")
                if not video_url:
                    raise RuntimeError(f"No video_url found in final status: {json.dumps(final_status, indent=2)}")
                st.session_state["heygen_video_url"] = video_url
                st.toast("HeyGen video ready", icon="🎬")
            except Exception as e:
                st.error(f"HeyGen rendering failed: {e}")

        if st.session_state.get("heygen_video_url"):
            st.video(st.session_state["heygen_video_url"])
            st.link_button("Open video in new tab", st.session_state["heygen_video_url"], use_container_width=True)
        else:
            st.caption("No HeyGen video yet. Render to preview.")

elif clicked and not topic:
    st.warning("Enter a topic first.")
elif clicked and not st.session_state.get("meta"):
    st.warning("Corpus index not ready. Ensure `scripts.txt` is present and index built successfully.")

# If user didn't click this run, keep previous results available (so buttons work after rerun)
if not clicked and st.session_state.get("has_script"):
    script = st.session_state.get("last_script", "")
    topic = st.session_state.get("last_topic", "")
    facts_payload = st.session_state.get("last_facts_payload")

    current_script_text = script_box.text_area("Generated script", value=script, height=320, key="generated_script_text")
    st.caption("Tip: Edit above before generating audio or downloading.")
    download_buttons_area(current_script_text, topic, facts_payload)

    st.subheader("3) Voiceover (ElevenLabs)")
    if not st.session_state.get("eleven_api_key"):
        st.info("Add ELEVENLABS_API_KEY in the sidebar to enable audio generation.")
    elif not st.session_state.get("eleven_voice_id"):
        st.warning("Select a voice in the sidebar.")
    else:
        c1, c2, _ = st.columns([1, 1, 2])
        make_audio = c1.button("Generate Audio", type="primary", use_container_width=True)
        regen_audio = c2.button("Regenerate", use_container_width=True)

        if make_audio or regen_audio:
            try:
                with st.spinner("Generating voiceover…"):
                    audio_bytes = eleven_tts(
                        st.session_state["eleven_api_key"],
                        st.session_state["eleven_voice_id"],
                        current_script_text or script,
                        model_id="eleven_multilingual_v2",
                        **(st.session_state.get("eleven_settings") or {}),
                    )
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state.pop("heygen_video_url", None)
                st.toast("Audio ready", icon="🔊")
            except Exception as e:
                st.error(f"Audio generation failed: {e}")

        if st.session_state.get("audio_bytes"):
            st.audio(st.session_state["audio_bytes"], format="audio/mpeg")
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = f"{ts}-{slugify(topic)}"
            mp3_name = f"{base}.mp3"
            st.download_button("Download audio (.mp3)", st.session_state["audio_bytes"], file_name=mp3_name, mime="audio/mpeg", use_container_width=True)
        else:
            st.caption("No audio yet. Generate to preview.")

    st.subheader("4) Render Video (HeyGen)")
    if not st.session_state.get("heygen_api_key"):
        st.info("Add HEYGEN_API_KEY in the sidebar to enable video rendering.")
    else:
        colv1, colv2, _ = st.columns([1, 1, 2])
        btn_render = colv1.button("Upload MP3 to HeyGen & Make Video", type="primary", use_container_width=True)
        btn_rerender = colv2.button("Re-render (reuse inputs)", use_container_width=True)

        if btn_render or btn_rerender:
            chosen_asset_id = (st.session_state.get("audio_asset_id_input") or "").strip()
            chosen_audio_url = (st.session_state.get("audio_url_input") or "").strip()

            if not chosen_asset_id and not chosen_audio_url:
                if not st.session_state.get("audio_bytes"):
                    st.error("No ElevenLabs audio found. Generate audio first, or paste an existing HeyGen audio_asset_id / public audio_url.")
                else:
                    try:
                        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                        filename = f"{slugify(topic) or 'voiceover'}-{ts}.mp3"
                        with st.spinner("Uploading MP3 to HeyGen Assets…"):
                            chosen_asset_id = heygen_upload_audio_asset(
                                st.session_state["heygen_api_key"],
                                filename,
                                st.session_state["audio_bytes"],
                            )
                        st.toast(f"Uploaded to HeyGen Assets (asset_id: {chosen_asset_id[:8]}…)", icon="⬆️")
                    except Exception as e:
                        st.error(f"HeyGen audio upload failed: {e}")

            avatar_id = st.session_state.get("avatar_id_input")

            try:
                if avatar_id:
                    heygen_get_avatar_details(st.session_state["heygen_api_key"], avatar_id)
                else:
                    st.error("Please provide a HeyGen AVATAR ID (video avatar).")
                    st.stop()

                with st.spinner("Starting HeyGen video generation…"):
                    video_id = heygen_generate_video(
                        st.session_state["heygen_api_key"],
                        avatar_id=avatar_id or None,
                        audio_asset_id=chosen_asset_id or None,
                        audio_url=chosen_audio_url or None,
                        width=int(st.session_state["heygen_width"]),
                        height=int(st.session_state["heygen_height"]),
                        background_color=(st.session_state["heygen_bg"] or "").strip() or None,
                    )

                with st.spinner("Waiting for HeyGen to finish…"):
                    final_status = heygen_wait_for_video(
                        st.session_state["heygen_api_key"],
                        video_id,
                        poll_seconds=5,
                        timeout_seconds=900,
                    )
                video_url = (final_status.get("data", {}) or {}).get("video_url") or final_status.get("video_url")
                if not video_url:
                    raise RuntimeError(f"No video_url found in final status: {json.dumps(final_status, indent=2)}")
                st.session_state["heygen_video_url"] = video_url
                st.toast("HeyGen video ready", icon="🎬")
            except Exception as e:
                st.error(f"HeyGen rendering failed: {e}")

        if st.session_state.get("heygen_video_url"):
            st.video(st.session_state["heygen_video_url"])
            st.link_button("Open video in new tab", st.session_state["heygen_video_url"], use_container_width=True)
        else:
            st.caption("No HeyGen video yet. Render to preview.")
