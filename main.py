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

import numpy as np
import requests
import streamlit as st
from openai import OpenAI
from datetime import datetime, timedelta


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

def _news_item_label(it: dict, idx: int) -> str:
    name = (it.get("Name") or "").strip()
    headline = (it.get("Headline") or "").strip()
    published = (it.get("Published") or "").strip()
    pub = f" — {published}" if published else ""
    who = f"{name}: " if name else ""
    return f"{idx+1}. {who}{headline}{pub}"

def _news_item_anchor(it: dict) -> str:
    # Compact, script-safe “fact anchor”
    name = (it.get("Name") or "").strip()
    headline = (it.get("Headline") or "").strip()
    summary = (it.get("Summary") or "").strip()
    published = (it.get("Published") or "").strip()
    url = (it.get("URL") or "").strip()

    lines = []
    if headline:
        lines.append(f"Headline: {headline}")
    if name:
        lines.append(f"Person/Org: {name}")
    if published:
        lines.append(f"Date: {published}")
    if summary:
        lines.append(f"Summary: {summary}")
    if url:
        lines.append(f"Source: {url}")
    return "\n".join(lines).strip()

def _news_item_topic(it: dict) -> str:
    # This becomes the “topic” string used for retrieval similarity & naming outputs
    name = (it.get("Name") or "").strip()
    headline = (it.get("Headline") or "").strip()
    if name and headline:
        return f"{name} — {headline}"
    return headline or name or "Selected news story"


def _date_range_from_option(opt: str) -> tuple[str, str]:
    """
    Returns ISO date strings (YYYY-MM-DD) for (start_date, end_date) based on a UI option.
    End date is inclusive (today).
    """
    today = datetime.now().date()

    mapping = {
        "One day": 1,
        "One week": 7,
        "One month": 30,
        "3 months": 90,
        "6 months": 180,
        "One year": 365,
    }
    days = mapping.get(opt, 30)
    start = today - timedelta(days=days)
    return start.isoformat(), today.isoformat()

def _safe_json_load(s: str) -> dict:
    if not s:
        return {}

    # 1) Try direct
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Try extracting from ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 3) Balanced-brace extraction for first JSON object
    start = s.find("{")
    if start == -1:
        return {}

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return {}
    return {}



def _looks_unsafe_creators_text(t: str) -> bool:
    """
    Basic post-filter for creators news to avoid harmful/explicit/deepfake scandal content.
    Not perfect, but helps catch mistakes.
    """
    bad = [
        "deepfake", "porn", "pornography", "explicit", "sex", "sexual", "nude", "nudity",
        "onlyfans", "leak", "leaked", "scandal", "harassment", "assault", "rape",
        "gore", "violent", "suicide", "self-harm",
    ]
    x = (t or "").lower()
    return any(b in x for b in bad)


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
NEWS_SYS = (
    "You are a web research assistant. Return ONLY valid JSON.\n"
    "Rules:\n"
    "- Only include real, recent news you found on the web.\n"
    "- Do not hallucinate.\n"
    "- Summaries must be 1–2 sentences.\n"
)

def find_news(
    client: OpenAI,
    *,
    mode: str,  # "Creators" or "Founders"
    date_range_label: str,
    name_filter: str = "",
    max_items: int = 10,
) -> tuple[list[dict], dict]:
    """
    Fetches uplifting/positive recent news about Pakistani creators or founders using web_search via gpt.
    Returns list of items with: name, headline, summary, url, published.
    """
    start_date, end_date = _date_range_from_option(date_range_label)

    # Build a query that nudges the web tool toward the requested timeframe.
    # (Web search tools don't always guarantee date filtering, so we enforce in instructions too.)
    base_query = "Pakistan" if not name_filter.strip() else f"Pakistan {name_filter.strip()}"

    if mode == "Creators":
        query = (
            f"{base_query} YouTuber OR \"content creator\" OR influencer "
            f"positive OR educational OR inspiring news {start_date}..{end_date}"
        )
        extra_rules = (
            "Only include positive, educational, or uplifting news.\n"
            "Do NOT include any 18+ content, deepfake scandals, or harmful/explicit material.\n"
        )
        who = "Pakistani YouTubers and internet personalities"
    else:
        query = (
            f"{base_query} founder OR startup founder OR entrepreneur "
            f"funding OR award OR expansion OR partnership news {start_date}..{end_date}"
        )
        extra_rules = ""
        who = "Pakistani founders"

    user_prompt = f"""
Task: Search the web for recent news on {who}.

DATE RANGE: {start_date} to {end_date} (inclusive). If a result has no clear date, include it only if it is obviously within this range.

{extra_rules}

If a name filter is provided, prioritize results about that person: "{name_filter.strip()}"

Return JSON with this schema:
{{
  "items": [
    {{
      "name": "Full name",
      "headline": "News headline",
      "summary": "1-2 sentence summary",
      "url": "Source URL",
      "published": "YYYY-MM-DD or empty string"
    }}
  ]
}}

Constraints:
- Items must be about Pakistani {mode.lower()} (or clearly Pakistan-based).
- Avoid duplicates.
- Prefer credible sources (major news sites, reputable business/tech outlets, official announcements).
- Return at most {max_items} items.
""".strip()

    resp = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": NEWS_SYS},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{
            "type": "web_search",
            "user_location": {"type": "approximate"},
            "search_context_size": "medium",
        }],
        text={"format": {"type": "text"}, "verbosity": "medium"},

        max_output_tokens=2400,
        include=["web_search_call.action.sources"],
        store=False,
    )

    raw = _extract_text_from_responses(resp) or getattr(resp, "output_text", "") or ""
    if not raw:
        try:
            raw = json.dumps(resp.model_dump(), ensure_ascii=False)
        except Exception:
            raw = str(resp)

    data = _safe_json_load(raw) or {}

    if not isinstance(data, dict):
        data = {}
    
    items = data.get("items") or []
    if not isinstance(items, list):
        items = []

    with st.expander("🔎 Debug: raw Find News output", expanded=False):
        st.code(raw or "(empty)")
        st.write("Parsed keys:", list(data.keys()))


    # Normalize + minimal validation
    out: list[dict] = []
    for it in items[:max_items]:
        if not isinstance(it, dict):
            continue
        name = (it.get("name") or "").strip()
        headline = (it.get("headline") or "").strip()
        summary = (it.get("summary") or "").strip()
        url = (it.get("url") or "").strip()
        published = (it.get("published") or "").strip()

        if not name or not headline or not summary:
            continue

        # Post-filter creators for safety (belt + suspenders)
        if mode == "Creators":
            combo = f"{name} {headline} {summary}"
            if _looks_unsafe_creators_text(combo):
                continue

        out.append(
            {
                "Name": name,
                "Headline": headline,
                "Summary": summary,
                "Published": published,
                "URL": url,
            }
        )

    return out, {"raw": raw, "data": data}


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

REVISION_SYSTEM_MSG = (
    "You are GPT-5 acting as a script editor for short Instagram reels. "
    "Preserve the voice, cadence, and tonality from the original script and style samples. "
    "Only change what the user asks, keep the hook strong, and keep length similar unless asked to change it. "
    "Stay neutral and positive, no emojis, no hashtags, no clickbait. "
    "Return only the revised script, no commentary or markdown."
)


def make_prompt(topic: str, retrieved_chunks: List[str], news_anchor: str | None = None) -> List[Dict[str, str]]:
    context_block = "\n\n--- EXCERPTED STYLE/CONTENT SAMPLES ---\n" + "\n\n".join(retrieved_chunks)

    anchor_block = ""
    if news_anchor:
        anchor_block = (
            "\n\n--- SELECTED NEWS ITEM (FACT ANCHOR) ---\n"
            f"{news_anchor}\n"
            "\nRules for the selected news item:\n"
            "- Build the reel around THIS specific item.\n"
            "- Do not invent facts beyond what's in the anchor.\n"
            "- If you add context, keep it general and non-claimy (no numbers/dates unless in anchor).\n"
            "- Do not mention 'sources' or 'fact anchor' in the script.\n"
        )

    user_msg = f"""Topic: {topic}

Using the style learned from the samples below (structure, cadence, sentence length, and tone), write a new script tailored to the topic. Avoid copying lines verbatim. Prefer locally relevant angles (Pakistan when applicable), concrete numbers, and short paragraph beats.
{anchor_block}
{context_block}
"""
    return [
        {"role": "system", "content": STYLE_SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]


def build_revision_messages(
    *,
    original_script: str,
    current_script: str,
    request: str,
    style_samples: List[str] | None,
    recent_requests: List[str] | None,
) -> List[Dict[str, str]]:
    style_block = ""
    if style_samples:
        style_block = "\n\n--- STYLE SAMPLES (tone reference) ---\n" + "\n\n".join(style_samples)

    history_block = ""
    if recent_requests:
        trimmed = recent_requests[-6:]
        history_block = "\n\n--- PRIOR CHANGE REQUESTS (most recent last) ---\n" + "\n".join(f"- {r}" for r in trimmed)

    user_msg = f"""You will revise the CURRENT script using the user's request.

Rules:
- Keep the voice, cadence, and tonality from the original script and style samples.
- Preserve meaning and factual content unless the request changes it.
- Keep length similar unless the request asks to shorten or expand.
- Return ONLY the revised script, no commentary or markdown.

ORIGINAL SCRIPT:
{original_script}

CURRENT SCRIPT:
{current_script}

USER REQUEST:
{request}
{history_block}
{style_block}
"""

    return [
        {"role": "system", "content": REVISION_SYSTEM_MSG},
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
            ctype = c.get("type")
            if ctype not in ("output_text", "text"):
                return ""
            text_obj = c.get("text")
            if isinstance(text_obj, dict):
                val = text_obj.get("value") or text_obj.get("text")
                if isinstance(val, str):
                    return val
                return str(val) if val is not None else ""
            if isinstance(text_obj, str):
                return text_obj
            return str(text_obj) if text_obj is not None else ""


        # object shape
        c_type = getattr(c, "type", "")
        if c_type not in ("output_text", "text"):
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


def revise_script_gpt5(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    resp = client.responses.create(
        model="gpt-5",
        input=messages,
        text={"format": {"type": "text"}, "verbosity": "medium"},
        reasoning={"effort": "medium", "summary": "auto"},
        store=False,
    )
    text = _extract_text_from_responses(resp)
    if not text:
        raise RuntimeError("No text content returned by GPT-5 revision response.")
    return text.strip()






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
            st.rerun()

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
    avatar_id_input = st.text_input("HeyGen AVATAR ID (video avatar)", value="", placeholder="e.g. 920e0e2ea96e4fcf8d3cc9a7457840bf")
    st.session_state["avatar_id_input"] = avatar_id_input.strip()

    engine = "Video Avatar (Avatar IV disabled)"
    st.caption(f"Engine: {engine}")

    heygen_width = st.number_input("Video Width", min_value=320, max_value=1920, value=1280, step=10)
    heygen_height = st.number_input("Video Height", min_value=240, max_value=1080, value=720, step=10)
    heygen_bg = st.text_input("Background color (hex, optional)", value="#000000")

    audio_asset_id_input = st.text_input("Use existing HeyGen audio_asset_id (optional)", value="")
    audio_url_input = st.text_input("Or use a public audio_url (optional)", value="", placeholder="https://.../voiceover.mp3")

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
# Helpers for rendering (render ONCE per rerun)
# -------------------------------
script_box = st.empty()
news_box = st.container()
facts_expander = st.expander("Fact-check results", expanded=False)

def set_script_state(*, script: str, topic: str, facts_payload: dict | None, style_samples: List[str] | None = None):
    st.session_state.update(
        {
            "has_script": True,
            "last_script": script,
            "last_topic": topic,
            "last_facts_payload": facts_payload,
            "original_script": script,
            "script_revision_history": [script],
            "script_revision_requests": [],
            "last_style_samples": (style_samples or [])[:6],
            "generated_script_text": script,
            "original_script_view": script,
            "script_revision_request": "",
        }
    )

def download_buttons_area(text: str, topic: str, facts_payload: dict | None):
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


def render_script_editor_once():
    """
    The ONLY place in the app that renders the text_area with key='generated_script_text'.
    This prevents StreamlitDuplicateElementKey.
    """
    if not st.session_state.get("has_script"):
        return

    script = st.session_state.get("last_script", "") or ""
    topic = st.session_state.get("last_topic", "") or ""
    facts_payload = st.session_state.get("last_facts_payload")

    current_script_text = script_box.text_area(
        "Generated script",
        value=script,
        height=320,
        key="generated_script_text",
    )
    st.caption("Tip: Edit above before generating audio or downloading.")
    download_buttons_area(current_script_text, topic, facts_payload)


def render_revision_controls():
    st.subheader("3) Refine the script")

    if not st.session_state.get("has_script"):
        st.caption("Generate a script first.")
        return

    if not st.session_state.get("openai_api_key"):
        st.info("Add your OPENAI_API_KEY in the sidebar to enable GPT-5 revisions.")
        return

    current_script = st.session_state.get("generated_script_text") or st.session_state.get("last_script") or ""
    original_script = st.session_state.get("original_script") or current_script

    with st.expander("Original script (saved)", expanded=False):
        st.text_area(
            "Original script",
            value=original_script,
            height=240,
            key="original_script_view",
            disabled=True,
        )

    st.caption("Ask for tweaks. GPT-5 keeps tone/cadence from the corpus and original script.")
    revision_request = st.text_area(
        "Change request",
        key="script_revision_request",
        placeholder="e.g., reduce length by ~15% and tighten the last paragraph",
    )

    col1, col2, _ = st.columns([1, 1, 2])
    apply_changes = col1.button(
        "Apply changes",
        type="primary",
        use_container_width=True,
        key="btn_apply_revision",
    )
    reset_original = col2.button(
        "Reset to original",
        use_container_width=True,
        key="btn_reset_original",
    )

    if apply_changes:
        if not revision_request.strip():
            st.warning("Enter a change request first.")
            return

        try:
            client = get_openai_client(st.session_state.get("openai_api_key"))
            recent_requests = st.session_state.get("script_revision_requests") or []
            style_samples = st.session_state.get("last_style_samples") or []
            messages = build_revision_messages(
                original_script=original_script,
                current_script=current_script,
                request=revision_request.strip(),
                style_samples=style_samples,
                recent_requests=recent_requests,
            )
            with st.spinner("Applying changes with GPT-5…"):
                revised = revise_script_gpt5(client, messages)
        except Exception as e:
            st.error(f"Revision failed: {e}")
            return

        history = st.session_state.get("script_revision_history") or [original_script]
        history.append(revised)
        st.session_state["script_revision_history"] = history
        st.session_state["script_revision_requests"] = recent_requests + [revision_request.strip()]
        st.session_state["last_script"] = revised
        st.session_state["generated_script_text"] = revised
        st.session_state["last_facts_payload"] = None
        st.session_state["script_revision_request"] = ""
        st.session_state.pop("audio_bytes", None)
        st.session_state.pop("heygen_video_url", None)
        st.toast("Script updated", icon="✏️")
        st.rerun()

    if reset_original:
        if not original_script:
            st.warning("No original script saved yet.")
            return

        st.session_state["last_script"] = original_script
        st.session_state["generated_script_text"] = original_script
        st.session_state["script_revision_history"] = [original_script]
        st.session_state["script_revision_requests"] = []
        st.session_state["last_facts_payload"] = None
        st.session_state["script_revision_request"] = ""
        st.session_state.pop("audio_bytes", None)
        st.session_state.pop("heygen_video_url", None)
        st.toast("Reverted to original script", icon="↩️")
        st.rerun()


def render_voiceover_section():
    st.subheader("4) Voiceover (ElevenLabs)")

    if not st.session_state.get("has_script"):
        st.caption("Generate a script first.")
        return

    if not st.session_state.get("eleven_api_key"):
        st.info("Add ELEVENLABS_API_KEY in the sidebar to enable audio generation.")
        return
    if not st.session_state.get("eleven_voice_id"):
        st.warning("Select a voice in the sidebar.")
        return

    c1, c2, _ = st.columns([1, 1, 2])
    make_audio = c1.button("Generate Audio", type="primary", use_container_width=True, key="btn_make_audio")
    regen_audio = c2.button("Regenerate", use_container_width=True, key="btn_regen_audio")
    do_tts = make_audio or regen_audio

    tts_text = st.session_state.get("generated_script_text") or st.session_state.get("last_script") or ""

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
        base = f"{ts}-{slugify(st.session_state.get('last_topic','') or 'script')}"
        mp3_name = f"{base}.mp3"
        st.download_button(
            "Download audio (.mp3)",
            st.session_state["audio_bytes"],
            file_name=mp3_name,
            mime="audio/mpeg",
            use_container_width=True,
        )
    else:
        st.caption("No audio yet. Generate to preview.")


def render_video_section():
    st.subheader("5) Render Video (HeyGen)")

    if not st.session_state.get("has_script"):
        st.caption("Generate a script first.")
        return

    if not st.session_state.get("heygen_api_key"):
        st.info("Add HEYGEN_API_KEY in the sidebar to enable video rendering.")
        return

    colv1, colv2, _ = st.columns([1, 1, 2])
    btn_render = colv1.button("Upload MP3 to HeyGen & Make Video", type="primary", use_container_width=True, key="btn_render_video")
    btn_rerender = colv2.button("Re-render (reuse inputs)", use_container_width=True, key="btn_rerender_video")

    if btn_render or btn_rerender:
        chosen_asset_id = (st.session_state.get("audio_asset_id_input") or "").strip()
        chosen_audio_url = (st.session_state.get("audio_url_input") or "").strip()

        if not chosen_asset_id and not chosen_audio_url:
            if not st.session_state.get("audio_bytes"):
                st.error("No ElevenLabs audio found. Generate audio first, or paste an existing HeyGen audio_asset_id / public audio_url.")
                return
            try:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                topic = st.session_state.get("last_topic", "") or "voiceover"
                filename = f"{slugify(topic)}-{ts}.mp3"
                with st.spinner("Uploading MP3 to HeyGen Assets…"):
                    chosen_asset_id = heygen_upload_audio_asset(
                        st.session_state["heygen_api_key"],
                        filename,
                        st.session_state["audio_bytes"],
                    )
                st.toast(f"Uploaded to HeyGen Assets (asset_id: {chosen_asset_id[:8]}…)", icon="⬆️")
            except Exception as e:
                st.error(f"HeyGen audio upload failed: {e}")
                return

        avatar_id = st.session_state.get("avatar_id_input")
        if not avatar_id:
            st.error("Please provide a HeyGen AVATAR ID (video avatar).")
            return

        try:
            heygen_get_avatar_details(st.session_state["heygen_api_key"], avatar_id)

            with st.spinner("Starting HeyGen video generation…"):
                video_id = heygen_generate_video(
                    st.session_state["heygen_api_key"],
                    avatar_id=avatar_id,
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

# -------------------------------
# 2) Generate a script
# -------------------------------
st.subheader("2) Generate a script")

topic_input = st.text_input("Topic", placeholder="e.g., Pakistan's fintech wave in 2025", key="topic_input")

# --- Paste existing script ---
with st.expander("Use your own script (paste)", expanded=False):
    pasted_script = st.text_area(
        "Paste script",
        key="pasted_script_text",
        height=220,
        placeholder="Paste your script here…",
    )
    pasted_title = st.text_input(
        "Optional title/topic",
        key="pasted_script_title",
        placeholder="e.g., Custom reel script",
    )
    colp1, _ = st.columns([1, 3])
    use_pasted = colp1.button(
        "Use pasted script",
        type="primary",
        use_container_width=True,
        key="btn_use_pasted_script",
    )

    if use_pasted:
        if not pasted_script.strip():
            st.warning("Paste a script first.")
        else:
            topic_for_paste = pasted_title.strip() or topic_input.strip() or "Custom script"
            style_samples: List[str] = []
            if st.session_state.get("meta") and st.session_state.get("openai_api_key"):
                try:
                    client = get_openai_client(st.session_state.get("openai_api_key"))
                    meta = st.session_state["meta"]
                    top = retrieve_top_k(topic_for_paste, meta, client, embed_model, k=k)
                    running = 0
                    for idx in top:
                        chunk = meta["chunks"][idx]
                        if running + len(chunk) <= max_ctx_chars:
                            style_samples.append(chunk)
                            running += len(chunk)
                    if not style_samples and top:
                        style_samples.append(meta["chunks"][top[0]])
                except Exception:
                    style_samples = []

            set_script_state(
                script=pasted_script.strip(),
                topic=topic_for_paste,
                facts_payload=None,
                style_samples=style_samples,
            )
            st.session_state.pop("audio_bytes", None)
            st.session_state.pop("heygen_video_url", None)
            st.rerun()

# --- News controls ---
news_col1, news_col2, news_col3 = st.columns([1.2, 1.2, 1.6])
news_mode = news_col1.selectbox("Find News for", ["Creators", "Founders"], index=0)
news_range = news_col2.selectbox(
    "Date range",
    ["One day", "One week", "One month", "3 months", "6 months", "One year"],
    index=2,
)
news_name = news_col3.text_input("Name of founder/creator (optional)", value="", placeholder="e.g., Mooroo, Irfan Junejo, Jehan Ara…")

colg = st.columns([1, 1, 2, 2])
clicked_generate = colg[0].button("Generate", type="primary", use_container_width=True, key="btn_generate_topic")
clicked_news = colg[1].button("Find News", use_container_width=True, key="btn_find_news")

# -------------------------------
# Topic generate flow
# -------------------------------
if clicked_generate:
    if not topic_input.strip():
        st.warning("Enter a topic first.")
    elif not st.session_state.get("meta"):
        st.warning("Corpus index not ready. Ensure `scripts.txt` is present and index built successfully.")
    else:
        client = get_openai_client(st.session_state.get("openai_api_key"))
        meta = st.session_state["meta"]
        topic = topic_input.strip()

        # Retrieval
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
                            decision = {
                                "claim": c,
                                "verdict": "error",
                                "correction": str(e),
                                "confidence": 0.0,
                                "citations": [],
                            }
                        decisions.append(decision)

                        cites = ", ".join(decision.get("citations", [])[:2]) or "—"
                        facts_table_rows.append(
                            {
                                "#": i,
                                "Claim": c,
                                "Verdict": decision.get("verdict", ""),
                                "Confidence": f"{decision.get('confidence', 0):.2f}",
                                "Citations": cites,
                            }
                        )
                        progress.progress(i / len(claims))
                    progress.empty()

                    try:
                        script, bibliography = apply_corrections_and_citations(
                            client, gen_model, script, decisions, add_inline_citations=citations
                        )
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

        set_script_state(script=script, topic=topic, facts_payload=facts_payload, style_samples=retrieved)
        st.rerun()

# -------------------------------
# News search flow
# -------------------------------
if clicked_news:
    if not st.session_state.get("openai_api_key"):
        st.error("Add your OPENAI_API_KEY in the sidebar to use Find News.")
    else:
        client = get_openai_client(st.session_state.get("openai_api_key"))
        with st.spinner("Searching the web for news…"):
            try:
                items, debug = find_news(
                    client,
                    mode=news_mode,
                    date_range_label=news_range,
                    name_filter=news_name,
                    max_items=10,
                )
            except Exception as e:
                st.error(f"Find News failed: {e}")
                items = []
                debug = {"error": str(e)}

        st.session_state["last_news_items"] = items
        st.session_state["last_news_mode"] = news_mode
        st.session_state["last_news_range"] = news_range
        st.session_state["last_news_name"] = news_name
        st.session_state["last_news_debug"] = debug
        st.rerun()

# -------------------------------
# Persisted news results + “generate from news” flow (no duplicate script editor)
# -------------------------------
if st.session_state.get("last_news_items") is not None and len(st.session_state.get("last_news_items") or []) >= 0:
    saved_items = st.session_state.get("last_news_items") or []
    if saved_items:
        with news_box:
            nm = st.session_state.get("last_news_mode", "Creators")
            nr = st.session_state.get("last_news_range", "One month")
            nn = st.session_state.get("last_news_name", "")
            st.subheader(f"📰 News results — {nm} ({nr})")
            if nn.strip():
                st.caption(f"Filter: {nn.strip()}")
            st.dataframe(saved_items, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("🎯 Turn a news item into a script")

            selected_idx = st.selectbox(
                "Choose a news item",
                options=list(range(len(saved_items))),
                format_func=lambda i: _news_item_label(saved_items[i], i),
                key="selected_news_idx_once",
            )

            gen_from_news = st.button(
                "Generate Script from Selected News",
                type="primary",
                use_container_width=True,
                key="btn_generate_from_news_once",
            )

            if gen_from_news:
                if not st.session_state.get("meta"):
                    st.error("Corpus index not ready. Ensure `scripts.txt` is present and indexed.")
                else:
                    client = get_openai_client(st.session_state.get("openai_api_key"))
                    meta = st.session_state["meta"]

                    chosen = saved_items[int(selected_idx)]
                    topic_from_news = _news_item_topic(chosen)
                    anchor = _news_item_anchor(chosen)

                    with st.spinner("Retrieving top chunks…"):
                        top = retrieve_top_k(topic_from_news, meta, client, embed_model, k=k)
                        retrieved: List[str] = []
                        running = 0
                        for idx in top:
                            chunk = meta["chunks"][idx]
                            if running + len(chunk) <= max_ctx_chars:
                                retrieved.append(chunk)
                                running += len(chunk)
                        if not retrieved and top:
                            retrieved.append(meta["chunks"][top[0]])

                    with st.spinner("Generating script from selected news…"):
                        messages = make_prompt(topic_from_news, retrieved, news_anchor=anchor)
                        try:
                            script = generate_script(client, gen_model, messages, temperature=temperature)
                            st.session_state.pop("audio_bytes", None)
                            st.session_state.pop("heygen_video_url", None)
                        except Exception as e:
                            st.error(f"Generation failed: {e}")
                            st.stop()

                    set_script_state(script=script, topic=topic_from_news, facts_payload=None, style_samples=retrieved)
                    st.rerun()

# -------------------------------
# Render script editor ONCE + then downstream sections
# -------------------------------
render_script_editor_once()
render_revision_controls()
render_voiceover_section()
render_video_section()
