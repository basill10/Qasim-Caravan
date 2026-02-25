"""
Streamlit UI for Instagram-style script generation with optional fact checking,
ElevenLabs voiceover generation (preview, regenerate, download), and HeyGen avatar video
rendering with Avatar IV Talking Photos or standard video avatars.

Run:
    streamlit run streamlit_reel_writer_app.py

Requirements:
    pip install streamlit openai requests numpy

Environment (can also be entered in the sidebar at runtime):
    - OPENAI_API_KEY
    - TAVILY_API_KEY   (optional; required only if Fact Checking is enabled)
    - ELEVENLABS_API_KEY (for voiceover)
    - ELEVENLABS_VOICE_ID (optional; default voice id for ElevenLabs)
    - HEYGEN_API_KEY     (for HeyGen assets + video)

Auth:
    - APP_USERNAME (in .streamlit/secrets.toml preferred)
    - APP_PASSWORD (in .streamlit/secrets.toml preferred)

Notes:
- For GPT-5 models ("gpt-5", "gpt-5-pro"), we use the Responses API and NEVER send `temperature`.
- HeyGen integration uses Avatar IV for Talking Photos via `/v2/video/generate` and supports video avatars too.
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
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import requests
import streamlit as st
from openai import OpenAI
try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:
    genai = None
    types = None
from datetime import datetime, timedelta

try:
    from google.oauth2 import service_account  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.http import MediaFileUpload  # type: ignore
    from googleapiclient.errors import HttpError  # type: ignore
except Exception:
    service_account = None
    build = None
    MediaFileUpload = None
    HttpError = None


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
THUMB_TEMPLATE_DIR = Path(os.getenv("THUMB_TEMPLATE_DIR", Path(__file__).parent))

TAVILY_API_KEY_ENV = os.getenv("TAVILY_API_KEY")
TAVILY_ENDPOINT = "https://api.tavily.com/search"

GOOGLE_CSE_API_KEY_ENV = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX_ENV = os.getenv("GOOGLE_CSE_CX")
GEMINI_API_KEY_ENV = os.getenv("GEMINI_API_KEY")

# --- ElevenLabs ---
ELEVEN_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY", "")
DEFAULT_ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "ZKtPxZZTlravOYReKIFJ")
ELEVEN_BASE = "https://api.elevenlabs.io/v1"

# --- Script cleanup (remove hyperlinks for voiceover) ---
LINK_CLEAN_MODEL = os.getenv("LINK_CLEAN_MODEL", "gpt-5-mini")

# --- HeyGen ---
HEYGEN_API_KEY_ENV = os.getenv("HEYGEN_API_KEY", "")
HEYGEN_BASE_V2 = "https://api.heygen.com/v2"
HEYGEN_STATUS_V1 = "https://api.heygen.com/v1/video_status.get"

DEFAULT_DRIVE_FOLDER_LINK = os.getenv(
    "DRIVE_FOLDER_LINK",
    "https://drive.google.com/drive/folders/0AOmUeJjNfdC_Uk9PVA",
)
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


# -------------------------------
# Utilities
# -------------------------------
def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def get_secret(name: str) -> Optional[str]:
    """
    Priority:
    1) OS environment variables
    2) Streamlit secrets (if configured)
    """
    val = os.getenv(name)
    if val:
        return val
    try:
        if name in st.secrets:
            return st.secrets[name]  # type: ignore[index]
    except Exception:
        return None
    return None

def get_secret_json_bytes(*names: str) -> Optional[bytes]:
    """
    Fetch a JSON object from Streamlit secrets and return it as UTF-8 bytes.

    Supports:
    - Secret as a JSON string
    - Secret as a TOML table (dict-like), which we json.dumps
    """
    for name in names:
        if not name:
            continue
        try:
            if name not in st.secrets:  # type: ignore[operator]
                continue
            v = st.secrets.get(name)  # type: ignore[attr-defined]
        except Exception:
            v = None

        if v is None:
            continue

        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            return s.encode("utf-8")
        if isinstance(v, dict):
            try:
                return json.dumps(v).encode("utf-8")
            except Exception:
                continue
    return None


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "output"

def extract_drive_folder_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if not s:
        return ""

    # Already looks like an ID
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
        return s

    # Common patterns
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", s)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", s)
    if m:
        return m.group(1)
    return ""

def _drive_deps_ready() -> bool:
    return bool(service_account and build and MediaFileUpload and HttpError)

def _sa_email_from_json_bytes(sa_json_bytes: Optional[bytes]) -> str:
    if not sa_json_bytes:
        return ""
    try:
        info = json.loads(sa_json_bytes.decode("utf-8"))
        return (info.get("client_email") or "").strip()
    except Exception:
        return ""

@st.cache_resource(show_spinner=False)
def get_drive_service_from_sa_json(sa_json_bytes: bytes):
    if not _drive_deps_ready():
        raise RuntimeError("Google Drive dependencies missing. Install google-api-python-client and google-auth.")
    if not sa_json_bytes:
        raise RuntimeError("Service account JSON is empty.")
    info = json.loads(sa_json_bytes.decode("utf-8"))
    creds = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    svc = build("drive", "v3", credentials=creds, cache_discovery=False)
    try:
        timeout_s = int(os.getenv("DRIVE_HTTP_TIMEOUT", "600"))
        if hasattr(svc, "_http") and getattr(svc._http, "timeout", None) is not None:  # type: ignore[attr-defined]
            svc._http.timeout = timeout_s  # type: ignore[attr-defined]
    except Exception:
        pass
    return svc

def download_url_to_file(url: str, out_path: Path, *, timeout: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def upload_file_to_drive(
    drive_service,
    *,
    local_path: Path,
    parent_folder_id: str,
    mime_type: str,
    drive_filename: Optional[str] = None,
) -> Dict[str, str]:
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")
    if not local_path.is_file():
        raise IsADirectoryError(f"Path is not a file: {local_path}")
    if not parent_folder_id:
        raise ValueError("Missing Drive folder id.")

    def _is_retryable_exc(e: BaseException) -> bool:
        if isinstance(e, (BrokenPipeError, ConnectionResetError, TimeoutError)):
            return True
        if isinstance(e, OSError) and getattr(e, "errno", None) in (32, 104, 110):  # broken pipe / conn reset / timeout
            return True
        if HttpError is not None and isinstance(e, HttpError):
            try:
                status = int(getattr(getattr(e, "resp", None), "status", 0))
            except Exception:
                status = 0
            return status in (408, 429) or 500 <= status < 600
        return False

    metadata: Dict[str, Any] = {"name": drive_filename or local_path.name, "parents": [parent_folder_id]}

    # Resumable, chunked upload with retries for flaky connections (e.g., Streamlit Cloud).
    retries = int(os.getenv("DRIVE_UPLOAD_RETRIES", "5"))
    chunk_size = int(os.getenv("DRIVE_UPLOAD_CHUNK_SIZE", str(8 * 1024 * 1024)))  # 8MB
    fields = "id,name,webViewLink,webContentLink"

    created: Optional[Dict[str, Any]] = None
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        progress = None
        try:
            media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True, chunksize=chunk_size)
            request = drive_service.files().create(
                body=metadata,
                media_body=media,
                fields=fields,
                supportsAllDrives=True,
            )

            progress = st.progress(0.0, text="Uploading to Google Drive…")
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status is not None:
                    try:
                        progress.progress(min(max(float(status.progress()), 0.0), 1.0))
                    except Exception:
                        pass
            try:
                progress.progress(1.0)
            except Exception:
                pass
            created = response
            last_exc = None
            break
        except BaseException as e:
            last_exc = e
            if attempt >= retries or not _is_retryable_exc(e):
                raise
            backoff = min(2 ** (attempt - 1), 16)
            st.warning(f"Drive upload interrupted ({type(e).__name__}). Retrying in {backoff}s… (attempt {attempt}/{retries})")
            time.sleep(backoff)
        finally:
            try:
                if progress is not None:
                    progress.empty()
            except Exception:
                pass

    if created is None:
        if last_exc:
            raise last_exc
        raise RuntimeError("Drive upload failed without an exception.")
    return {
        "id": created.get("id") or "",
        "name": created.get("name") or (drive_filename or local_path.name),
        "webViewLink": created.get("webViewLink") or "",
        "webContentLink": created.get("webContentLink") or "",
    }

def drive_check_folder_access(drive_service, folder_id: str) -> tuple[bool, str]:
    if not folder_id:
        return False, "Drive folder ID is empty."
    try:
        meta = (
            drive_service.files()
            .get(
                fileId=folder_id,
                fields="id,name,mimeType",
                supportsAllDrives=True,
            )
            .execute()
        )
    except Exception as e:
        if HttpError is not None and isinstance(e, HttpError):
            try:
                status = int(getattr(getattr(e, "resp", None), "status", 0))
            except Exception:
                status = 0
            if status == 404:
                return False, f"Drive folder not found / not shared with this service account: {folder_id}"
        return False, f"Drive folder check failed: {e}"

    if (meta.get("mimeType") or "") != "application/vnd.google-apps.folder":
        return False, f"Drive ID is not a folder: {folder_id}"
    name = (meta.get("name") or "").strip() or folder_id
    return True, f"Drive folder OK: {name}"

def _drive_preflight() -> Optional[str]:
    if not st.session_state.get("drive_folder_id"):
        return "Drive folder ID is missing (set it in the sidebar)."
    if not st.session_state.get("drive_sa_json_bytes"):
        return "Service account JSON is missing (set it in Streamlit secrets or upload it in the sidebar)."
    if not _drive_deps_ready():
        return "Drive dependencies are not installed (`google-api-python-client`, `google-auth`)."
    try:
        svc = get_drive_service_from_sa_json(st.session_state["drive_sa_json_bytes"])
        ok, msg = drive_check_folder_access(svc, st.session_state["drive_folder_id"])
        if not ok:
            email = _sa_email_from_json_bytes(st.session_state.get("drive_sa_json_bytes"))
            extra = f" (service account: {email})" if email else ""
            return (
                msg
                + extra
                + ". Share the folder with the service account email (Editor). If this is a Shared Drive, add the service account as a member."
            )
    except Exception as e:
        return f"Drive setup check failed: {e}"
    return None

def _optional_path(s: object) -> Optional[Path]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    p = Path(s)
    return p


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

def parse_uploaded_news_txt(text: str) -> List[Dict[str, str]]:
    """
    Parse uploaded TXT into news items.

    Expected format (repeated blocks):
    1. Name:
    Headline line
    Summary line (optional)
    """
    if not text:
        return []

    blocks: List[List[str]] = []
    current: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue

        m = re.match(r"^\s*\d+\.\s*(.*)$", line)
        if m:
            if current:
                blocks.append(current)
            current = [m.group(1).strip()]
        else:
            current.append(line)

    if current:
        blocks.append(current)

    items: List[Dict[str, str]] = []
    for b in blocks:
        if not b:
            continue
        name = (b[0] or "").rstrip(":").strip()
        headline = b[1].strip() if len(b) >= 2 else ""
        if not headline:
            continue
        if len(b) >= 3:
            summary = " ".join(x.strip() for x in b[2:] if x.strip())
        else:
            summary = headline

        items.append(
            {
                "Name": name,
                "Headline": headline,
                "Summary": summary,
                "Published": "",
                "URL": "",
                "Source": "Uploaded",
            }
        )

    return items

def merge_news_items(existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(it: Dict[str, Any]) -> tuple[str, str]:
        name = (it.get("Name") or "").strip().lower()
        headline = (it.get("Headline") or "").strip().lower()
        return (name, headline)

    seen = {_key(it) for it in (existing or [])}
    merged = list(existing or [])
    for it in new_items:
        key = _key(it)
        if key in seen:
            continue
        seen.add(key)
        merged.append(it)
    return merged


# -------------------------------
# Thumbnail helpers
# -------------------------------
def cse_image_search(
    api_key: str,
    cx: str,
    query: str,
    *,
    start: int = 1,
    num: int = 10,
    safe: str = "active",
) -> List[Dict[str, str]]:
    if not api_key or not cx:
        raise RuntimeError("Google CSE API key or CX missing.")
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "searchType": "image",
        "start": start,
        "num": min(max(1, num), 10),
        "safe": safe,
        "imgType": "photo",
    }
    r = requests.get(endpoint, params=params, timeout=20)
    r.raise_for_status()
    data = r.json() or {}
    out: List[Dict[str, str]] = []
    for it in (data.get("items") or []):
        img = it.get("image") or {}
        out.append(
            {
                "original": it.get("link") or "",
                "thumbnail": img.get("thumbnailLink") or it.get("link") or "",
                "source": img.get("contextLink") or it.get("link") or "",
                "title": it.get("title") or query,
            }
        )
    return out


def _openai_json_response(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
) -> Dict[str, Any]:
    if is_gpt5(model):
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            text={"format": {"type": "json_object"}, "verbosity": "medium"},
            reasoning={"effort": "medium", "summary": "auto"},
        )
        content = _extract_text_from_responses(resp)
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=600,
        )
        content = resp.choices[0].message.content

    data = _safe_json_load(content or "")
    return data if isinstance(data, dict) else {}


def generate_heading_from_transcript(client: OpenAI, transcript: str) -> Optional[Dict[str, str]]:
    if not transcript.strip():
        return None

    system_msg = (
        "You are a copywriter for a Pakistani business/tech brand. "
        "You write 2-part thumbnail headings for Instagram carousels and short videos. "
        "Your output must be JSON with keys 'subheading' and 'main_heading'. "
        "Tone: sharp, high-impact, concise, story-driven, serious but energetic."
    )

    examples_block = """
Here are three examples of the style, tone, and format:

Example 1:
Subheading:
The flavour of generations
Main Heading:
shezan became the taste of pakistan
Transcript:
It’s the mango bottle many Pakistanis grew up with— but Shezaan began as more than a drink.

The name was on Lahore’s dining scene in the late ’50s; by 1964, the Shahnawaaz Group launched Shezaan International as a JV with an American partner. In 1965, they set up a fruit-processing plant in Lahore to bottle orange, adding mango by 1967. 

Scale followed quickly: a Karachi unit in 1981 to serve Sindh and exports, an independent Tetra Brik line in 1987, and a juice factory in 1990. 

Parallel to the juices, Lahore got a hometown staple: Shezaan Bakers opened its first outlet in 1975 and today runs a network of bakery outlets across the city. Patties at tea-time, birthday cakes, wedding sweets— a ritual for generations. 

The brand spread abroad with pallets of ketchup, jams, and mango drinks— Shezaan products now ship to 40+ countries, carrying a taste of home to diaspora aisles. 

Today, Shezaan runs plants in Lahore, Karachi, and Hattar, pushing classic lines like Mango, All Pure, and sauces through national distribution. 

In FY25, it posted Rs 9.18 billion in sales and swung back to profit— proof the legacy label still moves with the market. 

From school canteens to iftar tables to motorway stops— Shezaan became a flavor of everyday Pakistan.

Example 2:
Subheading:
built on affordability and trust
Main Heading:
imtiaz changed retail forever
Transcript:
From a one-room kiryaana in Karachi to Pakistan’s most-watched supermarket story. This is Imtiaaz — and the retailer who scaled it, Imtiaaz Hussain.

In 1955, his father Hakeem Khan  opened a small corner shop in Karachi. The son kept the name, and rebuilt the model.

He bet on trust with suppliers, tight inventory, and everyday low prices. First came bigger aisles and self-service. Then the shift to multi-format retail.

Imtiaaz launched Super and Mega — from neighborhood convenience to hypermarket scale. Private labels followed. A loyalty program turned walk-ins into repeat families.

The 2010s took it beyond Karachi — into Punjab. Back-end operations and the supply chain scaled to feed dozens of stores, daily.

During lockdowns, it leaned on phone orders and e-commerce pilots to keep households stocked. The local shop mindset — but digitized.

By 2024, Imtiaaz was operating 31 stores across 13 cities. A team of 14,000 served over a million customers a day. Independent estimates placed annual revenue between PKR 60–70 billion.

What stayed constant was the floor discipline: fast turns, wide assortments, sharp pricing. What changed was scale.

From ledger books to ERP dashboards. From one cash counter to national brand power.

Imtiaz isn’t just a chain — it’s a playbook for Pakistani retail. Proof that a homegrown local shop can grow up without losing its edge on price, service, and trust.

Example 3:
Subheading:
Pakistani founders web3 bet
Main Heading:
Myco’s global streaming wave
Transcript:
He turned watch time into income. This is Myco — founded by Umair Usmani. 

Myco is a Web3 streaming platform featuring a watch-and-earn model that rewards viewers and uses a decentralized revenue-sharing framework that benefits creators. 

The numbers travel fast. 20 million users in 18 months, 100 million monthly video impressions, and users in 100+ countries. 
And Pakistan remains a key market — for growth and audience engagement. 

2025 was the breakthrough. Myco won the global finale of Meet the Drapers, securing a package reported at roughly $1.5M—including a $1M SaaS services contract—reported as the season’s top deal.

Sports powered the flywheel. In 2025, Myco engaged fans around PSL 10 with watch-and-earn rewards across its mobile, TV apps, and web experience. 
Beyond cricket, the slate spans MMA, squash, padel, and regional entertainment. 

The roadmap stays bold — expanding across MENA, North America, and South Asia, targeting 50M+ users by end-2025 through telco and media partnerships.

A Pakistani founder, a global play. Turning viewers into participants—and attention into an economy.
"""

    user_prompt = f"""{examples_block}

Now, given the following transcript, write a new subheading and main heading in the same tone and structure, both in ALL CAPS:

Transcript:
<<<
{transcript}
>>>

Rules:
- Output MUST be valid JSON.
- Keys: "subheading" and "main_heading".
- Subheading: short, 3–7 words.
- Main heading: bold, 5–10 words, all lowercase except brand names.
- Do NOT include any explanations or extra text, only JSON.
"""

    data = _openai_json_response(
        client,
        model="gpt-5-nano",
        system=system_msg,
        user=user_prompt,
    )
    sub = (data.get("subheading") or "").strip()
    main = (data.get("main_heading") or "").strip()
    if not sub or not main:
        return None
    return {"subheading": sub, "main_heading": main}


def analyze_visual_concept(client: OpenAI, transcript: str) -> Optional[Dict[str, str]]:
    if not transcript.strip():
        return None

    system_msg = (
        "You analyze transcripts of short videos and decide what is the best visual for a YouTube/Instagram thumbnail. "
        "You respond ONLY with JSON."
    )
    user_prompt = f"""
Transcript:
<<<
{transcript}
>>>

Decide the thumbnail concept.

Rules:
1. If the transcript clearly centers around a specific PERSON or a specific PLACE/INSTITUTION:
   - Set "mode" to "entity".
   - Set "entity_name" to exactly that person or place name.
   - Set "thumbnail_prompt" to an empty string.

2. Otherwise:
   - Set "mode" to "prompt".
   - Set "entity_name" to an empty string.
   - Set "thumbnail_prompt" to a bold, cinematic, visually clear description suitable for a thumbnail (no text overlays, just imagery).

3. Output MUST be valid JSON with keys: "mode", "entity_name", "thumbnail_prompt".
4. Do not add any explanations. JSON only.
"""

    data = _openai_json_response(
        client,
        model="gpt-5-nano",
        system=system_msg,
        user=user_prompt,
    )
    return {
        "mode": (data.get("mode") or "").strip(),
        "entity_name": (data.get("entity_name") or "").strip(),
        "thumbnail_prompt": (data.get("thumbnail_prompt") or "").strip(),
    }


def generate_ai_image(client: OpenAI, prompt: str) -> Optional[bytes]:
    if not prompt.strip():
        return None
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )
        image_base64 = result.data[0].b64_json
        return base64.b64decode(image_base64)
    except Exception as e:
        st.error(f"Error generating AI image: {e}")
        return None


def compose_thumbnail_with_gemini(
    gemini_client,
    base_color: str,
    subheading: str,
    main_heading: str,
    main_image_bytes: bytes,
) -> Optional[bytes]:
    if not gemini_client or not types:
        st.error("Gemini client not configured.")
        return None

    color_to_filename = {
        "Blue": "blue.png",
        "Black": "black.png",
        "White": "white.png",
    }
    template_name = color_to_filename.get(base_color)
    template_path = (THUMB_TEMPLATE_DIR / template_name) if template_name else None
    if not template_path or not template_path.exists():
        st.error(f"Base template not found for color '{base_color}'. Expected file: {template_name}")
        return None

    with open(template_path, "rb") as f:
        base_bytes = f.read()

    instruction = f"""
You are a thumbnail designer.

You are given:
1) A base thumbnail template image (first image), in 1080x1920 resolution (width 1080 px, height 1920 px).
2) A main photo image (second image).

Tasks:

- KEEP THE CANVAS THE SAME:
  - The final output MUST have exactly the same resolution as the base template: 1080 px wide and 1920 px tall.
  - Do NOT crop, resize, or change the canvas size of the base template. Use it as the background.

- TEXT:
  - Replace the existing heading text entirely on the base template with these two lines:
      Subheading: "{subheading}"
      Main heading: "{main_heading}"
  - Both should be in ALL CAPS
  - Keep font size, weight, and color similar to the original heading style.
  - Keep the shadows and glows around the text 
  - Place the subheading as a smaller line above, and the main heading as a large, bold line below.

- MAIN IMAGE (FOREGROUND SUBJECT):
  - From the second image, REMOVE the background where possible and isolate the main subject/person/object.
  - Place the cut-out subject onto the template in the main image/photo area.
  - Avoid visible rectangular borders or white boxes around the subject.
  - Preserve a clean, professional compositing with soft or natural edges.

- OVERALL:
  - Keep the base template layout, colors, logos, and framing intact.
  - Ensure the final image looks like a polished social media thumbnail.
Return a single finished thumbnail image.
""".strip()

    try:
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=instruction),
                    types.Part.from_bytes(
                        data=base_bytes,
                        mime_type="image/png",
                    ),
                    types.Part.from_bytes(
                        data=main_image_bytes,
                        mime_type="image/png",
                    ),
                ],
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )

        response = gemini_client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=generate_content_config,
        )

        if getattr(response, "parts", None):
            for part in response.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    return part.inline_data.data

        if (
            getattr(response, "candidates", None)
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    return part.inline_data.data

        st.error("No inline image data found in Gemini response.")
        return None
    except Exception as e:
        st.error(f"Error composing thumbnail with Gemini: {e}")
        return None

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

@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str | None):
    if not api_key or not genai:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None

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

NEWS_RESEARCH_SYS = (
    "You are a meticulous research assistant. Use web_search to verify and expand on the selected news item.\n"
    "Rules:\n"
    "- Do NOT hallucinate or invent facts.\n"
    "- If a detail is unclear, mark it as uncertain.\n"
    "- Keep output concise and structured with headings.\n"
    "- Return ONLY the research dump text, no markdown.\n"
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


def research_news_dump_gpt52(client: OpenAI, *, topic: str, anchor: str) -> str:
    user_prompt = f"""
Selected news item:
{anchor}

Task:
- Use web_search to verify the key facts about this item and gather any relevant context.
- Focus on names, dates, locations, organizations, and what happened.
- Keep the output concise and structured.

Output format (plain text, no markdown):
Key Facts:
- ...

Timeline:
- ...

Context:
...

Names/Orgs:
- ...

Sources:
- URL
""".strip()

    resp = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": NEWS_RESEARCH_SYS},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{
            "type": "web_search",
            "user_location": {"type": "approximate"},
            "search_context_size": "medium",
        }],
        text={"format": {"type": "text"}, "verbosity": "medium"},
        max_output_tokens=1800,
        include=["web_search_call.action.sources"],
        store=False,
    )

    research = _extract_text_from_responses(resp)
    return research.strip() if research else ""


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


def make_prompt(
    topic: str,
    retrieved_chunks: List[str],
    news_anchor: str | None = None,
    research_dump: str | None = None,
) -> List[Dict[str, str]]:
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

    research_block = ""
    if research_dump:
        research_block = (
            "\n\n--- RESEARCH DUMP (verified context) ---\n"
            f"{research_dump}\n"
            "\nRules for the research dump:\n"
            "- Use it to ground facts and details.\n"
            "- Do not mention 'research dump' in the script.\n"
        )

    user_msg = f"""Topic: {topic}

Using the style learned from the samples below (structure, cadence, sentence length, and tone), write a new script tailored to the topic. Avoid copying lines verbatim. Prefer locally relevant angles (Pakistan when applicable), concrete numbers, and short paragraph beats.
{anchor_block}{research_block}
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




def generate_script(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    *,
    use_web_search: bool = True,
) -> str:
    if is_gpt5(model):
        payload: Dict[str, Any] = {
            "model": model,
            "input": messages,
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "reasoning": {"effort": "medium", "summary": "auto"},
            "store": True,
        }
        if use_web_search:
            payload["tools"] = [{
                "type": "web_search",
                "user_location": {"type": "approximate"},
                "search_context_size": "medium",
            }]
            payload["include"] = ["reasoning.encrypted_content", "web_search_call.action.sources"]
        else:
            payload["include"] = ["reasoning.encrypted_content"]

        resp = client.responses.create(**payload)
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


def urdu_scriptify_text(client: OpenAI, text: str) -> str:
    """
    Convert Urdu words/names written in Latin characters into Urdu script for TTS.
    Keeps English text, punctuation, and line breaks unchanged.
    """
    if not text.strip():
        return text

    sys_msg = (
        "You convert Urdu words and names written in Latin letters into Urdu script. "
        "Keep English words in Latin script. Preserve punctuation, spacing, numbers, and line breaks. "
        "Do not translate English content. Output ONLY the converted text."
    )
    user_msg = f"TEXT:\n{text}"

    resp = client.responses.create(
        model="gpt-5.2",
        input=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        text={"format": {"type": "text"}, "verbosity": "low"},
        reasoning={"effort": "low", "summary": "auto"},
        store=False,
    )
    out = _extract_text_from_responses(resp)
    return out.strip() if out else text


def _regex_strip_links(text: str) -> str:
    if not text:
        return text

    # Markdown links: [text](url) -> text
    out = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", text)

    # Naked URLs
    out = re.sub(r"https?://\S+", "", out)
    out = re.sub(r"\bwww\.\S+", "", out)

    # Common citation markers that become noisy in TTS
    out = re.sub(r"\[(\d{1,3})\]", "", out)
    out = re.sub(r"\(\s*source\s*:\s*[^)]+\)", "", out, flags=re.IGNORECASE)

    # Drop trailing "References/Sources" sections if present
    out = re.split(r"\n\s*(references|sources)\s*\n", out, maxsplit=1, flags=re.IGNORECASE)[0]

    # Tidy whitespace left by removals
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def strip_source_links_gpt5_mini(client: OpenAI, script: str) -> str:
    """
    Remove URLs/hyperlinks/citation markers from a script while keeping wording the same.
    Falls back to regex cleanup if the model call fails.
    """
    if not script.strip():
        return script

    sys_msg = (
        "You are a careful script editor for voiceover. "
        "Task: remove ALL URLs and hyperlinks from the script, including markdown links like [text](url), "
        "naked URLs, and any trailing References/Sources section. Also remove citation markers like [1], [2], etc. "
        "Do NOT rewrite the script. Keep wording, punctuation, paragraphing, and line breaks the same as much as possible. "
        "Output ONLY the cleaned script text."
    )
    user_msg = f"SCRIPT:\n{script}"

    for model in [LINK_CLEAN_MODEL, "gpt-5"]:
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                text={"format": {"type": "text"}, "verbosity": "low"},
                reasoning={"effort": "low", "summary": "auto"},
                store=False,
            )
            out = _extract_text_from_responses(resp)
            if out and out.strip():
                return out.strip()
        except Exception:
            continue

    return _regex_strip_links(script)





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
    character_type: str,
    avatar_id: Optional[str] = None,
    talking_photo_id: Optional[str] = None,
    use_avatar_iv_model: bool = False,
    audio_asset_id: Optional[str] = None,
    audio_url: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    background_color: Optional[str] = None,
) -> str:
    """
    Generate a HeyGen video. Supports video avatars and Avatar IV (talking photo).
    """
    if not (audio_asset_id or audio_url):
        raise ValueError("Provide either audio_asset_id or audio_url for HeyGen voice.")

    if character_type == "talking_photo":
        if not talking_photo_id:
            raise ValueError("Provide talking_photo_id for Avatar IV (talking photo).")
        character = {"type": "talking_photo", "talking_photo_id": talking_photo_id}
    else:
        if not avatar_id:
            raise ValueError("Provide avatar_id (video avatar).")
        character = {"type": "avatar", "avatar_id": avatar_id, "avatar_style": "normal"}

    voice: Dict[str, Any] = {"type": "audio"}
    if audio_asset_id:
        voice["audio_asset_id"] = audio_asset_id
    if audio_url:
        voice["audio_url"] = audio_url

    video_input: Dict[str, Any] = {"character": character, "voice": voice}
    if background_color:
        video_input["background"] = {"type": "color", "value": background_color}

    payload: Dict[str, Any] = {"video_inputs": [video_input], "dimension": {"width": width, "height": height}}
    if use_avatar_iv_model:
        payload["use_avatar_iv_model"] = True

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
st.caption("Generate scripts, voice them with ElevenLabs, then upload MP3 to HeyGen Assets and render the avatar video (Avatar IV Talking Photos supported).")

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

    voice_id_fallback = (st.text_input("ELEVENLABS_VOICE_ID", value=DEFAULT_ELEVEN_VOICE_ID, key="eleven_voice_id_input") or "").strip()

    stability = 0.5
    similarity_boost = 0.75
    style = 0.0
    use_speaker_boost = True

    voices = eleven_list_voices(eleven_api_key) if eleven_api_key else []
    voice_names = [f"{v['name']}  ·  {v.get('category') or ''}".strip() for v in voices] if voices else []
    default_voice_idx = (
        next((i for i, v in enumerate(voices) if v.get("id") == voice_id_fallback), 0) if voices else None
    )
    voice_idx = st.selectbox(
        "Voice",
        list(range(len(voice_names))) if voice_names else [],
        format_func=lambda i: voice_names[i],
        index=default_voice_idx if voice_names else None,
        placeholder="Login key first to load voices…",
        disabled=not voices,
    )
    st.caption("Tip: Use multilingual v2 for Urdu/English mixed content.")
    with st.popover("Advanced voice settings"):
        stability = st.slider("Stability", 0.0, 1.0, stability, 0.05)
        similarity_boost = st.slider("Similarity Boost", 0.0, 1.0, similarity_boost, 0.05)
        style = st.slider("Style", 0.0, 1.0, style, 0.05)
        use_speaker_boost = st.toggle("Use Speaker Boost", value=use_speaker_boost)

    if voices and voice_idx is not None:
        st.session_state["eleven_voice_id"] = voices[voice_idx]["id"]
    else:
        st.session_state["eleven_voice_id"] = voice_id_fallback or None
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

    avatar_type = st.selectbox(
        "Avatar type",
        ["Talking Photo (Avatar IV)", "Video Avatar"],
        index=0,
    )
    st.session_state["heygen_avatar_type"] = "talking_photo" if avatar_type.startswith("Talking Photo") else "avatar"

    if st.session_state["heygen_avatar_type"] == "talking_photo":
        st.caption("Avatar IV engine is enabled for Talking Photos.")
        talking_photo_id_input = st.text_input(
            "Talking Photo ID (talking_photo_id)",
            value="d53434113ef240969f03d1fd376176e0",
            placeholder="e.g. 920e0e2ea96e4fcf8d3cc9a7457840bf",
        )
        st.session_state["talking_photo_id_input"] = talking_photo_id_input.strip()
        st.session_state["avatar_id_input"] = ""
    else:
        st.caption("Uses standard video avatar engine (avatar_id).")
        avatar_id_input = st.text_input(
            "Video Avatar ID (avatar_id)",
            value="",
            placeholder="e.g. 920e0e2ea96e4fcf8d3cc9a7457840bf",
        )
        st.session_state["avatar_id_input"] = avatar_id_input.strip()
        st.session_state["talking_photo_id_input"] = ""

    st.caption("Output defaults to vertical 9:16.")
    heygen_width = st.number_input("Video Width", min_value=320, max_value=1920, value=1080, step=10)
    heygen_height = st.number_input("Video Height", min_value=240, max_value=1920, value=1920, step=10)
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
            "use_avatar_iv_model": st.session_state.get("heygen_avatar_type") == "talking_photo",
        }
    )

    st.divider()
    # --- Thumbnails ---
    st.subheader("Thumbnails")
    gemini_api_key = st.text_input("GEMINI_API_KEY (for thumbnail composer)", value=GEMINI_API_KEY_ENV or "", type="password")
    st.session_state["gemini_api_key"] = gemini_api_key
    google_cse_api_key = st.text_input("GOOGLE_CSE_API_KEY (image search)", value=GOOGLE_CSE_API_KEY_ENV or "", type="password")
    google_cse_cx = st.text_input("GOOGLE_CSE_CX (search engine ID)", value=GOOGLE_CSE_CX_ENV or "")
    st.session_state["google_cse_api_key"] = google_cse_api_key
    st.session_state["google_cse_cx"] = google_cse_cx

    st.divider()
    # --- Google Drive upload ---
    st.subheader("Google Drive Upload")
    st.toggle("Auto-upload MP4 + thumbnail", value=bool(os.getenv("DRIVE_UPLOAD_ENABLED")), key="drive_enabled")

    drive_folder_input = st.text_input(
        "Drive folder link or ID",
        value=os.getenv("DRIVE_FOLDER_ID") or DEFAULT_DRIVE_FOLDER_LINK,
        key="drive_folder_input",
        help="Paste the Drive folder URL or just the folder ID.",
    )
    drive_folder_id = extract_drive_folder_id(drive_folder_input)
    st.session_state["drive_folder_id"] = drive_folder_id
    if drive_folder_id:
        st.caption(f"Folder ID: {drive_folder_id}")
    else:
        st.warning("Enter a valid Drive folder link/ID to enable uploads.")

    drive_sa_path = st.text_input(
        "Service account JSON path (optional)",
        value=os.getenv("DRIVE_SA_JSON_PATH", ""),
        key="drive_sa_json_path",
        help="If running locally, you can point to an on-disk key file instead of uploading it each run.",
    )
    drive_sa_upload = st.file_uploader("Or upload service account JSON key", type=["json"], key="drive_sa_json_uploader")
    st.caption("Streamlit Cloud: set `DRIVE_SA_JSON` in secrets (JSON string), or `drive_service_account` as a TOML table.")

    sa_json_bytes: Optional[bytes] = None
    if drive_sa_upload is not None:
        sa_json_bytes = drive_sa_upload.getvalue()
    elif drive_sa_path.strip():
        try:
            p = Path(drive_sa_path.strip()).expanduser()
            if p.exists() and p.is_file():
                sa_json_bytes = p.read_bytes()
        except Exception:
            sa_json_bytes = None
    else:
        # Streamlit Cloud-friendly: store the service account JSON in secrets.
        # Recommended secret names:
        # - DRIVE_SA_JSON (JSON string) or DRIVE_SERVICE_ACCOUNT_JSON
        # - drive_service_account (TOML table)
        sa_json_bytes = get_secret_json_bytes(
            "DRIVE_SA_JSON",
            "DRIVE_SERVICE_ACCOUNT_JSON",
            "drive_service_account",
            "gcp_service_account",
        )

    st.session_state["drive_sa_json_bytes"] = sa_json_bytes
    if sa_json_bytes:
        try:
            info = json.loads(sa_json_bytes.decode("utf-8"))
            email = (info.get("client_email") or "").strip()
            if email:
                st.caption(f"Service account: {email}")
        except Exception:
            st.warning("Service account JSON couldn't be parsed.")

    if st.session_state.get("drive_enabled"):
        if not _drive_deps_ready():
            st.warning("Drive deps missing. Add `google-api-python-client` and `google-auth` to requirements.")
        st.caption("Make sure the Drive folder is shared with the service account email (Editor).")
        if st.button("Test Drive access", use_container_width=True, key="btn_drive_test_access"):
            err = _drive_preflight()
            if err:
                st.error(err)
            else:
                try:
                    svc = get_drive_service_from_sa_json(st.session_state["drive_sa_json_bytes"])
                    ok, msg = drive_check_folder_access(svc, st.session_state["drive_folder_id"])
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Drive access test failed: {e}")

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

def set_script_state(
    *,
    script: str,
    topic: str,
    facts_payload: dict | None,
    style_samples: List[str] | None = None,
    source: str | None = None,
    script_with_links: str | None = None,
    script_clean: str | None = None,
):
    script_with_links_final = script_with_links if script_with_links is not None else script
    script_clean_final = script_clean if script_clean is not None else script
    st.session_state.update(
        {
            "has_script": True,
            "last_script": script_clean_final,
            "last_script_with_links": script_with_links_final,
            "last_topic": topic,
            "last_facts_payload": facts_payload,
            "original_script": script_clean_final,
            "original_script_clean": script_clean_final,
            "original_script_with_links": script_with_links_final,
            "script_revision_history": [script_clean_final],
            "script_revision_requests": [],
            "last_style_samples": (style_samples or [])[:6],
            "generated_script_text": script_clean_final,
            "original_script_view": script_with_links_final,
            "generated_script_with_links_view": script_with_links_final,
            "script_revision_request": "",
            "script_source": source or "generated",
            "tts_ready_text": "",
            "tts_source_hash": "",
            "last_tts_text": "",
        }
    )

def download_buttons_area(text_clean: str, text_with_links: str, topic: str, facts_payload: dict | None):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{ts}-{slugify(topic)}"
    txt_name_clean = f"{base}.txt"
    txt_name_with_links = f"{base}-with-links.txt"
    st.download_button("Download script (.txt)", text_clean, file_name=txt_name_clean, mime="text/plain")
    if (text_with_links or "").strip() and (text_with_links.strip() != (text_clean or "").strip()):
        st.download_button(
            "Download script with links (.txt)",
            text_with_links,
            file_name=txt_name_with_links,
            mime="text/plain",
        )

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

    if "pending_script_text" in st.session_state:
        st.session_state["generated_script_text"] = st.session_state.pop("pending_script_text")

    script = st.session_state.get("last_script", "") or ""
    script_with_links = st.session_state.get("last_script_with_links", "") or script
    topic = st.session_state.get("last_topic", "") or ""
    facts_payload = st.session_state.get("last_facts_payload")

    script_box.text_area(
        "Generated script (original, with sources/links)",
        value=script_with_links,
        height=240,
        key="generated_script_with_links_view",
        disabled=True,
    )

    current_script_text = script_box.text_area(
        "Generated script (cleaned for voiceover)",
        value=script,
        height=320,
        key="generated_script_text",
    )
    st.caption("Tip: Edit the cleaned script above before generating audio or downloading.")
    download_buttons_area(current_script_text, script_with_links, topic, facts_payload)


def render_revision_controls():
    st.subheader("3) Refine the script")

    if not st.session_state.get("has_script"):
        st.caption("Generate a script first.")
        return

    if not st.session_state.get("openai_api_key"):
        st.info("Add your OPENAI_API_KEY in the sidebar to enable GPT-5 revisions.")
        return

    if "pending_revision_request" in st.session_state:
        st.session_state["script_revision_request"] = st.session_state.pop("pending_revision_request")

    current_script = st.session_state.get("generated_script_text") or st.session_state.get("last_script") or ""
    original_script_clean = (
        st.session_state.get("original_script_clean") or st.session_state.get("original_script") or current_script
    )
    original_script_with_links = st.session_state.get("original_script_with_links") or original_script_clean

    with st.expander("Original script (saved, with links)", expanded=False):
        st.text_area(
            "Original script",
            value=original_script_with_links,
            height=240,
            key="original_script_view",
            disabled=True,
        )

    if st.session_state.get("script_source") == "pasted":
        st.caption("Working from your pasted script. GPT-5 keeps tone/cadence from the original and corpus.")
    else:
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
                original_script=original_script_clean,
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

        history = st.session_state.get("script_revision_history") or [original_script_clean]
        history.append(revised)
        st.session_state["script_revision_history"] = history
        st.session_state["script_revision_requests"] = recent_requests + [revision_request.strip()]
        st.session_state["last_script"] = revised
        st.session_state["pending_script_text"] = revised
        st.session_state["last_facts_payload"] = None
        st.session_state["pending_revision_request"] = ""
        st.session_state.pop("audio_bytes", None)
        st.session_state.pop("heygen_video_url", None)
        st.session_state.pop("tts_ready_text", None)
        st.session_state.pop("tts_source_hash", None)
        st.session_state.pop("last_tts_text", None)
        st.toast("Script updated", icon="✏️")
        st.rerun()

    if reset_original:
        if not original_script_clean:
            st.warning("No original script saved yet.")
            return

        st.session_state["last_script"] = original_script_clean
        st.session_state["pending_script_text"] = original_script_clean
        st.session_state["script_revision_history"] = [original_script_clean]
        st.session_state["script_revision_requests"] = []
        st.session_state["last_facts_payload"] = None
        st.session_state["pending_revision_request"] = ""
        st.session_state.pop("audio_bytes", None)
        st.session_state.pop("heygen_video_url", None)
        st.session_state.pop("tts_ready_text", None)
        st.session_state.pop("tts_source_hash", None)
        st.session_state.pop("last_tts_text", None)
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

    if "pending_tts_text" in st.session_state:
        st.session_state["tts_ready_text"] = st.session_state.pop("pending_tts_text")
    if "pending_tts_source_hash" in st.session_state:
        st.session_state["tts_source_hash"] = st.session_state.pop("pending_tts_source_hash")

    current_script = st.session_state.get("generated_script_text") or st.session_state.get("last_script") or ""
    current_hash = sha256_bytes(current_script.encode("utf-8")) if current_script else ""

    tts_mode = st.radio(
        "TTS transcript source",
        options=["As-is (use script)", "Urdu script (optional conversion)"],
        horizontal=True,
        key="tts_mode",
    )
    wants_urdu = tts_mode.startswith("Urdu")

    if wants_urdu and not st.session_state.get("openai_api_key"):
        st.info("Add OPENAI_API_KEY in the sidebar to enable Urdu script conversion (optional).")

    c1, c2, c3 = st.columns([1, 1, 2])
    if wants_urdu:
        convert_btn = c1.button(
            "Convert to Urdu Script",
            type="primary",
            use_container_width=True,
            key="btn_convert_urdu",
            disabled=not bool(st.session_state.get("openai_api_key")),
        )
        reconvert_btn = c2.button(
            "Reconvert",
            use_container_width=True,
            key="btn_reconvert_urdu",
            disabled=not bool(st.session_state.get("openai_api_key")),
        )
        do_convert = convert_btn or reconvert_btn
    else:
        do_convert = False
        reset_btn = c1.button("Use current script", use_container_width=True, key="btn_tts_use_script")
        if reset_btn or st.session_state.get("tts_source_kind") != "script" or not (st.session_state.get("tts_ready_text") or "").strip():
            st.session_state["tts_ready_text"] = current_script
            st.session_state["tts_source_hash"] = current_hash
            st.session_state["tts_source_kind"] = "script"

    if do_convert:
        if not current_script.strip():
            st.warning("No script to convert yet.")
        elif not st.session_state.get("openai_api_key"):
            st.warning("Add OPENAI_API_KEY in the sidebar to convert to Urdu script.")
        else:
            try:
                client = get_openai_client(st.session_state.get("openai_api_key"))
                cache = st.session_state.setdefault("tts_urdu_cache", {})
                key = sha256_bytes(current_script.encode("utf-8"))
                if key in cache:
                    converted = cache[key]
                else:
                    with st.spinner("Converting Urdu words to Urdu script…"):
                        converted = urdu_scriptify_text(client, current_script)
                    cache[key] = converted
                    st.session_state["tts_urdu_cache"] = cache

                st.session_state["pending_tts_text"] = converted
                st.session_state["pending_tts_source_hash"] = key
                st.session_state["tts_source_kind"] = "urdu"
                st.session_state["last_tts_text"] = converted
                st.toast("Urdu transcript ready", icon="📝")
                st.rerun()
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    label = "TTS transcript (editable, used for TTS)"
    if wants_urdu:
        label = "Urdu transcript (editable, used for TTS)"
    st.text_area(label, key="tts_ready_text", height=240)
    if wants_urdu and not (st.session_state.get("tts_ready_text") or "").strip():
        st.caption("Click “Convert to Urdu Script” to generate the TTS transcript (or switch to As-is).")

    tts_ready_text = st.session_state.get("tts_ready_text") or ""
    tts_source_hash = st.session_state.get("tts_source_hash") or ""
    tts_source_kind = st.session_state.get("tts_source_kind") or ""
    stale_conversion = bool(tts_ready_text and tts_source_hash and tts_source_hash != current_hash)
    if wants_urdu and tts_source_kind == "urdu" and stale_conversion:
        st.warning("The script has changed since conversion. Please reconvert before generating audio.")
    elif (not wants_urdu) and tts_source_kind == "script" and stale_conversion:
        st.caption("The script has changed since you last loaded it here. Click “Use current script” to refresh.")

    c4, c5, _ = st.columns([1, 1, 2])
    make_audio = c4.button("Generate Audio", type="primary", use_container_width=True, key="btn_make_audio")
    regen_audio = c5.button("Regenerate", use_container_width=True, key="btn_regen_audio")
    do_tts = make_audio or regen_audio

    if do_tts:
        if not tts_ready_text.strip():
            if wants_urdu:
                st.error("Please convert to Urdu script first (or switch to As-is).")
            else:
                st.error("TTS transcript is empty.")
        elif wants_urdu and tts_source_kind == "urdu" and stale_conversion:
            st.error("Please reconvert after script changes.")
        else:
            try:
                with st.spinner("Generating voiceover…"):
                    audio_bytes = eleven_tts(
                        st.session_state["eleven_api_key"],
                        st.session_state["eleven_voice_id"],
                        tts_ready_text,
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

        avatar_type = st.session_state.get("heygen_avatar_type", "talking_photo")
        avatar_id = st.session_state.get("avatar_id_input")
        talking_photo_id = st.session_state.get("talking_photo_id_input")

        if avatar_type == "talking_photo":
            if not talking_photo_id:
                st.error("Please provide a Talking Photo ID (talking_photo_id) for Avatar IV.")
                return
        else:
            if not avatar_id:
                st.error("Please provide a Video Avatar ID (avatar_id).")
                return

        try:
            if avatar_type == "avatar" and avatar_id:
                heygen_get_avatar_details(st.session_state["heygen_api_key"], avatar_id)

            with st.spinner("Starting HeyGen video generation…"):
                width = int(st.session_state["heygen_width"])
                height = int(st.session_state["heygen_height"])
                if width >= height:
                    width, height = min(width, height), max(width, height)
                video_id = heygen_generate_video(
                    st.session_state["heygen_api_key"],
                    character_type=avatar_type,
                    avatar_id=avatar_id,
                    talking_photo_id=talking_photo_id,
                    use_avatar_iv_model=bool(st.session_state.get("use_avatar_iv_model")),
                    audio_asset_id=chosen_asset_id or None,
                    audio_url=chosen_audio_url or None,
                    width=width,
                    height=height,
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
            st.session_state["heygen_video_id"] = video_id
            st.toast("HeyGen video ready", icon="🎬")

            # Optional: download MP4 and upload to Google Drive
            if st.session_state.get("drive_enabled"):
                already = st.session_state.get("drive_uploaded_video_id")
                if already != video_id:
                    err = _drive_preflight()
                    if err:
                        st.warning(err)
                    else:
                        folder_id = (st.session_state.get("drive_folder_id") or "").strip()
                        sa_json_bytes = st.session_state.get("drive_sa_json_bytes")
                        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                        topic = st.session_state.get("last_topic", "") or "reel"
                        base = f"{ts}-{slugify(topic)}"
                        mp4_path = DEFAULT_OUTPUT_DIR / f"{base}-video.mp4"
                        try:
                            with st.spinner("Downloading MP4…"):
                                download_url_to_file(video_url, mp4_path, timeout=600)
                            st.session_state["last_video_path"] = str(mp4_path)
                            with st.spinner("Uploading MP4 to Google Drive…"):
                                svc = get_drive_service_from_sa_json(sa_json_bytes)
                                info = upload_file_to_drive(
                                    svc,
                                    local_path=mp4_path,
                                    parent_folder_id=folder_id,
                                    mime_type="video/mp4",
                                    drive_filename=mp4_path.name,
                                )
                            st.session_state["drive_uploaded_video_id"] = video_id
                            st.session_state["drive_uploaded_video_info"] = info
                            st.toast("Uploaded video to Drive", icon="✅")
                        except Exception as e:
                            st.error(f"Drive upload (video) failed: {e}")

        except Exception as e:
            st.error(f"HeyGen rendering failed: {e}")

    if st.session_state.get("heygen_video_url"):
        st.video(st.session_state["heygen_video_url"])
        st.link_button("Open video in new tab", st.session_state["heygen_video_url"], use_container_width=True)

        colu1, colu2, _ = st.columns([1, 1, 2])
        if colu1.button("Upload video to Google Drive", use_container_width=True, key="btn_drive_upload_video"):
            err = _drive_preflight()
            if err:
                st.error(err)
            else:
                video_url = st.session_state.get("heygen_video_url") or ""
                if not video_url:
                    st.error("No video_url available to upload.")
                else:
                    video_id = (st.session_state.get("heygen_video_id") or "").strip()
                    topic = st.session_state.get("last_topic", "") or "reel"
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    suffix = f"-{video_id[:8]}" if video_id else ""
                    base = f"{ts}-{slugify(topic)}{suffix}"
                    mp4_path = _optional_path(st.session_state.get("last_video_path"))
                    if not (mp4_path and mp4_path.is_file()):
                        mp4_path = DEFAULT_OUTPUT_DIR / f"{base}-video.mp4"
                        try:
                            with st.spinner("Downloading MP4…"):
                                download_url_to_file(video_url, mp4_path, timeout=600)
                            st.session_state["last_video_path"] = str(mp4_path)
                        except Exception as e:
                            st.error(f"Failed to download MP4: {e}")
                            mp4_path = None
                    if mp4_path and mp4_path.is_file():
                        try:
                            with st.spinner("Uploading MP4 to Google Drive…"):
                                svc = get_drive_service_from_sa_json(st.session_state["drive_sa_json_bytes"])
                                info = upload_file_to_drive(
                                    svc,
                                    local_path=mp4_path,
                                    parent_folder_id=st.session_state["drive_folder_id"],
                                    mime_type="video/mp4",
                                    drive_filename=mp4_path.name,
                                )
                            st.session_state["drive_uploaded_video_info"] = info
                            st.toast("Uploaded video to Drive", icon="✅")
                        except Exception as e:
                            st.error(f"Drive upload (video) failed: {e}")
                    else:
                        st.error("No local MP4 file available to upload.")

        if colu2.button("Upload thumbnail to Google Drive", use_container_width=True, key="btn_drive_upload_thumb_from_video"):
            err = _drive_preflight()
            if err:
                st.error(err)
            else:
                thumb_bytes = st.session_state.get("thumb_final_thumbnail_bytes")
                thumb_path = _optional_path(st.session_state.get("last_thumbnail_path"))
                if not (thumb_path and thumb_path.exists()) and thumb_bytes:
                    topic = st.session_state.get("last_topic", "") or "reel"
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    base = f"{ts}-{slugify(topic)}"
                    thumb_path = DEFAULT_OUTPUT_DIR / f"{base}-thumbnail.png"
                    try:
                        thumb_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(thumb_path, "wb") as f:
                            f.write(thumb_bytes)
                        st.session_state["last_thumbnail_path"] = str(thumb_path)
                    except Exception as e:
                        st.error(f"Could not save thumbnail locally: {e}")
                        thumb_path = None

                if not (thumb_path and thumb_path.is_file()):
                    st.error("No thumbnail found to upload. Generate a thumbnail first.")
                else:
                    try:
                        with st.spinner("Uploading thumbnail to Google Drive…"):
                            svc = get_drive_service_from_sa_json(st.session_state["drive_sa_json_bytes"])
                            info = upload_file_to_drive(
                                svc,
                                local_path=thumb_path,
                                parent_folder_id=st.session_state["drive_folder_id"],
                                mime_type="image/png",
                                drive_filename=thumb_path.name,
                            )
                        st.session_state["drive_uploaded_thumbnail_info"] = info
                        st.toast("Uploaded thumbnail to Drive", icon="✅")
                    except Exception as e:
                        st.error(f"Drive upload (thumbnail) failed: {e}")

        info = st.session_state.get("drive_uploaded_video_info") or {}
        if info.get("webViewLink"):
            st.link_button("Open uploaded video in Drive", info["webViewLink"], use_container_width=True)
    else:
        st.caption("No HeyGen video yet. Render to preview.")

# -------------------------------
# Thumbnail UI
# -------------------------------
def init_thumbnail_state():
    defaults = {
        "thumb_transcript": "",
        "thumb_heading_sub": "",
        "thumb_heading_main": "",
        "thumb_visual_mode": "",
        "thumb_entity_name": "",
        "thumb_thumbnail_prompt": "",
        "thumb_cse_results": [],
        "thumb_selected_cse_index": 0,
        "thumb_main_image_bytes": None,
        "thumb_cse_query": "",
        "thumb_cse_next_start": 1,
        "thumb_final_thumbnail_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_thumbnail_section():
    st.subheader("6) Thumbnail Generator")

    if not st.session_state.get("has_script"):
        st.caption("Generate a script first.")
        return

    init_thumbnail_state()

    if "thumb_pending_heading" in st.session_state:
        data = st.session_state.pop("thumb_pending_heading") or {}
        st.session_state["thumb_heading_sub"] = data.get("subheading", "")
        st.session_state["thumb_heading_main"] = data.get("main_heading", "")

    if "thumb_pending_visual" in st.session_state:
        data = st.session_state.pop("thumb_pending_visual") or {}
        st.session_state["thumb_visual_mode"] = data.get("mode", "")
        st.session_state["thumb_entity_name"] = data.get("entity_name", "")
        st.session_state["thumb_thumbnail_prompt"] = data.get("thumbnail_prompt", "")
        st.session_state["thumb_cse_results"] = data.get("cse_results", [])
        st.session_state["thumb_cse_query"] = data.get("cse_query", "")
        st.session_state["thumb_cse_next_start"] = data.get("cse_next_start", 1)

    if "thumb_pending_main_image" in st.session_state:
        st.session_state["thumb_main_image_bytes"] = st.session_state.pop("thumb_pending_main_image")

    if "thumb_pending_final" in st.session_state:
        st.session_state["thumb_final_thumbnail_bytes"] = st.session_state.pop("thumb_pending_final")

    current_script = st.session_state.get("generated_script_text") or st.session_state.get("last_script") or ""
    if not st.session_state.get("thumb_transcript") and current_script:
        st.session_state["thumb_transcript"] = current_script

    st.markdown("### Step 1 — Transcript & Heading")
    transcript = st.text_area(
        "Paste Instagram video transcript",
        key="thumb_transcript",
        height=250,
    )

    if st.button("Generate Heading", type="primary", key="btn_thumb_generate_heading"):
        if not transcript.strip():
            st.warning("Please paste a transcript first.")
        elif not st.session_state.get("openai_api_key"):
            st.warning("Add OPENAI_API_KEY in the sidebar to generate headings.")
        else:
            try:
                client = get_openai_client(st.session_state.get("openai_api_key"))
                with st.spinner("Generating headings…"):
                    result = generate_heading_from_transcript(client, transcript)
                if result:
                    st.session_state["thumb_pending_heading"] = result
                    st.rerun()
            except Exception as e:
                st.error(f"Error generating heading: {e}")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input(
            "Subheading",
            key="thumb_heading_sub",
            placeholder="Subheading (smaller line on top)",
        )
    with col2:
        st.text_input(
            "Main Heading",
            key="thumb_heading_main",
            placeholder="Main heading (big bold line)",
        )

    st.markdown("---")
    st.markdown("### Step 2 — Select / Generate / Upload Main Image")

    if st.button("Analyze Transcript for Visual Concept", key="btn_thumb_analyze"):
        if not transcript.strip():
            st.warning("Please paste a transcript first.")
        elif not st.session_state.get("openai_api_key"):
            st.warning("Add OPENAI_API_KEY in the sidebar to analyze the visual concept.")
        else:
            try:
                client = get_openai_client(st.session_state.get("openai_api_key"))
                with st.spinner("Analyzing visual concept…"):
                    concept = analyze_visual_concept(client, transcript)
                if concept:
                    pending = {
                        "mode": concept.get("mode", ""),
                        "entity_name": concept.get("entity_name", ""),
                        "thumbnail_prompt": concept.get("thumbnail_prompt", ""),
                        "cse_results": [],
                        "cse_query": "",
                        "cse_next_start": 1,
                    }
                    if concept.get("mode") == "entity":
                        api_key = st.session_state.get("google_cse_api_key") or GOOGLE_CSE_API_KEY_ENV or ""
                        cx = st.session_state.get("google_cse_cx") or GOOGLE_CSE_CX_ENV or ""
                        if api_key and cx and concept.get("entity_name"):
                            try:
                                results = cse_image_search(
                                    api_key,
                                    cx,
                                    concept["entity_name"],
                                    start=1,
                                    num=6,
                                )
                                pending["cse_results"] = results
                                pending["cse_query"] = concept["entity_name"]
                                pending["cse_next_start"] = 1 + len(results)
                            except Exception as e:
                                st.error(f"Error calling Google Custom Search: {e}")
                        else:
                            st.warning("Google CSE keys missing, cannot search images.")

                    st.session_state["thumb_pending_visual"] = pending
                    st.rerun()
            except Exception as e:
                st.error(f"Error analyzing concept: {e}")

    if st.session_state.get("thumb_visual_mode") == "entity" and st.session_state.get("thumb_cse_results"):
        st.markdown("#### Choose an image from search results")
        results = st.session_state["thumb_cse_results"]
        num_cols = 3
        cols = st.columns(num_cols)
        for i, item in enumerate(results):
            col = cols[i % num_cols]
            with col:
                if item.get("thumbnail"):
                    st.image(item["thumbnail"], caption=item.get("title") or "", use_column_width=True)
                if st.button("Use this image", key=f"thumb_use_img_{i}"):
                    try:
                        url = item.get("original") or ""
                        if not url:
                            st.error("No image URL found for this result.")
                            return
                        resp = requests.get(url, timeout=20)
                        resp.raise_for_status()
                        st.session_state["thumb_pending_main_image"] = resp.content
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error downloading image: {e}")

        api_key = st.session_state.get("google_cse_api_key") or GOOGLE_CSE_API_KEY_ENV or ""
        cx = st.session_state.get("google_cse_cx") or GOOGLE_CSE_CX_ENV or ""
        if api_key and cx and st.session_state.get("thumb_cse_query"):
            if st.button("Load more images", key="btn_thumb_load_more"):
                try:
                    more = cse_image_search(
                        api_key,
                        cx,
                        st.session_state.get("thumb_cse_query", ""),
                        start=st.session_state.get("thumb_cse_next_start", 1),
                        num=6,
                    )
                    if more:
                        st.session_state["thumb_cse_results"].extend(more)
                        st.session_state["thumb_cse_next_start"] += len(more)
                        st.rerun()
                    else:
                        st.info("No more images found.")
                except Exception as e:
                    st.error(f"Error loading more images: {e}")

    if st.session_state.get("thumb_visual_mode") == "prompt":
        st.markdown("#### AI Image Prompt")
        st.text_area(
            "Image prompt for gpt-image-1 (editable)",
            key="thumb_thumbnail_prompt",
            height=150,
        )

        if st.button("Generate AI Image", key="btn_thumb_gen_ai"):
            prompt = st.session_state.get("thumb_thumbnail_prompt", "").strip()
            if not prompt:
                st.warning("Please provide an image prompt.")
            elif not st.session_state.get("openai_api_key"):
                st.warning("Add OPENAI_API_KEY in the sidebar to generate an AI image.")
            else:
                try:
                    client = get_openai_client(st.session_state.get("openai_api_key"))
                    with st.spinner("Generating image…"):
                        img_bytes = generate_ai_image(client, prompt)
                    if img_bytes:
                        st.session_state["thumb_pending_main_image"] = img_bytes
                        st.rerun()
                except Exception as e:
                    st.error(f"Error generating AI image: {e}")

    st.markdown("#### Or upload an image from your computer")
    uploaded_file = st.file_uploader(
        "Upload a PNG/JPG/JPEG/WEBP image",
        type=["png", "jpg", "jpeg", "webp"],
        key="thumb_user_image_uploader",
    )
    if uploaded_file is not None:
        st.session_state["thumb_main_image_bytes"] = uploaded_file.getvalue()
        st.session_state["thumb_final_thumbnail_bytes"] = None
        st.success("Uploaded image stored for final thumbnail.")

    if st.session_state.get("thumb_main_image_bytes"):
        st.markdown("#### Current main image preview")
        st.image(st.session_state["thumb_main_image_bytes"], use_column_width=True)

    st.markdown("---")
    st.markdown("### Step 3 — Compose Final Thumbnail")

    base_color = st.radio(
        "Select background color / template",
        options=["Blue", "Black", "White"],
        horizontal=True,
        key="thumb_base_color",
    )

    ready_for_final = (
        bool(st.session_state.get("thumb_heading_sub", "").strip())
        and bool(st.session_state.get("thumb_heading_main", "").strip())
        and st.session_state.get("thumb_main_image_bytes") is not None
    )

    if not ready_for_final:
        st.info("To generate a final thumbnail, you need: a heading (sub + main) and a selected main image.")

    if st.button("Generate Final Thumbnail", type="primary", disabled=not ready_for_final, key="btn_thumb_final"):
        if ready_for_final:
            gemini_api_key = st.session_state.get("gemini_api_key") or GEMINI_API_KEY_ENV or ""
            gemini_client = get_gemini_client(gemini_api_key)
            if not gemini_client:
                st.error("Gemini client not configured. Add GEMINI_API_KEY in the sidebar.")
            else:
                with st.spinner("Composing thumbnail…"):
                    final_bytes = compose_thumbnail_with_gemini(
                        gemini_client,
                        base_color=base_color,
                        subheading=st.session_state.get("thumb_heading_sub", ""),
                        main_heading=st.session_state.get("thumb_heading_main", ""),
                        main_image_bytes=st.session_state.get("thumb_main_image_bytes"),
                    )
                if final_bytes:
                    # Save locally (and optionally upload to Drive) before rerun.
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    topic = st.session_state.get("last_topic", "") or "reel"
                    base = f"{ts}-{slugify(topic)}"
                    png_path = DEFAULT_OUTPUT_DIR / f"{base}-thumbnail.png"
                    try:
                        png_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(png_path, "wb") as f:
                            f.write(final_bytes)
                        st.session_state["last_thumbnail_path"] = str(png_path)
                    except Exception as e:
                        st.warning(f"Could not save thumbnail locally: {e}")

                    if st.session_state.get("drive_enabled"):
                        thumb_hash = sha256_bytes(final_bytes)
                        already = st.session_state.get("drive_uploaded_thumbnail_hash")
                        if already == thumb_hash:
                            pass
                        else:
                            err = _drive_preflight()
                            if err:
                                st.warning(err)
                            else:
                                folder_id = (st.session_state.get("drive_folder_id") or "").strip()
                                sa_json_bytes = st.session_state.get("drive_sa_json_bytes")
                                try:
                                    with st.spinner("Uploading thumbnail to Google Drive…"):
                                        svc = get_drive_service_from_sa_json(sa_json_bytes)
                                        info = upload_file_to_drive(
                                            svc,
                                            local_path=png_path,
                                            parent_folder_id=folder_id,
                                            mime_type="image/png",
                                            drive_filename=png_path.name,
                                        )
                                    st.session_state["drive_uploaded_thumbnail_hash"] = thumb_hash
                                    st.session_state["drive_uploaded_thumbnail_info"] = info
                                    st.toast("Uploaded thumbnail to Drive", icon="✅")
                                except Exception as e:
                                    st.error(f"Drive upload (thumbnail) failed: {e}")

                    st.session_state["thumb_pending_final"] = final_bytes
                    st.rerun()

    if st.session_state.get("thumb_final_thumbnail_bytes"):
        st.markdown("#### Final Thumbnail")
        st.image(st.session_state["thumb_final_thumbnail_bytes"], use_column_width=True)
        if st.button("Upload thumbnail to Google Drive", use_container_width=True, key="btn_drive_upload_thumbnail"):
            err = _drive_preflight()
            if err:
                st.error(err)
            else:
                thumb_bytes = st.session_state.get("thumb_final_thumbnail_bytes") or b""
                thumb_path = _optional_path(st.session_state.get("last_thumbnail_path"))
                if not (thumb_path and thumb_path.is_file()):
                    topic = st.session_state.get("last_topic", "") or "reel"
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    base = f"{ts}-{slugify(topic)}"
                    thumb_path = DEFAULT_OUTPUT_DIR / f"{base}-thumbnail.png"
                    try:
                        thumb_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(thumb_path, "wb") as f:
                            f.write(thumb_bytes)
                        st.session_state["last_thumbnail_path"] = str(thumb_path)
                    except Exception as e:
                        st.error(f"Could not save thumbnail locally: {e}")
                        thumb_path = None
                if not (thumb_path and thumb_path.is_file()):
                    st.error("Thumbnail file missing; please try generating again.")
                else:
                    try:
                        with st.spinner("Uploading thumbnail to Google Drive…"):
                            svc = get_drive_service_from_sa_json(st.session_state["drive_sa_json_bytes"])
                            info = upload_file_to_drive(
                                svc,
                                local_path=thumb_path,
                                parent_folder_id=st.session_state["drive_folder_id"],
                                mime_type="image/png",
                                drive_filename=thumb_path.name,
                            )
                        st.session_state["drive_uploaded_thumbnail_info"] = info
                        st.toast("Uploaded thumbnail to Drive", icon="✅")
                    except Exception as e:
                        st.error(f"Drive upload (thumbnail) failed: {e}")
        info = st.session_state.get("drive_uploaded_thumbnail_info") or {}
        if info.get("webViewLink"):
            st.link_button("Open uploaded thumbnail in Drive", info["webViewLink"], use_container_width=True)
        st.download_button(
            "Download Thumbnail",
            data=st.session_state["thumb_final_thumbnail_bytes"],
            file_name="thumbnail_final.png",
            mime="image/png",
        )

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
                source="pasted",
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

# --- Upload news TXT (optional) ---
with st.expander("Upload news list (.txt)", expanded=False):
    uploaded_news_file = st.file_uploader("Upload TXT", type=["txt"], key="news_upload_txt")
    parsed_items: List[Dict[str, Any]] = []
    if uploaded_news_file is not None:
        try:
            raw_txt = uploaded_news_file.read().decode("utf-8", errors="ignore")
            parsed_items = parse_uploaded_news_txt(raw_txt)
            st.caption(f"Parsed {len(parsed_items)} item(s) from upload.")
            if parsed_items:
                st.dataframe(parsed_items, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not parse upload: {e}")

    colu1, _ = st.columns([1, 3])
    add_uploaded = colu1.button(
        "Add uploaded news to list",
        type="primary",
        use_container_width=True,
        key="btn_add_uploaded_news",
        disabled=not parsed_items,
    )
    if add_uploaded and parsed_items:
        existing = st.session_state.get("last_news_items") or []
        merged = merge_news_items(existing, parsed_items)
        st.session_state["last_news_items"] = merged
        st.session_state["last_news_mode"] = st.session_state.get("last_news_mode") or "Uploaded"
        st.session_state["last_news_range"] = st.session_state.get("last_news_range") or "Custom"
        st.session_state["last_news_name"] = st.session_state.get("last_news_name") or ""
        st.rerun()

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

        script_with_links = script
        with st.spinner("Removing source links for voiceover (GPT-5 mini)…"):
            script_clean = strip_source_links_gpt5_mini(client, script_with_links)
        set_script_state(
            script=script_clean,
            script_clean=script_clean,
            script_with_links=script_with_links,
            topic=topic,
            facts_payload=facts_payload,
            style_samples=retrieved,
            source="generated",
        )
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

                    with st.spinner("Researching selected news with GPT-5.2…"):
                        research_dump = research_news_dump_gpt52(
                            client,
                            topic=topic_from_news,
                            anchor=anchor,
                        )

                    with st.spinner("Generating script from research…"):
                        messages = make_prompt(
                            topic_from_news,
                            retrieved,
                            news_anchor=anchor,
                            research_dump=research_dump,
                        )
                        try:
                            script = generate_script(
                                client,
                                "gpt-5",
                                messages,
                                temperature=temperature,
                                use_web_search=False,
                            )
                            st.session_state.pop("audio_bytes", None)
                            st.session_state.pop("heygen_video_url", None)
                        except Exception as e:
                            st.error(f"Generation failed: {e}")
                            st.stop()

                    script_with_links = script
                    with st.spinner("Removing source links for voiceover (GPT-5 mini)…"):
                        script_clean = strip_source_links_gpt5_mini(client, script_with_links)
                    set_script_state(
                        script=script_clean,
                        script_clean=script_clean,
                        script_with_links=script_with_links,
                        topic=topic_from_news,
                        facts_payload=None,
                        style_samples=retrieved,
                        source="news",
                    )
                    st.rerun()

# -------------------------------
# Render script editor ONCE + then downstream sections
# -------------------------------
render_script_editor_once()
render_revision_controls()
render_voiceover_section()
render_video_section()
render_thumbnail_section()
