"""
CompanyLens - AI company intelligence dashboard.

Pipeline:
URL -> scrape -> clean text -> limit -> ONE prompt to ChatGroq -> JSON
    -> robust parse + validate -> render dashboard
"""

import os
import re
import json
import math
import html as htmllib
from urllib.parse import urlparse, urljoin

import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
try:
    GROQ_API_KEY = GROQ_API_KEY or st.secrets.get("GROQ_API_KEY", "")
except Exception:
    pass

MODEL_NAME = "openai/gpt-oss-120b"
MAX_TOTAL_CHARS = 11000
MAX_CHARS_PER_PAGE = 4500
REQUEST_TIMEOUT = 12
EXTRA_PATHS = ["/about", "/pricing", "/product", "/solutions", "/company"]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

st.set_page_config(
    page_title="CompanyLens - Company Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: #0b1020; color: #e6e9f2; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1150px; padding-top: 0.5rem; padding-bottom: 3rem; }

.cl-nav { display:flex; align-items:center; justify-content:space-between;
  padding: 16px 4px; border-bottom: 1px solid #1c2440; margin-bottom: 8px; }
.cl-logo { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; color:#fff; }
.cl-logo span { color:#6c8cff; }
.cl-nav-links { display:flex; gap:22px; font-size:0.9rem; color:#8b95b5; }
.cl-nav-links b { color:#e6e9f2; font-weight:600; }

.cl-hero { text-align:center; padding: 56px 12px 26px 12px; }
.cl-badge { display:inline-block; padding:5px 14px; border-radius:999px; font-size:0.78rem;
  font-weight:600; color:#9db2ff; background:rgba(108,140,255,0.12);
  border:1px solid rgba(108,140,255,0.3); margin-bottom:18px; }
.cl-hero h1 { font-size: 3rem; font-weight: 800; letter-spacing:-0.035em; line-height:1.1;
  color:#fff; margin: 0 0 14px 0; }
.cl-hero h1 em { font-style:normal; background: linear-gradient(90deg,#6c8cff,#b06cff);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.cl-hero p { color:#8b95b5; font-size:1.08rem; max-width:620px; margin:0 auto; }

div[data-testid="stTextInput"] input {
  background:#121a33; border:1px solid #26305a; color:#fff; border-radius:12px;
  padding: 14px 16px; font-size:1rem; }
div[data-testid="stTextInput"] input:focus { border-color:#6c8cff; box-shadow:0 0 0 2px rgba(108,140,255,.25); }
div[data-testid="stTextInput"] label { display:none; }
div.stButton > button {
  width:100%; height:52px; border-radius:12px; border:none; font-weight:700; font-size:1rem;
  color:#fff; background: linear-gradient(90deg,#5b7cff,#9b5cff); }
div.stButton > button:hover { filter:brightness(1.1); color:#fff; border:none; }

.cl-steps { display:flex; gap:10px; margin: 22px 0 8px 0; flex-wrap:wrap; }
.cl-step { flex:1; min-width:200px; padding:12px 14px; border-radius:12px; background:#121a33;
  border:1px solid #1f2850; font-size:0.9rem; color:#6f7aa0; display:flex; gap:10px; align-items:center; }
.cl-step.done { color:#5ee0a0; border-color:rgba(94,224,160,.35); }
.cl-step.active { color:#9db2ff; border-color:#6c8cff; background:rgba(108,140,255,.1); }
.cl-step .ic { font-weight:800; width:18px; text-align:center; }

.cl-card { background:#121a33; border:1px solid #1f2850; border-radius:16px; padding:22px 24px;
  margin-bottom:18px; }
.cl-card h3 { margin:0 0 12px 0; font-size:0.78rem; letter-spacing:.09em; text-transform:uppercase;
  color:#8b95b5; font-weight:700; }
.cl-card p, .cl-card li { color:#c9d0e6; line-height:1.6; font-size:0.95rem; }
.cl-card ul { margin:0; padding-left:18px; }
.cl-muted { color:#8b95b5; }

.cl-company { font-size:2rem; font-weight:800; color:#fff; letter-spacing:-.02em; margin:0; }
.cl-tagline { color:#9db2ff; font-size:1.05rem; margin:4px 0 0 0; }
.cl-stage { display:inline-block; padding:4px 12px; border-radius:999px; font-size:.8rem; font-weight:700;
  color:#5ee0a0; background:rgba(94,224,160,.12); border:1px solid rgba(94,224,160,.3); margin-top:12px; }

.cl-bar-row { margin-bottom:14px; }
.cl-bar-head { display:flex; justify-content:space-between; font-size:.88rem; color:#c9d0e6; margin-bottom:6px; }
.cl-bar-track { height:8px; background:#1a2447; border-radius:999px; overflow:hidden; }
.cl-bar-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#5b7cff,#9b5cff); }

.cl-chip { display:inline-block; padding:6px 12px; margin:0 8px 8px 0; border-radius:10px;
  background:#1a2447; color:#c9d0e6; font-size:.86rem; border:1px solid #26305a; }
.cl-item { padding:12px 0; border-bottom:1px solid #1c2440; }
.cl-item:last-child { border-bottom:none; padding-bottom:0; }
.cl-item b { color:#fff; font-size:.95rem; }
.cl-item div { color:#9aa4c4; font-size:.88rem; margin-top:3px; line-height:1.5; }
.cl-risk b { color:#ff8a8a; }
.cl-grow b { color:#5ee0a0; }
.cl-good li::marker { color:#5ee0a0; }

.cl-footer { text-align:center; color:#5d6788; font-size:.82rem; padding:36px 0 8px 0;
  border-top:1px solid #1c2440; margin-top:30px; }

@media (max-width: 768px) {
  .cl-hero h1 { font-size:2.1rem; }
  .cl-nav-links { display:none; }
  .cl-step { min-width:100%; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def esc(value) -> str:
    return htmllib.escape(str(value if value is not None else ""))


def normalize_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if not re.match(r"^https?://", raw, flags=re.I):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.netloc or "." not in parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/") or raw


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text or "")
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def fetch_html(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and "html" in r.headers.get("Content-Type", "").lower():
            return r.text
    except requests.RequestException:
        pass
    return None


def extract_text(page_html: str) -> str:
    text = ""
    try:
        text = trafilatura.extract(page_html, include_comments=False, include_tables=True) or ""
    except Exception:
        text = ""
    if len(text) < 200:
        try:
            soup = BeautifulSoup(page_html, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "form"]):
                tag.decompose()
            title = soup.title.get_text(strip=True) if soup.title else ""
            meta = soup.find("meta", attrs={"name": "description"})
            desc = meta.get("content", "") if meta else ""
            body = soup.get_text("\n", strip=True)
            text = "\n".join(p for p in [title, desc, body] if p)
        except Exception:
            pass
    return clean_text(text)


def scrape_company(url: str):
    """Returns (combined_text, pages_ok). Ignores failing sub-pages."""
    base = urlparse(url)
    root = f"{base.scheme}://{base.netloc}"
    candidates = [url] + [urljoin(root, p) for p in EXTRA_PATHS]

    seen, chunks, pages_ok, total = set(), [], 0, 0
    for page_url in candidates:
        if page_url in seen:
            continue
        seen.add(page_url)
        page_html = fetch_html(page_url)
        if not page_html:
            continue
        text = extract_text(page_html)
        if len(text) < 80:
            continue
        text = text[:MAX_CHARS_PER_PAGE]
        remaining = MAX_TOTAL_CHARS - total
        if remaining <= 0:
            break
        text = text[:remaining]
        path = urlparse(page_url).path or "/"
        chunks.append(f"[PAGE: {path}]\n{text}")
        total += len(text)
        pages_ok += 1

    return "\n\n".join(chunks)[:MAX_TOTAL_CHARS], pages_ok


# ----------------------------------------------------------------------------
# Prompt + LLM
# ----------------------------------------------------------------------------
JSON_TEMPLATE = """{
  "company_name": "",
  "tagline": "",
  "company_score": 0,
  "score_rationale": "",
  "scores_breakdown": {
    "product": 0,
    "market_position": 0,
    "revenue_quality": 0,
    "moat_strength": 0,
    "growth_potential": 0
  },
  "revenue_model": {
    "primary": "",
    "details": ""
  },
  "target_market": "",
  "key_products": [],
  "competitors": [
    {"name": "", "why": ""}
  ],
  "competitive_radar": {
    "price": 0,
    "technology": 0,
    "reach": 0,
    "support": 0,
    "speed": 0
  },
  "strengths": [],
  "risks": [
    {"title": "", "detail": ""}
  ],
  "growth_opportunities": [
    {"title": "", "detail": ""}
  ],
  "moat": "",
  "stage": "",
  "summary": ""
}"""


def build_prompt(url: str, content: str) -> str:
    return f"""You are a senior company analyst producing a structured intelligence report.

Analyze the company behind this website using ONLY the scraped content below plus well-established general knowledge about its industry.

Website: {url}

=== SCRAPED WEBSITE CONTENT ===
{content}
=== END CONTENT ===

OUTPUT RULES (critical):
- Return ONLY a single valid JSON object.
- No markdown. No ```json fences. No explanation or text outside the JSON.
- Use exactly this structure and these keys:

{JSON_TEMPLATE}

FIELD RULES:
- All score values (company_score, every value in scores_breakdown, every value in competitive_radar) must be INTEGERS from 0 to 100.
- key_products: list of 3-6 short strings.
- competitors: 3-5 objects with "name" and a one-sentence "why".
- strengths: list of 3-5 short strings.
- risks: 3-4 objects with "title" and "detail".
- growth_opportunities: 3-4 objects with "title" and "detail".
- stage: one of "Early-stage", "Growth", "Scale-up", "Mature", or "Unclear".
- summary: 3-5 sentence executive summary.
- Do NOT invent facts (funding, revenue, customer counts, headcount) that the website content does not support.
- When information is unavailable, say so plainly (e.g. "Not disclosed on website") and lower the relevant scores rather than guessing.
- Escape all quotes inside strings properly so the JSON is valid."""


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.3,
        max_tokens=2000,
    )


# ----------------------------------------------------------------------------
# Robust JSON parsing + validation
# ----------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_balanced(text: str):
    """Return first balanced {...} block, or the tail from first '{' if unbalanced."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]  # truncated


def _repair(text: str) -> str:
    """Fix trailing commas, smart quotes, and close truncated structures."""
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = re.sub(r",\s*([}\]])", r"\1", text)

    stack, in_str, escape = [], False, False
    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()

    if in_str:
        text += '"'
    text = re.sub(r",\s*$", "", text.rstrip())
    for opener in reversed(stack):
        text += "}" if opener == "{" else "]"
    return re.sub(r",\s*([}\]])", r"\1", text)


def parse_json_response(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("The model returned an empty response.")
    cleaned = _strip_fences(raw)
    candidates = [cleaned]
    block = _extract_balanced(cleaned)
    if block:
        candidates.append(block)
        candidates.append(_repair(block))

    last_err = None
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError(f"Could not parse model output as JSON ({last_err}).")


def _int_score(value, default=0) -> int:
    try:
        if isinstance(value, str):
            m = re.search(r"-?\d+(\.\d+)?", value)
            value = float(m.group()) if m else default
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _str(value, default="") -> str:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return default
    return str(value).strip() or default


def _str_list(value) -> list:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        if isinstance(v, dict):
            v = v.get("name") or v.get("title") or next(iter(v.values()), "")
        s = _str(v)
        if s:
            out.append(s)
    return out


def _obj_list(value, k1, k2) -> list:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            a, b = _str(item.get(k1)), _str(item.get(k2))
        else:
            a, b = _str(item), ""
        if a:
            out.append({k1: a, k2: b})
    return out


def validate_report(data: dict) -> dict:
    sb = data.get("scores_breakdown") if isinstance(data.get("scores_breakdown"), dict) else {}
    rm = data.get("revenue_model") if isinstance(data.get("revenue_model"), dict) else {}
    cr = data.get("competitive_radar") if isinstance(data.get("competitive_radar"), dict) else {}

    breakdown = {k: _int_score(sb.get(k)) for k in
                 ["product", "market_position", "revenue_quality", "moat_strength", "growth_potential"]}
    radar = {k: _int_score(cr.get(k)) for k in ["price", "technology", "reach", "support", "speed"]}

    score = _int_score(data.get("company_score"), -1)
    if score < 0 or (score == 0 and any(breakdown.values())):
        vals = list(breakdown.values())
        score = int(round(sum(vals) / len(vals))) if vals else 0

    if isinstance(data.get("revenue_model"), str):
        rm = {"primary": data["revenue_model"], "details": ""}

    return {
        "company_name": _str(data.get("company_name"), "Unknown company"),
        "tagline": _str(data.get("tagline")),
        "company_score": score,
        "score_rationale": _str(data.get("score_rationale")),
        "scores_breakdown": breakdown,
        "revenue_model": {
            "primary": _str(rm.get("primary"), "Not disclosed"),
            "details": _str(rm.get("details")),
        },
        "target_market": _str(data.get("target_market"), "Not clearly stated"),
        "key_products": _str_list(data.get("key_products")),
        "competitors": _obj_list(data.get("competitors"), "name", "why"),
        "competitive_radar": radar,
        "strengths": _str_list(data.get("strengths")),
        "risks": _obj_list(data.get("risks"), "title", "detail"),
        "growth_opportunities": _obj_list(data.get("growth_opportunities"), "title", "detail"),
        "moat": _str(data.get("moat"), "Not clearly identifiable"),
        "stage": _str(data.get("stage"), "Unclear"),
        "summary": _str(data.get("summary")),
    }


def analyze_with_groq(url: str, content: str) -> dict:
    llm = get_llm()
    prompt = build_prompt(url, content)

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    try:
        return validate_report(parse_json_response(raw))
    except ValueError:
        # One retry: ask the model to re-emit valid JSON only.
        retry_prompt = (
            prompt
            + "\n\nYour previous reply was not valid JSON. Reply again with ONLY the "
              "complete, valid JSON object and nothing else."
        )
        response = llm.invoke(retry_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        return validate_report(parse_json_response(raw))


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
STEPS = [
    "Fetching company website",
    "Extracting page content",
    "Running AI analysis",
    "Building intelligence report",
]


def steps_html(active: int, all_done: bool = False) -> str:
    parts = []
    for i, label in enumerate(STEPS):
        if all_done or i < active:
            cls, icon = "done", "✓"
        elif i == active:
            cls, icon = "active", "●"
        else:
            cls, icon = "", "○"
        parts.append(f'<div class="cl-step {cls}"><span class="ic">{icon}</span>{esc(label)}</div>')
    return '<div class="cl-steps">' + "".join(parts) + "</div>"


def score_color(score: int) -> str:
    if score >= 75:
        return "#5ee0a0"
    if score >= 50:
        return "#ffc857"
    return "#ff7a7a"


def score_ring_svg(score: int) -> str:
    r, c = 62, 2 * math.pi * 62
    dash = c * score / 100
    color = score_color(score)
    return (
        '<svg viewBox="0 0 160 160" width="170" height="170" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="80" cy="80" r="{r}" fill="none" stroke="#1a2447" stroke-width="12"/>'
        f'<circle cx="80" cy="80" r="{r}" fill="none" stroke="{color}" stroke-width="12" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {c:.1f}" transform="rotate(-90 80 80)"/>'
        f'<text x="80" y="88" text-anchor="middle" font-size="40" font-weight="800" fill="#ffffff" '
        f'font-family="Inter,sans-serif">{score}</text>'
        '<text x="80" y="108" text-anchor="middle" font-size="11" fill="#8b95b5" '
        'font-family="Inter,sans-serif">out of 100</text></svg>'
    )


def radar_svg(radar: dict) -> str:
    labels = [("price", "Price"), ("technology", "Technology"), ("reach", "Reach"),
              ("support", "Support"), ("speed", "Speed")]
    cx, cy, R = 170, 155, 100
    n = len(labels)

    def pt(i, frac):
        ang = math.radians(-90 + i * 360 / n)
        return cx + R * frac * math.cos(ang), cy + R * frac * math.sin(ang)

    out = ['<svg viewBox="0 0 340 320" width="100%" style="max-width:380px" '
           'xmlns="http://www.w3.org/2000/svg">']
    for level in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{pt(i, level)[0]:.1f},{pt(i, level)[1]:.1f}" for i in range(n))
        out.append(f'<polygon points="{pts}" fill="none" stroke="#26305a" stroke-width="1"/>')
    for i in range(n):
        x, y = pt(i, 1.0)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#26305a"/>')
    data_pts = " ".join(
        f"{pt(i, radar.get(k, 0) / 100)[0]:.1f},{pt(i, radar.get(k, 0) / 100)[1]:.1f}"
        for i, (k, _) in enumerate(labels)
    )
    out.append(f'<polygon points="{data_pts}" fill="rgba(108,140,255,0.3)" stroke="#6c8cff" stroke-width="2"/>')
    for i, (k, label) in enumerate(labels):
        x, y = pt(i, radar.get(k, 0) / 100)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#9db2ff"/>')
        lx, ly = pt(i, 1.2)
        anchor = "middle" if abs(lx - cx) < 8 else ("start" if lx > cx else "end")
        out.append(
            f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" font-size="12" fill="#c9d0e6" '
            f'font-family="Inter,sans-serif">{label} ({radar.get(k, 0)})</text>'
        )
    out.append("</svg>")
    return "".join(out)


def bar_rows(items: dict) -> str:
    rows = []
    for key, val in items.items():
        label = key.replace("_", " ").title()
        rows.append(
            '<div class="cl-bar-row">'
            f'<div class="cl-bar-head"><span>{esc(label)}</span><span>{val}/100</span></div>'
            f'<div class="cl-bar-track"><div class="cl-bar-fill" style="width:{val}%"></div></div>'
            "</div>"
        )
    return "".join(rows)


def items_html(items: list, k1: str, k2: str, cls: str) -> str:
    if not items:
        return '<p class="cl-muted">Not enough information available.</p>'
    rows = "".join(
        f'<div class="cl-item {cls}"><b>{esc(i[k1])}</b>'
        + (f'<div>{esc(i[k2])}</div>' if i.get(k2) else "")
        + "</div>"
        for i in items
    )
    return rows


def render_report(rep: dict, url: str):
    st.markdown(
        '<div class="cl-card">'
        f'<p class="cl-company">{esc(rep["company_name"])}</p>'
        f'<p class="cl-tagline">{esc(rep["tagline"])}</p>'
        f'<span class="cl-stage">Stage: {esc(rep["stage"])}</span>'
        f'<span class="cl-muted" style="margin-left:12px;font-size:.85rem">{esc(url)}</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Score + breakdown
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            '<div class="cl-card" style="text-align:center"><h3>Company Score</h3>'
            f'{score_ring_svg(rep["company_score"])}'
            f'<p class="cl-muted" style="font-size:.85rem;margin-top:8px">{esc(rep["score_rationale"])}</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="cl-card"><h3>Score Breakdown</h3>{bar_rows(rep["scores_breakdown"])}</div>',
            unsafe_allow_html=True,
        )

    # Executive summary
    st.markdown(
        f'<div class="cl-card"><h3>Executive Summary</h3><p>{esc(rep["summary"] or "No summary available.")}</p></div>',
        unsafe_allow_html=True,
    )

    # Revenue model + target market
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="cl-card"><h3>Revenue Model</h3>'
            f'<p><b style="color:#fff">{esc(rep["revenue_model"]["primary"])}</b></p>'
            f'<p>{esc(rep["revenue_model"]["details"])}</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="cl-card"><h3>Target Market</h3><p>{esc(rep["target_market"])}</p></div>',
            unsafe_allow_html=True,
        )

    # Key products
    chips = "".join(f'<span class="cl-chip">{esc(p)}</span>' for p in rep["key_products"]) \
        or '<p class="cl-muted">No products identified.</p>'
    st.markdown(f'<div class="cl-card"><h3>Key Products</h3>{chips}</div>', unsafe_allow_html=True)

    # Competitors + radar
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            f'<div class="cl-card"><h3>Competitors</h3>{items_html(rep["competitors"], "name", "why", "")}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="cl-card" style="text-align:center"><h3>Competitive Radar</h3>'
            f'{radar_svg(rep["competitive_radar"])}</div>',
            unsafe_allow_html=True,
        )

    # Strengths + Risks
    c1, c2 = st.columns(2)
    with c1:
        lis = "".join(f"<li>{esc(s)}</li>" for s in rep["strengths"]) or "<li>None identified.</li>"
        st.markdown(f'<div class="cl-card"><h3>Strengths</h3><ul class="cl-good">{lis}</ul></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="cl-card"><h3>Risks</h3>{items_html(rep["risks"], "title", "detail", "cl-risk")}</div>',
            unsafe_allow_html=True,
        )

    # Growth + Moat
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            '<div class="cl-card"><h3>Growth Opportunities</h3>'
            f'{items_html(rep["growth_opportunities"], "title", "detail", "cl-grow")}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(f'<div class="cl-card"><h3>Moat</h3><p>{esc(rep["moat"])}</p></div>',
                    unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Page layout
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="cl-nav"><div class="cl-logo">Company<span>Lens</span></div>'
    '<div class="cl-nav-links"><b>Analyze</b><span>How it works</span><span>About</span></div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="cl-hero"><div class="cl-badge">✨ AI-powered company intelligence</div>'
    "<h1>See any company <em>clearly</em></h1>"
    "<p>Paste a company website and get a structured intelligence report: score, revenue model, "
    "competitors, risks and growth opportunities in seconds.</p></div>",
    unsafe_allow_html=True,
)

col_in, col_btn = st.columns([4, 1])
with col_in:
    url_input = st.text_input(
        "Company website URL",
        placeholder="e.g. stripe.com or https://www.notion.so",
        label_visibility="collapsed",
    )
with col_btn:
    analyze = st.button("Analyze Now")

steps_slot = st.empty()
error_slot = st.empty()

if "report" in st.session_state and not analyze:
    render_report(st.session_state["report"], st.session_state.get("report_url", ""))

if analyze:
    st.session_state.pop("report", None)

    if not GROQ_API_KEY:
        error_slot.error(
            "GROQ_API_KEY is missing. Add it under Settings → Secrets in Streamlit Cloud "
            "(GROQ_API_KEY = \"your_key\") or set it as an environment variable."
        )
        st.stop()

    url = normalize_url(url_input)
    if not url:
        error_slot.warning("Please enter a valid company website, e.g. stripe.com")
        st.stop()

    try:
        # Step 1: fetch
        steps_slot.markdown(steps_html(0), unsafe_allow_html=True)
        first_html = fetch_html(url)
        if not first_html:
            # try www. variant as a fallback
            p = urlparse(url)
            if not p.netloc.startswith("www."):
                alt = f"{p.scheme}://www.{p.netloc}{p.path}"
                if fetch_html(alt):
                    url = alt

        # Step 2: extract
        steps_slot.markdown(steps_html(1), unsafe_allow_html=True)
        content, pages_ok = scrape_company(url)
        if pages_ok == 0 or len(content) < 100:
            steps_slot.empty()
            error_slot.error(
                "Couldn't read any content from that website. It may block automated access, "
                "require JavaScript to render, or the URL may be wrong."
            )
            st.stop()

        # Step 3: AI analysis
        steps_slot.markdown(steps_html(2), unsafe_allow_html=True)
        report = analyze_with_groq(url, content)

        # Step 4: build report
        steps_slot.markdown(steps_html(3), unsafe_allow_html=True)
        st.session_state["report"] = report
        st.session_state["report_url"] = url
        steps_slot.markdown(steps_html(4, all_done=True), unsafe_allow_html=True)

        render_report(report, url)

    except ValueError as e:
        steps_slot.empty()
        error_slot.error(f"The AI response couldn't be processed: {e} Please try again.")
    except Exception as e:  # noqa: BLE001
        steps_slot.empty()
        error_slot.error(f"Something went wrong during analysis: {type(e).__name__}: {e}")

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="cl-footer">CompanyLens · Powered by Groq · '
    "Reports are AI-generated from public website content and may contain errors.</div>",
    unsafe_allow_html=True,
)
