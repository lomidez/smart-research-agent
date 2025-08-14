# Full code for research agent
import argparse, os, hashlib, json, re, sqlite3, sys, uuid, time, pathlib, asyncio
from datetime import datetime
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import feedparser
import trafilatura
from bs4 import BeautifulSoup

import chromadb
import numpy as np
from chromadb.config import Settings

import io
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from contextlib import suppress

from dotenv import load_dotenv
load_dotenv()

from observability import ls_start_root_run, ls_end_root_run, ls_span, ls_log_llm_call
# ========================== CLI status prints =========================
def status(msg: str):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)

# =============================== Structured logs ==============================
import logging

LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

_run_id_ctx: ContextVar[str] = ContextVar("_run_id", default="-")

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "event": getattr(record, "event", None) or "log",
            "run_id": getattr(record, "run_id", None) or _run_id_ctx.get(),
        }
        # Merge extra dict payload if provided
        extra_payload = getattr(record, "extra_payload", None)
        if isinstance(extra_payload, dict):
            payload.update(extra_payload)
        # If the basic message is JSON, merge it; else keep as "msg"
        try:
            msg_dict = json.loads(record.getMessage())
            if isinstance(msg_dict, dict):
                payload.update(msg_dict)
            else:
                payload["msg"] = record.getMessage()
        except Exception:
            payload["msg"] = record.getMessage()
        return json.dumps(payload, ensure_ascii=False)

def _make_logger():
    logger = logging.getLogger("agent")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.handlers[:] = []

    fileh = RotatingFileHandler(
        os.path.join(LOG_DIR, "agent.log"),
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8"
    )
    fileh.setFormatter(JsonFormatter())
    logger.addHandler(fileh)
    return logger

log = _make_logger()

def new_run_id() -> str:
    rid = uuid.uuid4().hex
    _run_id_ctx.set(rid)
    return rid

def log_span(event_name: str):
    def deco(fn):
        if asyncio.iscoroutinefunction(fn):
            async def _async(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    out = await fn(*args, **kwargs)
                    dur = int((time.perf_counter() - t0) * 1000)
                    log.info("", extra={"event": event_name, "extra_payload": {"duration_ms": dur}})
                    return out
                except Exception as e:
                    dur = int((time.perf_counter() - t0) * 1000)
                    log.error("", extra={"event": event_name + ".err", "extra_payload": {"duration_ms": dur, "error": str(e)}})
                    raise
            return _async
        else:
            def _sync(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    out = fn(*args, **kwargs)
                    dur = int((time.perf_counter() - t0) * 1000)
                    log.info("", extra={"event": event_name, "extra_payload": {"duration_ms": dur}})
                    return out
                except Exception as e:
                    dur = int((time.perf_counter() - t0) * 1000)
                    log.error("", extra={"event": event_name + ".err", "extra_payload": {"duration_ms": dur, "error": str(e)}})
                    raise
            return _sync
    return deco

# ============================== Prompts loader ================================
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "./prompts")

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    try:
        return load_text(path)
    except FileNotFoundError:
        return f"[MISSING PROMPT: {filename}]"

def render_template(filename: str, **vars):
    """Very small {{var}} replacement (no external deps)."""
    tpl = load_prompt(filename)
    out = tpl
    for k, v in vars.items():
        out = out.replace("{{" + k + "}}", str(v) if v is not None else "")
    out = re.sub(r"\{\{[^}]+\}\}", "", out)
    return out


# ================================ OpenAI SDK =================================
from openai import OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def get_openai() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)

# --- wrappers with token usage logging

def chat_completion(messages, temperature=0.2, max_tokens=700):
    client = get_openai()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    dur = int((time.perf_counter() - t0) * 1000)
    usage = getattr(resp, "usage", None)
    if usage:
        log.info("", extra={"event":"llm.usage","extra_payload":{
            "model": OPENAI_MODEL,
            "duration_ms": dur,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }})
        ls_log_llm_call(
            model=OPENAI_MODEL,
            messages=messages,
            response_text=(resp.choices[0].message.content or ""),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )
    return resp.choices[0].message.content or ""

def embed_texts(texts):
    with ls_span("embeddings.create", run_type="tool", inputs={"n_texts": len(texts), "model": OPENAI_EMBED_MODEL}):
        client = get_openai()
        t0 = time.perf_counter()
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
        dur = int((time.perf_counter() - t0) * 1000)
        usage = getattr(resp, "usage", None)
        if usage:
            log.info("", extra={"event":"embeddings.usage","extra_payload":{
                "model": OPENAI_EMBED_MODEL,
                "duration_ms": dur,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None)
            }})
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

# ================================ Paths / DB =================================
DB_PATH = os.getenv("DB_PATH", "./research.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./.chroma")

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources(
            id INTEGER PRIMARY KEY,
            url TEXT,
            canonical_url TEXT,
            title TEXT,
            authors TEXT,
            published_at TEXT,
            fetched_at TEXT,
            checksum_sha256 TEXT,
            source_type TEXT,
            raw_html BLOB,
            text TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_canon ON sources(canonical_url);
        CREATE INDEX IF NOT EXISTS idx_sources_checksum ON sources(checksum_sha256);
        """)

# ================================= Utils =====================================
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

UTM_PARAMS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_name","utm_id","utm_reader","utm_brand",
    "gclid","fbclid","mc_cid","mc_eid","igshid","si"
}

def canonicalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in UTM_PARAMS]
        q.sort()
        p2 = p._replace(query=urlencode(q, doseq=True), fragment="")
        netloc = p2.netloc.lower()
        scheme = (p2.scheme.lower() if p2.scheme else "https")
        return urlunparse((scheme, netloc, p2.path, p2.params, p2.query, ""))
    except Exception:
        return url

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

# ============================= Vector store (Chroma) ==========================
def get_vstore():
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(name="chunks", metadata={"hnsw:space": "cosine"})

# ============================= Source helpers ================================
def precheck_source(conn, url: str):
    canon = canonicalize_url(url)
    row = conn.execute(
        "SELECT id, COALESCE(LENGTH(text),0) FROM sources WHERE canonical_url=?",
        (canon,)
    ).fetchone()
    if row:
        return {"exists": True, "source_id": row[0], "has_text": row[1] > 0}
    return {"exists": False, "source_id": None, "has_text": False}

def parse_meta_from_html(html: str):
    if not html:
        return None, None, None
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string.strip() if soup.title and soup.title.string else None)
    date = None
    for prop in ["article:published_time", "og:published_time"]:
        m = soup.find("meta", {"property": prop})
        if m and m.get("content"):
            date = m.get("content")[:10]
            break
    authors = None
    au = soup.find("meta", {"name": "author"})
    if au and au.get("content"):
        authors = au.get("content")
    return title, authors, date

def _merge_meta(raw_html, hinted_title, hinted_authors, hinted_date):
    title, authors, published_at = parse_meta_from_html(raw_html)
    title = title or hinted_title
    authors = authors or hinted_authors
    published_at = published_at or hinted_date
    return title, authors, published_at

def store_source(conn, url: str, raw_html: str, text: str, source_type: str,
                 hinted_title=None, hinted_authors=None, hinted_date=None):
    """
    Insert or update a source row.
    Returns: (source_id, action) where action ∈ {"inserted","updated","reused"}
    """
    canon = canonicalize_url(url)
    checksum = sha256_text(text or "")

    row = conn.execute(
        "SELECT id, checksum_sha256 FROM sources WHERE canonical_url=?",
        (canon,)
    ).fetchone()

    title, authors, published_at = _merge_meta(raw_html, hinted_title, hinted_authors, hinted_date)

    if row:
        sid, old_sum = row
        if old_sum == checksum:
            return sid, "reused"
        conn.execute("""
            UPDATE sources SET
              title=?, authors=?, published_at=?, fetched_at=?, checksum_sha256=?,
              source_type=?, raw_html=?, text=?, url=COALESCE(?, url)
            WHERE id=?
        """, (title, authors, published_at, now_iso(), checksum, source_type, raw_html, text, url, sid))
        conn.commit()
        log.info("", extra={"event":"source.update","extra_payload":{"source_id": sid, "url": url}})
        return sid, "updated"

    cur = conn.execute("""
        INSERT INTO sources(url, canonical_url, title, authors, published_at, fetched_at,
                            checksum_sha256, source_type, raw_html, text)
        VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (url, canon, title, authors, published_at, now_iso(), checksum, source_type, raw_html, text))
    conn.commit()
    sid = cur.lastrowid
    log.info("", extra={"event":"source.insert","extra_payload":{"source_id": sid, "url": url}})
    return sid, "inserted"

# =============================== Search tools ================================
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

@log_span("tool.search.web")
def web_search_serpapi(query: str, num: int = 3):
    with ls_span("web_search_serpapi", run_type="tool", inputs={"query": query, "num": num}):
        if not SERPAPI_KEY:
            log.info("", extra={"event":"tool.search.web.skip","extra_payload":{"reason":"no SERPAPI_API_KEY"}})
            return []
        params = {"q": query, "api_key": SERPAPI_KEY, "num": num}
        t0 = time.perf_counter()
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        r.raise_for_status()
        dur = int((time.perf_counter() - t0) * 1000)
        data = r.json()
        out = []
        for item in (data.get("organic_results") or [])[:num]:
            out.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "source_type": "web"
            })
        log.info("", extra={"event":"tool.search.web.result","extra_payload":{"query": query, "num_results": len(out), "duration_ms": dur}})
        return out

@log_span("tool.search.arxiv")
def arxiv_search(query: str, max_results: int = 3):
    with ls_span("arxiv_search", run_type="tool", inputs={"query": query, "max_results": max_results}):
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        t0 = time.perf_counter()
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        dur = int((time.perf_counter() - t0) * 1000)
        feed = feedparser.parse(r.text)
        out = []
        for e in feed.entries:
            link = next((l.href for l in e.links if getattr(l, 'type', '').endswith('pdf')), e.link)
            out.append({
                "title": e.title,
                "url": link,
                "snippet": clean_text(getattr(e, "summary", "")),
                "authors": ", ".join(a.name for a in getattr(e, "authors", []) if hasattr(a, "name")),
                "published": getattr(e, "published", "")[:10],
                "source_type": "arxiv"
            })
        log.info("", extra={"event":"tool.search.arxiv.result","extra_payload":{"query": query, "num_results": len(out), "duration_ms": dur}})
        return out

def merge_unique_candidates(web_results, arx_results, max_fetch: int):
    seen = set()
    candidates = []
    for r in (web_results + arx_results):
        u = r.get("url")
        if not u:
            continue
        canon = canonicalize_url(u)
        if canon in seen:
            continue
        seen.add(canon)
        candidates.append(r)
        if len(candidates) >= max_fetch:
            break
    log.info("", extra={"event":"candidates.merge","extra_payload":{"kept": len(candidates), "max_fetch": max_fetch}})
    return candidates

# =============================== Fetch & parse ================================
class FetchError(Exception): ...
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((httpx.HTTPError, FetchError))
)
@log_span("fetch.url")
def fetch_url(url: str) -> tuple[str, str]:
    with ls_span("fetch_url", run_type="tool", inputs={"url": url}):
        headers = {"User-Agent": "SmartResearchAgent/1.0 (+https://example.local)"}
        t0 = time.perf_counter()
        with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            content = resp.content

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content))
                pages = []
                for p in reader.pages:
                    with suppress(Exception):
                        pages.append(p.extract_text() or "")
                text = clean_text("\n".join(pages))
                log.info("", extra={"event":"fetch.ok","extra_payload":{"url": url, "bytes": len(content), "type":"pdf"}})
                return ("", text)
            except Exception as e:
                raise FetchError(f"PDF parse failed: {e}")

        html = content.decode("utf-8", errors="ignore")
        extracted = trafilatura.extract(html, include_links=False, include_images=False) or ""
        if not extracted:
            soup = BeautifulSoup(html, "html.parser")
            text = clean_text(soup.get_text(separator=" "))
        else:
            text = clean_text(extracted)
        log.info("", extra={"event":"fetch.ok","extra_payload":{"url": url, "bytes": len(content), "type":"html"}})
        return (html, text)

# ================================ Indexing ===================================
def chunk_text(txt: str, target_tokens: int = 900):
    if not txt:
        return []
    chunk_size = target_tokens * 4
    chunks = []
    i = 0
    while i < len(txt):
        j = min(len(txt), i + chunk_size)
        k = txt.rfind(". ", i, j)
        k = j if k == -1 else k + 1
        part = txt[i:k].strip()
        if len(part) > 300:
            chunks.append(part)
        i = k
    return chunks

@log_span("index.add")
def ensure_index_and_embed(source_id: int, text: str, vstore):
    with ls_span("ensure_index_and_embed", run_type="retriever", inputs={"source_id": source_id}):
        head = text[:5000]
        if head:
            qemb = embed_texts([head])[0]
            try:
                res = vstore.query(
                    query_embeddings=[qemb.tolist()],
                    n_results=1,
                    include=["metadatas", "distances"]
                )
                if res.get("distances") and res["distances"][0]:
                    dist = float(res["distances"][0][0])
                    sim = 1 - dist
                    meta = (res.get("metadatas") or [[{}]])[0][0] or {}
                    existing_sid = int(meta.get("source_id", -1)) if meta.get("source_id") is not None else -1
                    if existing_sid != -1 and existing_sid != source_id and sim >= 0.97:
                        log.info("", extra={"event":"index.skip.dupe","extra_payload":{"source_id": source_id, "near": existing_sid, "sim": round(sim, 4)}})
                        return False
            except Exception:
                pass

    parts = chunk_text(text)
    if not parts:
        log.info("", extra={"event":"index.skip.empty","extra_payload":{"source_id": source_id}})
        return True

    embeds = embed_texts(parts)
    ids = [f"{source_id}:{ix}" for ix in range(len(parts))]
    metadatas = [{"source_id": source_id, "chunk_ix": ix} for ix in range(len(parts))]
    vstore.add(ids=ids, embeddings=[e.tolist() for e in embeds], documents=parts, metadatas=metadatas)
    log.info("", extra={"event":"index.add.detail","extra_payload":{"source_id": source_id, "chunks": len(parts)}})
    return True

# ================================ Retrieval ==================================
def local_matching_source_ids(prompt: str, vstore, *, min_sim: float = 0.4, pool_size: int = 60, debug: bool = True):
    qemb = embed_texts([prompt])[0]
    try:
        res = vstore.query(
            query_embeddings=[qemb.tolist()],
            n_results=pool_size,
            include=["metadatas", "distances"]
        )
    except Exception:
        log.info("", extra={"event":"local.match.err","extra_payload":{"reason":"vstore.query failed"}})
        if debug: status("ℹ️ Local match check failed — proceeding with web search.")
        return set()

    if not res or not res.get("metadatas") or not res["metadatas"]:
        if debug: status("ℹ️ No local data found.")
        return set()

    metas = res["metadatas"][0]
    dists = (res.get("distances") or [[]])[0]
    source_ids = set()

    for i, meta in enumerate(metas):
        dist = float(dists[i]) if i < len(dists) else 1.0
        sim = 1.0 - dist
        sid = int(meta.get("source_id")) if meta and meta.get("source_id") is not None else -1
        if sid != -1 and sim >= min_sim:
            source_ids.add(sid)

    log.info("", extra={"event":"local.match.result","extra_payload":{"pool": len(metas), "unique_sources": len(source_ids), "threshold": min_sim}})
    if debug:
        status(f"🔍 Local DB match: {len(source_ids)} sources ≥ similarity {min_sim}")
    return source_ids

@log_span("retrieve")
def retrieve_evidence(prompt: str, k: int = 12):
    with ls_span("retrieve_evidence", run_type="retriever", inputs={"k": k}):
        vstore = get_vstore()
        qemb = embed_texts([prompt])[0]
        try:
            res = vstore.query(
                query_embeddings=[qemb.tolist()],
                n_results=k,
                include=["metadatas", "documents", "distances"]
            )
        except Exception:
            return [], {}

        if not res or not res.get("documents") or not res["documents"]:
            return [], {}

        docs, ordinals, counter = [], {}, 1
        for i, doc in enumerate(res["documents"][0]):
            meta = res["metadatas"][0][i] if res.get("metadatas") and res["metadatas"] else {}
            sid = int(meta.get("source_id")) if meta and meta.get("source_id") is not None else -1
            if sid == -1:
                continue
            if sid not in ordinals:
                ordinals[sid] = counter
                counter += 1
            docs.append({"chunk_text": doc, "source_id": sid, "ordinal": ordinals[sid]})
        log.info("", extra={"event":"retrieve.done","extra_payload":{"k": k, "unique_sources": len(ordinals)}})
        return docs, ordinals

def get_sources_by_ids(conn, ids):
    if not ids:
        return {}
    q = "SELECT id, title, url, authors, published_at FROM sources WHERE id IN ({})".format(",".join("?"*len(ids)))
    cur = conn.execute(q, list(ids))
    out = {}
    for row in cur.fetchall():
        out[row[0]] = {"title": row[1], "url": row[2], "authors": row[3], "published_at": row[4]}
    return out

# =========================== Planning & Synthesis =============================
@log_span("tool.plan")
def plan_steps_with_llm(prompt: str) -> dict:
    with ls_span("plan_steps_with_llm", run_type="chain", inputs={"prompt": prompt}):
        system = load_prompt("planner_system.txt")
        user = render_template("planner_user.txt", prompt=prompt)

        out = chat_completion(
            [{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0, max_tokens=200
        )
        try:
            j = json.loads(out.strip().strip("```json").strip("```"))
        except Exception:
            j = {"web_queries":[f"{prompt} 2024"], "arxiv_queries":["2024 LLM alignment"]}
        j["web_queries"] = j.get("web_queries", [])[:2]
        j["arxiv_queries"] = j.get("arxiv_queries", [])[:2]
        log.info("", extra={"event":"tool.plan.debug","extra_payload":{"web_queries": j["web_queries"], "arxiv_queries": j["arxiv_queries"]}})
        return j


@log_span("llm.synthesize")
def synthesize_with_citations(prompt: str, evidence_rows, source_meta):
    with ls_span("synthesize_with_citations", run_type="chain"):
        snippets_by_ord, used_pairs = {}, set()
        for row in evidence_rows:
            ordn, sid = row["ordinal"], row["source_id"]
            key = (ordn, sid)
            if key not in used_pairs:
                snippets_by_ord.setdefault(ordn, {"sid": sid, "snippets": []})
                if len(snippets_by_ord[ordn]["snippets"]) < 2:
                    snippets_by_ord[ordn]["snippets"].append(row["chunk_text"][:800])
                used_pairs.add(key)

    lines = []
    for ordn in sorted(snippets_by_ord.keys()):
        sid = snippets_by_ord[ordn]["sid"]
        meta = source_meta.get(sid, {})
        title = meta.get("title") or "(untitled)"
        label = f"[{ordn}] {title}"
        for snip in snippets_by_ord[ordn]["snippets"]:
            lines.append(f"{label} — {snip}")
    context_block = "\n\n".join(lines)

    system = load_prompt("synthesis_system.txt")
    user = render_template("synthesis_user.txt", prompt=prompt, context_block=context_block)

    return chat_completion(
        [{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2, max_tokens=900
    )

def build_references(ord_map, src_meta):
    lines = ["\n\n## References"]
    for sid, num in sorted(ord_map.items(), key=lambda kvp: kvp[1]):
        meta = src_meta.get(sid, {})
        title = meta.get("title") or "(untitled)"
        url = meta.get("url") or ""
        authors = meta.get("authors") or ""
        year = (meta.get("published_at") or "")[:4]
        lines.append(f"[{num}] {title} — {authors} ({year}). {url}")
    return "\n".join(lines)

# =============================== Orchestrator ================================
def handle_prompt(prompt: str, max_fetch: int = 10, k_retrieve: int = 10,
                  min_local_sources: int = 4,  # default 4, tested 3 too
                  precheck_max_chunks: int = 70):
    t0 = time.perf_counter()
    log.info("", extra={"event": "run.handle_prompt", "extra_payload": {
        "prompt": prompt, "max_fetch": max_fetch, "k_retrieve": k_retrieve, "min_local_sources": min_local_sources
    }})

    with db() as conn:
        vstore = get_vstore()

        # ---- Local coverage gate ----
        status("🔍 Checking local database for similar sources...")
        local_sids = local_matching_source_ids(
            prompt,
            vstore,
            min_sim=0.5,
            pool_size=60,
            debug=True
        )

        if len(local_sids) >= min_local_sources:
            status(f"✅ Found enough local sources ({len(local_sids)}) — skipping web search.")
            log.info("", extra={"event": "local.gate.pass", "extra_payload": {"found": len(local_sids), "needed": min_local_sources}})
            status("🔎 Retrieving best evidence from vector DB...")
            evidence_rows, ord_map = retrieve_evidence(prompt, k=k_retrieve)
            src_meta = get_sources_by_ids(conn, list(ord_map.keys()))
            status("✍️ Generating final answer with citations...")
            md = synthesize_with_citations(prompt, evidence_rows, src_meta)
            result = md + build_references(ord_map, src_meta)
            dur = int((time.perf_counter() - t0) * 1000)
            log.info("", extra={"event": "run.handle_prompt.done", "extra_payload": {"duration_ms": dur}})
            status("✅ Done!\n")
            return result

        status(f"⚠️ Only {len(local_sids)} local sources found — running web & arXiv searches.")
        log.info("", extra={"event": "local.gate.fail", "extra_payload": {"found": len(local_sids), "needed": min_local_sources}})

        # ---- Plan searches ----
        status("📝 Asking LLM to plan searches...")
        plan = plan_steps_with_llm(prompt)

        # ---- Search ----
        status("🌐 Searching the web...")
        web_results = []
        for q in plan["web_queries"]:
            status(f"   • Web: {q}")
            web_results += web_search_serpapi(q, num=6)

        status("📚 Searching arXiv...")
        arx_results = []
        for q in plan["arxiv_queries"]:
            status(f"   • arXiv: {q}")
            arx_results += arxiv_search(q, max_results=6)

        # ---- Deduplicate/trim ----
        status("🧹 Merging and deduplicating search results...")
        candidates = merge_unique_candidates(web_results, arx_results, max_fetch=max_fetch)

        # ---- Ingest ----
        status("📥 Fetching and indexing new sources...")
        for r in candidates:
            url = r["url"]
            title = r.get("title") or url
            status(f"   • {title}")

            pre = precheck_source(conn, url)

            # Case 1: already have in SQLite
            if pre["exists"] and pre["has_text"]:
                sid = pre["source_id"]
                row = conn.execute("SELECT text FROM sources WHERE id=?", (sid,)).fetchone()
                text = row[0] if row and row[0] else ""

                # Check if it's already in Chroma
                res = vstore.get(where={"source_id": sid})
                if not res["ids"]:  # Not indexed yet
                    ensure_index_and_embed(sid, text, vstore)
                    status(f"      ✓ Cached — added to vector DB")
                    log.info("", extra={"event": "source.reuse.indexed", "extra_payload": {"source_id": sid, "url": url}})
                else:
                    status(f"      ✓ Cached — already in vector DB")
                    log.info("", extra={"event": "source.reuse", "extra_payload": {"source_id": sid, "url": url}})
                continue

            # Case 2: needs fetching
            try:
                raw_html, text = fetch_url(url)
            except Exception as e:
                status("      ✗ Fetch failed")
                log.error("", extra={"event": "fetch.err", "extra_payload": {"url": url, "error": str(e)}})
                continue

            if not text or len(text) < 500:
                status("      ⚪ Skipped (too short after fetch)")
                log.info("", extra={"event": "source.skip.short", "extra_payload": {"url": url, "len": len(text or '')}})
                continue

            sid, action = store_source(
                conn, url, raw_html, text, r.get("source_type", "web"),
                hinted_title=r.get("title"),
                hinted_authors=r.get("authors"),
                hinted_date=r.get("published"),
            )
            ensure_index_and_embed(sid, text, vstore)
            status(f"      💾 Saved ({action}) and indexed")
            log.info("", extra={"event": "source.store", "extra_payload": {"action": action, "source_id": sid, "url": url}})

        # ---- Retrieve → Synthesize ----
        status("🔎 Retrieving best evidence from vector DB...")
        evidence_rows, ord_map = retrieve_evidence(prompt, k=k_retrieve)
        src_meta = get_sources_by_ids(conn, list(ord_map.keys()))
        status("✍️ Generating final answer with citations...")
        md = synthesize_with_citations(prompt, evidence_rows, src_meta)
        result = md + build_references(ord_map, src_meta)

        dur = int((time.perf_counter() - t0) * 1000)
        log.info("", extra={"event": "run.handle_prompt.done", "extra_payload": {"duration_ms": dur}})
        status("✅ Done!\n")
        return result


# =================================== CLI =====================================
def main():
    run_id = new_run_id()
    log.info("", extra={"event":"run.start","extra_payload":{"run_id": run_id}})

    parser = argparse.ArgumentParser(description="Mini Smart Research CLI")
    parser.add_argument("--prompt", help="Research question (skips interactive mode)")
    parser.add_argument("--max-fetch", type=int, default=10, help="Max sources to fetch")
    parser.add_argument("--k-retrieve", type=int, default=10, help="Number of chunks to retrieve from vector DB")
    parser.add_argument("--min-local-sources", type=int, default= 3, help="Number of resouces needed to skip web-search")
    args = parser.parse_args()

    init_db()

    def ask_and_answer(q: str):
        root = ls_start_root_run(
            name="handle_prompt",
            inputs={"prompt": q},
            tags=["cli", "smart-research-agent"],
            metadata={}
        )
        try:
            status("🤖 Working: fetching sources, indexing, and writing…")
            md = handle_prompt(q, max_fetch=args.max_fetch, k_retrieve=args.k_retrieve,
                            min_local_sources=args.min_local_sources)
            print(md)
            print("\n" + "-"*80 + "\n")
            ls_end_root_run(outputs={"answer_markdown": md})
        except Exception as e:
            ls_end_root_run(error=str(e))
            raise


    try:
        if args.prompt:
            ask_and_answer(args.prompt)
        else:
            print("Hello! I am a Smart Research Agent. Type your question and press Enter.")
            print("Type 'exit' to quit.\n")
            while True:
                try:
                    q = input("What do you want to research? ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nbye!")
                    break
                if not q:
                    continue
                if q.lower() in {"exit", "quit"}:
                    print("bye!")
                    break
                ask_and_answer(q)
        log.info("", extra={"event":"run.end","extra_payload":{"status":"ok"}})
    except Exception as e:
        log.error("", extra={"event":"run.end","extra_payload":{"status":"err","error":str(e)}})
        raise

if __name__ == "__main__":
    main()
