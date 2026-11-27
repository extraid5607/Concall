import io
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader
from deep_translator import GoogleTranslator
from gtts import gTTS
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

DATA_FILE = "published_concalls.json"
META_FILE = "sync_meta.json"

DEFAULT_GLOSSARY = [
    "EBITDA", "PAT", "YoY", "QoQ", "Revenue", "Margin", "Gross Margin", "Operating Margin",
    "Net Profit", "Capex", "Opex", "Guidance", "Order Book", "EPS", "ROE", "ROCE",
    "Free Cash Flow", "Working Capital", "Cash Flow", "Balance Sheet", "Management",
    "Analyst", "Moderator", "Quarter", "Basis Points", "bps", "CAGR", "Debt", "Equity",
    "Dividend", "Valuation", "Market Share", "Volume Growth", "Pipeline", "Headwinds",
    "Tailwinds", "Demand", "Supply Chain", "NPA", "GNPA", "NNPA", "NIM", "AUM", "ARPU",
    "FY25", "FY26", "FY27", "Q1", "Q2", "Q3", "Q4", "Cr", "Crore", "Lakh",
]

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")


def load_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def translate_text_with_glossary(text: str, glossary: List[str]) -> str:
    clean = text.strip()
    if not clean:
        return ""
    sorted_terms = sorted({t.strip() for t in (glossary or DEFAULT_GLOSSARY) if t.strip()}, key=len, reverse=True)
    token_map = []
    protected = clean
    for term in sorted_terms:
        pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
        def repl(m):
            tok = f"ZXQ{len(token_map)}QZ"
            token_map.append((tok, m.group(0)))
            return tok
        protected = pattern.sub(repl, protected)

    translator = GoogleTranslator(source="en", target="pa")
    chunks = []
    cur = ""
    for line in protected.split("\n"):
        if len(cur) + len(line) + 1 > 4000 and cur:
            chunks.append(cur)
            cur = line
        else:
            cur = (cur + "\n" + line) if cur else line
    if cur:
        chunks.append(cur)

    translated = "\n".join([translator.translate(c) or c for c in chunks])
    for tok, orig in token_map:
        translated = re.sub(re.escape(tok), orig, translated, flags=re.IGNORECASE)
    return translated


def split_paragraphs_clean(raw_text: str) -> List[str]:
    lines = [ln.strip() for ln in raw_text.replace("\r\n", "\n").split("\n")]
    paras = []
    buf = ""
    for ln in lines:
        if not ln:
            if buf:
                paras.append(buf.strip())
                buf = ""
            continue
        if re.match(r"^(page\s+\d+(\s+of\s+\d+)?|\d{1,3})$", ln, re.I):
            continue
        starts_speaker = bool(re.match(r"^[A-Z][a-zA-Z.\s]{1,35}:\s*", ln))
        prev_ended = bool(re.search(r'[.?!]["\')\]]?$', buf))
        if buf and (starts_speaker or (prev_ended and len(buf) > 220)):
            paras.append(buf.strip())
            buf = ln
        else:
            buf = f"{buf} {ln}".strip() if buf else ln
    if buf:
        paras.append(buf.strip())
    return [p for p in paras if len(p) >= 12]


async def run_bse_nse_auto_sync(limit: int = 5) -> Dict[str, Any]:
    """Pulls latest official BSE & NSE Earnings Call Transcript PDFs, extracts paragraphs,
    translates into Punjabi keeping English financial terms, and saves to DATA_FILE."""
    existing = load_json_file(DATA_FILE, [])
    existing_ids = {x.get("id") for x in existing}
    added = 0

    bse_headers = {**BROWSER_HEADERS, "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"}
    candidates = []

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # 1. Fetch BSE Earnings Call Transcripts
        try:
            r = await client.get(
                "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
                params={
                    "pageno": "1",
                    "strCat": "Company Update",
                    "strSearch": "P",
                    "strType": "C",
                    "subcategory": "Earnings Call Transcript",
                },
                headers=bse_headers,
            )
            if r.status_code == 200:
                for row in r.json().get("Table", [])[:25]:
                    att = row.get("ATTACHMENTNAME", "")
                    if att and att.lower().endswith(".pdf"):
                        scrip = str(row.get("SCRIP_CD", "BSE")).strip()
                        comp = (row.get("SLONGNAME") or scrip).strip()
                        dt_str = (row.get("NEWS_DT") or "").split("T")[0]
                        candidates.append({
                            "id": f"{scrip}_{dt_str}",
                            "symbol": scrip,
                            "companyName": comp,
                            "quarter": dt_str or "Latest",
                            "sourcePlatform": "BSE Auto-Sync",
                            "pdf_url": f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}",
                            "headers": bse_headers,
                        })
        except Exception:
            pass

        # 2. Fetch NSE Corporate Announcements filtered for transcripts
        try:
            nse_headers = {**BROWSER_HEADERS, "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements"}
            await client.get("https://www.nseindia.com", headers=nse_headers)
            r = await client.get(
                "https://www.nseindia.com/api/corporate-announcements",
                params={"index": "equities"},
                headers=nse_headers,
            )
            if r.status_code == 200:
                for row in r.json()[:60]:
                    desc = f"{row.get('desc', '')} {row.get('attchmntText', '')}"
                    pdf_url = row.get("attchmntFile", "")
                    if pdf_url.lower().endswith(".pdf") and re.search(r"transcript|concall|earnings\s*call", desc, re.I):
                        sym = (row.get("symbol") or "NSE").strip().upper()
                        comp = (row.get("sm_name") or sym).strip()
                        dt_str = (row.get("an_dt") or "").split(" ")[0]
                        candidates.append({
                            "id": f"{sym}_{dt_str}",
                            "symbol": sym,
                            "companyName": comp,
                            "quarter": dt_str or "Latest",
                            "sourcePlatform": "NSE Auto-Sync",
                            "pdf_url": pdf_url,
                            "headers": nse_headers,
                        })
        except Exception:
            pass

        # Process up to `limit` new filings
        for item in candidates:
            if added >= limit:
                break
            if item["id"] in existing_ids:
                continue
            try:
                pdf_res = await client.get(item["pdf_url"], headers=item["headers"])
                if pdf_res.status_code != 200:
                    continue
                reader = PdfReader(io.BytesIO(pdf_res.content))
                raw_text = "\n\n".join([(p.extract_text() or "").strip() for p in reader.pages])
                paras = split_paragraphs_clean(raw_text)
                if not paras:
                    continue

                translated_pairs = []
                for p in paras:
                    m = re.match(r"^([A-Z][a-zA-Z.\s]{1,35}):\s*(.+)$", p)
                    speaker = m.group(1).strip() if m else ""
                    is_qa = ("?" in p) or bool(re.search(r"analyst|question|moderator", speaker, re.I))
                    pa = translate_text_with_glossary(p, DEFAULT_GLOSSARY)
                    translated_pairs.append({"speaker": speaker, "isQA": is_qa, "en": p, "pa": pa})

                doc = {
                    "id": item["id"],
                    "symbol": item["symbol"],
                    "companyName": item["companyName"],
                    "quarter": item["quarter"],
                    "sourcePlatform": item["sourcePlatform"],
                    "sourceUrl": item["pdf_url"],
                    "engine": "Auto-Sync + Punjabi Glossary",
                    "kpis": [],
                    "summaryBullets": [x["pa"] for x in translated_pairs[:4]],
                    "paragraphs": translated_pairs,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
                existing.insert(0, doc)
                existing_ids.add(item["id"])
                added += 1
            except Exception:
                continue

    save_json_file(DATA_FILE, existing)
    meta = {
        "last_sync_at": datetime.now(timezone.utc).isoformat(),
        "last_added": added,
        "total_published": len(existing),
        "schedule": "Daily 6:00 PM IST (Asia/Kolkata)",
    }
    save_json_file(META_FILE, meta)
    return {"status": "synced", **meta}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schedule automatic daily sync at 6:00 PM IST (18:00 Asia/Kolkata)
    scheduler.add_job(
        run_bse_nse_auto_sync,
        trigger=CronTrigger(hour=18, minute=0, timezone="Asia/Kolkata"),
        id="daily_6pm_ist_sync",
        replace_existing=True,
        kwargs={"limit": 5},
    )
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="Punjab AI Concall Relay", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractUrlRequest(BaseModel):
    url: str


class TranslateRequest(BaseModel):
    text: str
    glossary: List[str] = []


@app.get("/health")
async def health():
    return {"status": "ok", "service": "punjab-ai-concall-relay-v4", "schedule": "Daily 6:00 PM IST"}


@app.get("/api/sync-status")
async def sync_status():
    meta = load_json_file(META_FILE, {})
    items = load_json_file(DATA_FILE, [])
    return {
        "schedule": "Daily 6:00 PM IST",
        "last_sync_at": meta.get("last_sync_at"),
        "last_added": meta.get("last_added", 0),
        "total_published": len(items),
    }


@app.get("/api/published")
async def get_published():
    items = load_json_file(DATA_FILE, [])
    return {"count": len(items), "items": items}


@app.post("/api/publish")
async def publish_concall(doc: Dict[str, Any]):
    items = load_json_file(DATA_FILE, [])
    doc_id = doc.get("id")
    if not doc_id:
        raise HTTPException(status_code=400, detail="Missing doc id")
    items = [x for x in items if x.get("id") != doc_id]
    items.insert(0, doc)
    save_json_file(DATA_FILE, items)
    return {"status": "published", "count": len(items)}


@app.post("/api/auto-sync")
@app.get("/api/auto-sync")
async def trigger_auto_sync(limit: int = Query(default=5, ge=1, le=15)):
    """Can be triggered manually by Admin anytime OR automatically by Render Cron at 6:00 PM IST."""
    return await run_bse_nse_auto_sync(limit=limit)


@app.get("/api/fetch-platform")
async def fetch_platform(
    platform: str = Query(..., description="NSE, BSE, SCREENER, STOCKSCAN, ALPHASTREET"),
    query: str = Query(..., description="Company symbol or name"),
):
    plat = platform.strip().upper()
    q = query.strip().upper()
    items = []

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        if plat == "NSE":
            try:
                hdrs = {**BROWSER_HEADERS, "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements"}
                await client.get("https://www.nseindia.com", headers=hdrs)
                r = await client.get(
                    "https://www.nseindia.com/api/corporate-announcements",
                    params={"index": "equities", "symbol": q},
                    headers=hdrs,
                )
                if r.status_code == 200:
                    for row in r.json()[:80]:
                        desc = f"{row.get('desc', '')} {row.get('attchmntText', '')}"
                        if re.search(r"transcript|concall|earnings\s*call|analyst|investor\s*call", desc, re.I):
                            items.append({
                                "symbol": row.get("symbol", q),
                                "company": row.get("sm_name", q),
                                "subject": row.get("attchmntText") or row.get("desc") or "Earnings Call Transcript",
                                "date": row.get("an_dt", ""),
                                "pdf_url": row.get("attchmntFile", ""),
                            })
            except Exception:
                pass

        elif plat == "BSE":
            try:
                hdrs = {**BROWSER_HEADERS, "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"}
                r = await client.get(
                    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
                    params={
                        "pageno": "1",
                        "strCat": "Company Update",
                        "strSearch": "P",
                        "strType": "C",
                        "subcategory": "Earnings Call Transcript",
                    },
                    headers=hdrs,
                )
                if r.status_code == 200:
                    for row in r.json().get("Table", [])[:60]:
                        comp = row.get("SLONGNAME", "")
                        if q not in comp.upper() and q not in str(row.get("SCRIP_CD", "")):
                            continue
                        att = row.get("ATTACHMENTNAME", "")
                        items.append({
                            "symbol": str(row.get("SCRIP_CD", q)),
                            "company": comp,
                            "subject": row.get("NEWSSUB", "Earnings Call Transcript"),
                            "date": row.get("NEWS_DT", ""),
                            "pdf_url": f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}" if att else "",
                        })
            except Exception:
                pass

        elif plat == "SCREENER":
            try:
                r = await client.get(f"https://www.screener.in/company/{q}/consolidated/", headers=BROWSER_HEADERS)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    for a in soup.find_all("a", href=True):
                        txt = a.get_text(" ", strip=True)
                        href = a["href"]
                        if re.search(r"transcript", txt, re.I) or href.lower().endswith(".pdf"):
                            if re.search(r"bseindia|nseindia|concall|transcript", href + " " + txt, re.I):
                                items.append({
                                    "symbol": q,
                                    "company": q,
                                    "subject": txt or "Screener Concall Transcript",
                                    "date": "Screener.in",
                                    "pdf_url": href if href.startswith("http") else f"https://www.screener.in{href}",
                                })
            except Exception:
                pass

        elif plat == "ALPHASTREET":
            try:
                r = await client.get(f"https://news.alphastreet.com/?s={q}+transcript", headers=BROWSER_HEADERS)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    for a in soup.select("h2 a, h3 a")[:15]:
                        href = a.get("href", "")
                        txt = a.get_text(" ", strip=True)
                        if href.startswith("https://") and re.search(r"transcript|earnings", txt, re.I):
                            items.append({
                                "symbol": q,
                                "company": q,
                                "subject": txt,
                                "date": "AlphaStreet",
                                "pdf_url": href,
                            })
            except Exception:
                pass

        elif plat == "STOCKSCAN":
            try:
                r = await client.get(f"https://stockscan.io/stocks/{q}/earnings-call-transcripts", headers=BROWSER_HEADERS)
                if r.status_code == 200:
                    items.append({
                        "symbol": q,
                        "company": q,
                        "subject": f"{q} StockScan Earnings Call Transcripts",
                        "date": "StockScan",
                        "pdf_url": str(r.url),
                    })
            except Exception:
                pass

    return {"platform": plat, "count": len(items), "items": items[:30]}


@app.post("/api/extract-url")
async def extract_url(req: ExtractUrlRequest):
    if not req.url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Only HTTPS URLs are supported.")

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        r = await client.get(req.url, headers=BROWSER_HEADERS)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Upstream returned HTTP {r.status_code}")

        ctype = (r.headers.get("content-type") or "").lower()
        if "pdf" in ctype or req.url.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(r.content))
            raw = "\n\n".join([(p.extract_text() or "").strip() for p in reader.pages])
            paras = split_paragraphs_clean(raw)
            return {"type": "pdf", "pages": len(reader.pages), "text": "\n\n".join(paras)}
        else:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            paras = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "div", "li"]) if len(p.get_text(strip=True)) > 40]
            seen = set()
            clean_paras = []
            for p in paras:
                if p not in seen:
                    seen.add(p)
                    clean_paras.append(p)
            return {"type": "html", "text": "\n\n".join(clean_paras)}


@app.post("/api/translate")
async def translate_punjabi(req: TranslateRequest):
    return {"punjabi": translate_text_with_glossary(req.text, req.glossary)}


@app.get("/api/tts")
async def punjabi_tts(text: str = Query(..., min_length=1, max_length=600)):
    buf = io.BytesIO()
    tts = gTTS(text=text, lang="pa", slow=False)
    tts.write_to_fp(buf)
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")
