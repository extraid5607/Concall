import io
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader
from deep_translator import GoogleTranslator
from gtts import gTTS
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "published_concalls.json")
META_FILE = os.path.join(BASE_DIR, "sync_meta.json")

DEFAULT_GLOSSARY = [
    "EBITDA", "PAT", "YoY", "QoQ", "Revenue", "Margin", "Gross Margin", "Operating Margin",
    "Net Profit", "Capex", "Opex", "Guidance", "Order Book", "EPS", "ROE", "ROCE",
    "Free Cash Flow", "Working Capital", "Cash Flow", "Balance Sheet", "Management",
    "Analyst", "Moderator", "Quarter", "Basis Points", "bps", "CAGR", "Debt", "Equity",
    "Dividend", "Valuation", "Market Share", "Volume Growth", "Pipeline", "Headwinds",
    "Tailwinds", "Demand", "Supply Chain", "NPA", "GNPA", "NNPA", "NIM", "AUM", "ARPU",
    "FY25", "FY26", "FY27", "Q1", "Q2", "Q3", "Q4", "Cr", "Crore", "Lakh",
]

SCREENER_AUTO_SYMBOLS = [
    ("RELIANCE", "Reliance Industries Ltd"),
    ("TCS", "Tata Consultancy Services Ltd"),
    ("HDFCBANK", "HDFC Bank Ltd"),
    ("INFY", "Infosys Ltd"),
    ("ICICIBANK", "ICICI Bank Ltd"),
    ("BAJFINANCE", "Bajaj Finance Ltd"),
    ("TATAMOTORS", "Tata Motors Ltd"),
    ("ITC", "ITC Ltd"),
    ("BHARTIARTL", "Bharti Airtel Ltd"),
    ("SBIN", "State Bank of India"),
    ("LT", "Larsen & Toubro Ltd"),
    ("SUNPHARMA", "Sun Pharmaceutical Industries Ltd"),
    ("MARUTI", "Maruti Suzuki India Ltd"),
    ("AXISBANK", "Axis Bank Ltd"),
    ("WIPRO", "Wipro Ltd"),
    ("HCLTECH", "HCL Technologies Ltd"),
    ("TITAN", "Titan Company Ltd"),
    ("ASIANPAINT", "Asian Paints Ltd"),
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


def protect_glossary(text: str, glossary: List[str]):
    sorted_terms = sorted({t.strip() for t in (glossary or DEFAULT_GLOSSARY) if t.strip()}, key=len, reverse=True)
    token_map = []
    protected = text
    for term in sorted_terms:
        pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
        def repl(m):
            tok = f"ZXQ{len(token_map)}QZ"
            token_map.append((tok, m.group(0)))
            return tok
        protected = pattern.sub(repl, protected)
    return protected, token_map


def restore_glossary(text: str, token_map) -> str:
    out = text
    for tok, orig in token_map:
        out = re.sub(re.escape(tok), orig, out, flags=re.IGNORECASE)
    return out


def translate_text_with_glossary(text: str, glossary: List[str]) -> str:
    clean = text.strip()
    if not clean:
        return ""
    protected, token_map = protect_glossary(clean, glossary)
    translator = GoogleTranslator(source="en", target="pa")
    chunks = []
    cur = ""
    for line in protected.split("\n"):
        if len(cur) + len(line) + 1 > 3800 and cur:
            chunks.append(cur)
            cur = line
        else:
            cur = (cur + "\n" + line) if cur else line
    if cur:
        chunks.append(cur)
    translated = "\n".join([translator.translate(c) or c for c in chunks])
    return restore_glossary(translated, token_map)


def translate_paragraphs_batched(paras: List[str], glossary: List[str]) -> List[str]:
    """Fast batched translation preserving exact 1-to-1 paragraph alignment (10x faster on Render)."""
    translator = GoogleTranslator(source="en", target="pa")
    results = []
    batch = []
    batch_len = 0

    def flush(cur_batch: List[str]):
        if not cur_batch:
            return
        sep = "\n||||\n"
        joined = sep.join(cur_batch)
        protected, token_map = protect_glossary(joined, glossary)
        try:
            tr = translator.translate(protected) or protected
            tr = restore_glossary(tr, token_map)
            parts = [p.strip() for p in re.split(r"\s*\|\|\|\|\s*", tr)]
            if len(parts) == len(cur_batch):
                results.extend(parts)
                return
        except Exception:
            pass
        # Fallback: translate each paragraph individually if separator count drifted
        for p in cur_batch:
            try:
                results.append(translate_text_with_glossary(p, glossary))
            except Exception:
                results.append(p)

    for p in paras:
        if batch and (batch_len + len(p) + 8) > 3400:
            flush(batch)
            batch = []
            batch_len = 0
        batch.append(p)
        batch_len += len(p) + 8
    flush(batch)
    return results


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
    return [p for p in paras if len(p) >= 15]


async def run_bse_nse_auto_sync(limit: int = 5) -> Dict[str, Any]:
    existing = load_json_file(DATA_FILE, [])
    existing_ids = {x.get("id") for x in existing}
    added = 0
    bse_found = 0
    nse_found = 0
    screener_found = 0
    errors: List[str] = []

    bse_headers = {**BROWSER_HEADERS, "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"}
    candidates = []

    today = datetime.now(timezone.utc)
    from_date = (today - timedelta(days=20)).strftime("%Y%m%d")
    to_date = today.strftime("%Y%m%d")

    async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
        # 1. BSE India Corporate Announcements (last 20 days, filtered for Transcripts)
        try:
            r = await client.get(
                "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
                params={
                    "pageno": "1",
                    "strCat": "Company Update",
                    "strPrevDate": from_date,
                    "strScrip": "",
                    "strSearch": "P",
                    "strToDate": to_date,
                    "strType": "C",
                    "subcategory": "",
                },
                headers=bse_headers,
            )
            if r.status_code == 200:
                rows = r.json().get("Table", [])
                for row in rows:
                    subj = f"{row.get('NEWSSUB', '')} {row.get('HEADLINE', '')}"
                    att = row.get("ATTACHMENTNAME", "")
                    if att and att.lower().endswith(".pdf") and re.search(r"transcript|concall|earnings\s*call|conference\s*call", subj, re.I):
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
                        bse_found += 1
        except Exception as e:
            errors.append(f"BSE: {str(e)[:80]}")

        # 2. NSE India Corporate Announcements
        try:
            nse_headers = {**BROWSER_HEADERS, "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements"}
            await client.get("https://www.nseindia.com", headers=nse_headers)
            r = await client.get(
                "https://www.nseindia.com/api/corporate-announcements",
                params={"index": "equities"},
                headers=nse_headers,
            )
            if r.status_code == 200:
                for row in r.json()[:80]:
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
                        nse_found += 1
        except Exception as e:
            errors.append(f"NSE: {str(e)[:80]}")

        # 3. Screener.in Auto-Sync (Cloud-friendly fallback for top Nifty companies — pulls official BSE/NSE PDFs!)
        if len(candidates) < limit:
            for sym, comp_name in SCREENER_AUTO_SYMBOLS:
                if screener_found >= limit + 2:
                    break
                try:
                    sr = await client.get(f"https://www.screener.in/company/{sym}/consolidated/", headers=BROWSER_HEADERS)
                    if sr.status_code != 200:
                        continue
                    soup = BeautifulSoup(sr.text, "html.parser")
                    concalls_div = soup.find("div", class_=re.compile(r"concalls", re.I)) or soup
                    for a in concalls_div.find_all("a", href=True):
                        txt = a.get_text(" ", strip=True)
                        href = a["href"]
                        if re.search(r"transcript", txt, re.I) and href.startswith("http"):
                            # Try to read quarter/date from parent list item
                            parent_txt = a.parent.get_text(" ", strip=True) if a.parent else ""
                            q_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+20\d\d", parent_txt, re.I)
                            q_label = q_match.group(0) if q_match else "Latest"
                            doc_id = f"{sym}_{q_label.replace(' ', '_')}"
                            if doc_id not in existing_ids:
                                candidates.append({
                                    "id": doc_id,
                                    "symbol": sym,
                                    "companyName": comp_name,
                                    "quarter": q_label,
                                    "sourcePlatform": "Screener Auto-Sync",
                                    "pdf_url": href,
                                    "headers": bse_headers if "bseindia.com" in href else BROWSER_HEADERS,
                                })
                                screener_found += 1
                            break
                except Exception:
                    continue

        # Download, extract, and translate up to `limit` new filings
        for item in candidates:
            if added >= limit:
                break
            if item["id"] in existing_ids:
                continue
            try:
                pdf_res = await client.get(item["pdf_url"], headers=item["headers"])
                if pdf_res.status_code != 200 or len(pdf_res.content) < 500:
                    continue
                reader = PdfReader(io.BytesIO(pdf_res.content))
                # Extract up to 25 pages cleanly
                raw_text = "\n\n".join([(p.extract_text() or "").strip() for p in reader.pages[:25]])
                paras = split_paragraphs_clean(raw_text)
                if len(paras) < 3:
                    continue

                # Batched Punjabi translation (fast!)
                pa_list = translate_paragraphs_batched(paras, DEFAULT_GLOSSARY)
                translated_pairs = []
                for en, pa in zip(paras, pa_list):
                    m = re.match(r"^([A-Z][a-zA-Z.\s]{1,35}):\s*(.+)$", en)
                    speaker = m.group(1).strip() if m else ""
                    is_qa = ("?" in en) or bool(re.search(r"analyst|question|moderator", speaker, re.I))
                    translated_pairs.append({"speaker": speaker, "isQA": is_qa, "en": en, "pa": pa})

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
            except Exception as e:
                errors.append(f"{item['symbol']}: {str(e)[:60]}")
                continue

    save_json_file(DATA_FILE, existing)
    meta = {
        "last_sync_at": datetime.now(timezone.utc).isoformat(),
        "last_added": added,
        "total_published": len(existing),
        "sources": {"bse": bse_found, "nse": nse_found, "screener": screener_found},
        "errors": errors[:5],
        "schedule": "Daily 6:00 PM IST (Asia/Kolkata)",
    }
    save_json_file(META_FILE, meta)
    return {"status": "synced", "added": added, **meta}


@asynccontextmanager
async def lifespan(app: FastAPI):
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


app = FastAPI(title="Concall Punjabi — Punjab AI (All-in-One)", version="5.1.0", lifespan=lifespan)

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


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"), media_type="text/html")


@app.get("/manifest.json")
async def serve_manifest():
    return FileResponse(os.path.join(BASE_DIR, "manifest.json"), media_type="application/manifest+json")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "concall-punjabi-all-in-one", "version": "5.1.0", "schedule": "Daily 6:00 PM IST"}


@app.get("/api/sync-status")
async def sync_status():
    meta = load_json_file(META_FILE, {})
    items = load_json_file(DATA_FILE, [])
    return {
        "schedule": "Daily 6:00 PM IST",
        "last_sync_at": meta.get("last_sync_at"),
        "last_added": meta.get("last_added", 0),
        "sources": meta.get("sources", {}),
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
async def trigger_auto_sync(limit: int = Query(default=3, ge=1, le=15)):
    return await run_bse_nse_auto_sync(limit=limit)


@app.get("/api/fetch-platform")
async def fetch_platform(
    platform: str = Query(..., description="NSE, BSE, SCREENER, STOCKSCAN, ALPHASTREET"),
    query: str = Query(..., description="Company symbol or name"),
):
    plat = platform.strip().upper()
    q = query.strip().upper()
    items = []
    today = datetime.now(timezone.utc)
    from_date = (today - timedelta(days=90)).strftime("%Y%m%d")
    to_date = today.strftime("%Y%m%d")

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
                        "strPrevDate": from_date,
                        "strScrip": "",
                        "strSearch": "P",
                        "strToDate": to_date,
                        "strType": "C",
                        "subcategory": "",
                    },
                    headers=hdrs,
                )
                if r.status_code == 200:
                    for row in r.json().get("Table", []):
                        comp = row.get("SLONGNAME", "")
                        subj = f"{row.get('NEWSSUB', '')} {row.get('HEADLINE', '')}"
                        if q not in comp.upper() and q not in str(row.get("SCRIP_CD", "")):
                            continue
                        if not re.search(r"transcript|concall|earnings\s*call", subj, re.I):
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
                    concalls_div = soup.find("div", class_=re.compile(r"concalls", re.I)) or soup
                    for a in concalls_div.find_all("a", href=True):
                        txt = a.get_text(" ", strip=True)
                        href = a["href"]
                        if re.search(r"transcript", txt, re.I) and href.startswith("http"):
                            parent_txt = a.parent.get_text(" ", strip=True) if a.parent else ""
                            items.append({
                                "symbol": q,
                                "company": q,
                                "subject": parent_txt[:90] or txt,
                                "date": "Screener.in",
                                "pdf_url": href,
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
        hdrs = {**BROWSER_HEADERS, "Referer": "https://www.bseindia.com/"} if "bseindia.com" in req.url else BROWSER_HEADERS
        r = await client.get(req.url, headers=hdrs)
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
