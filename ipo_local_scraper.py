import requests
import json
import os
import re
import shutil
from bs4 import BeautifulSoup
import html
from pathlib import Path
import time
import zipfile
import io
from typing import Optional
from datetime import datetime
from decimal import Decimal

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# Markdown conversion imports
from convert_to_markdown import (
    metadata_to_markdown,
    extract_urls_metadata,
    convert_pdf_to_markdown,
    slugify,
)


# --- Configuration ---
BASE_OUTPUT_DIR = "./ipo_data"
MD_OUTPUT_DIR = "./ipo_data_md"
LIST_URL = "https://webnodejs.chittorgarh.com/cloud/ipo/list-read"
HTML_BASE_URL = "https://www.chittorgarh.com/ipo/{folder}/{id}/"


# --- Utility Functions ---

def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_text(text):
    """
    Cleans text by stripping HTML, unescaping entities, replacing specific symbols,
    and standardizing whitespace.
    """
    if not text: return ""
    if "<" in text and ">" in text:
        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except: pass
    text = html.unescape(text)
    text = text.replace('\u200b', '').replace('\u2212', '-').replace('\u20b9', '₹').replace('&#8377;', '₹')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Similarity & GMP Matching Helpers ---

def normalize_company_name(name):
    """Normalize company name for comparison."""
    if not name: return ""
    name = name.lower().strip()
    name = re.sub(r'\(.*?\)', '', name)
    suffixes = ["limited", "ltd", "ltd.", "private", "pvt", "pvt.", "public", "inc", "inc.", "corporation", "corp", "corp.", "company", "co", "co.", "llp", "llc"]
    for suffix in suffixes:
        name = re.sub(rf"\b{re.escape(suffix)}\b", "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def calculate_similarity(s1, s2):
    """Calculate similarity score between two company names."""
    if not s1 or not s2: return 0.0
    n1, n2 = normalize_company_name(s1), normalize_company_name(s2)
    if not n1 or not n2: return 0.0
    if n1 == n2: return 1.0
    if n1 in n2 or n2 in n1: return 0.9
    w1, w2 = set(n1.split()), set(n2.split())
    jaccard = len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0.0
    longer = max(len(n1), len(n2))
    matches = sum(c1 == c2 for c1, c2 in zip(n1, n2))
    char_ratio = matches / longer if longer > 0 else 0.0
    return 0.6 * jaccard + 0.4 * char_ratio


# --- HTML Parsing ---

def extract_combined_html(initial_html):
    """
    Extracts additional HTML content potentially embedded in Next.js payloads.
    """
    all_html = [initial_html]
    patterns = [
        r'self\.__next_f\.push\(\[\d+,"(.*?)"\]\)',
        r'self\.__next_f\.push\(\[\d+,\s*"(.*?)"\]\)'
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, initial_html, flags=re.DOTALL):
            try:
                escaped_str = m.group(1)
                decoded = escaped_str.replace('\\"', '"').replace('\\\\', '\\')
                decoded = decoded.encode('utf-8').decode('unicode_escape')
                all_html.append(decoded)
            except Exception: pass
    return "\n".join(all_html)


def get_rendered_html(url: str, wait_selector: Optional[str] = None, timeout: int = 30):
    """Fetch page HTML using Playwright with lightweight stealth tweaks.
    
    Falls back to requests if Playwright is unavailable or fails.
    """
    stealth_js = """
    Object.defineProperty(navigator, 'webdriver', {get: () => false});
    window.chrome = { runtime: {} };
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
    Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
    const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
    if (originalQuery) {
      window.navigator.permissions.query = (parameters) =>
        parameters.name === 'notifications' ? Promise.resolve({ state: Notification.permission }) : originalQuery(parameters);
    }
    """

    if sync_playwright is None:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception:
            return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
            context = browser.new_context(user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ))
            context.add_init_script(stealth_js)
            page = context.new_page()
            try:
                page.goto(url, wait_until='domcontentloaded', timeout=timeout*1000)
            except Exception:
                page.goto(url, wait_until='networkidle', timeout=timeout*1000)
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=timeout*1000)
                except Exception:
                    pass
            content = page.content()
            try:
                page.close()
            except: pass
            try:
                context.close()
            except: pass
            try:
                browser.close()
            except: pass
            return content
    except Exception:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception:
            return None


def _extract_json_by_key(html: str, key: str) -> Optional[dict]:
    """Attempt to find a JSON object in HTML that starts with the given key."""
    if not html or not key:
        return None
    qkey = f'"{key}":'
    idx = html.find(qkey)
    if idx == -1:
        qkey = f"'{key}':"
        idx = html.find(qkey)
        if idx == -1:
            return None
    start = html.find('{', idx)
    if start == -1:
        return None
    depth = 0
    end = None
    for i in range(start, len(html)):
        ch = html[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    snippet = html[start:end]
    try:
        return json.loads(snippet)
    except Exception:
        try:
            fixed = snippet.replace("\\'", "'")
            return json.loads(fixed)
        except Exception:
            return None


def parse_any_table(table):
    """Generic table parser."""
    if not table: return []
    rows = []
    headers = []
    
    thead = table.find("thead")
    if thead:
        headers = []
        for i, th in enumerate(thead.find_all(["th", "td"])):
            h = clean_text(th.get_text())
            if not h:
                h = f"unnamed_col_{i+1}"
            headers.append(h)
    
    all_tr = table.find_all("tr")
    if (not headers or not any(headers)) and all_tr:
        headers = []
        for i, td in enumerate(all_tr[0].find_all(["th", "td"])):
            h = clean_text(td.get_text())
            if not h:
                h = f"unnamed_col_{i+1}"
            headers.append(h)
        all_tr = all_tr[1:]
    
    if not headers or not any(headers):
        for tr in all_tr:
            tds = tr.find_all(["td", "th"])
            if tds:
                rows.append([clean_text(td.get_text()) for td in tds])
        return rows

    for tr in all_tr:
        tds = tr.find_all(["td", "th"])
        if len(tds) == len(headers) and headers:
            item = {}
            for i in range(len(headers)):
                k = headers[i]
                v = clean_text(tds[i].get_text())
                if k:
                    item[k] = v
                else:
                    item[f"unnamed_col_{i+1}"] = v
            if any(item.get(h) != h for h in headers if h in item):
                rows.append(item)
        elif tds:
            rows.append([clean_text(td.get_text()) for td in tds])
            
    return rows


def parse_ipo_html_content(html_content):
    """Structural parser for the IPO page content."""
    combined_content = extract_combined_html(html_content)
    soup = BeautifulSoup(combined_content, "html.parser")
    ipo_data = {}

    section_map = {
        r"IPO Details": "ipo_details",
        r"Timetable": "timetable",
        r"Reservation": "reservation",
        r"Lot Size": "lot_size",
        r"Financials": "financials",
        r"Indicator": "kpis",
        r"Subscription Status": "subscription",
        r"Recommendations": "recommendations",
        r"Broker Review": "broker_reviews",
        r"Peer Comparison": "peer_comparison",
        r"Member Review": "member_reviews",
        r"Objects of the Issue": "objects_of_issue",
        r"FAQ": "faqs"
    }

    cards = {}
    for c in soup.find_all("div", class_="card-ipo"):
        l, v = c.find("p", class_="text-muted"), c.find("p", class_="fs-5")
        if l and v: cards[clean_text(l.get_text())] = clean_text(v.get_text())
    if cards: ipo_data["summary_cards"] = cards

    all_tags = list(soup.find_all(True))
    headers = [t for t in all_tags if t.name in ["h1", "h2", "h3", "h4"] and clean_text(t.get_text())]
    
    for i, h in enumerate(headers):
        title = clean_text(h.get_text())
        next_h = headers[i+1] if i+1 < len(headers) else None
        matched_key = None
        for pattern, key in section_map.items():
            if re.search(pattern, title, re.I):
                matched_key = key
                break
        if not matched_key: continue

        start_idx = all_tags.index(h)
        end_idx = all_tags.index(next_h) if next_h else len(all_tags)
        section_tags = all_tags[start_idx:end_idx]

        if matched_key == "ipo_details":
            details = ipo_data.get("ipo_details", {})
            for t in section_tags:
                if t.name == "table":
                    for r in t.find_all("tr"):
                        tds = r.find_all(["td", "th"])
                        if len(tds) == 2:
                            key = clean_text(tds[0].get_text())
                            val = clean_text(tds[1].get_text(separator=" "))
                            if key:
                                details[key] = val
            ipo_data["ipo_details"] = details

        elif matched_key == "timetable":
            tt = {}
            for t in section_tags:
                if t.name == "li":
                    spans = t.find_all("span")
                    if len(spans) == 2: tt[clean_text(spans[0].get_text())] = clean_text(spans[1].get_text())
                elif t.name == "table":
                    for r in t.find_all("tr"):
                        cells = r.find_all(["td", "th"])
                        if len(cells) == 2: tt[clean_text(cells[0].get_text())] = clean_text(cells[1].get_text())
            if tt: ipo_data["timetable"] = tt
        elif matched_key in ["reservation", "subscription"]:
            obj = ipo_data.get(matched_key, {"summary": "", "table": []})
            for t in section_tags:
                if t.name == "p" and not obj["summary"]: obj["summary"] = clean_text(t.get_text())
                if t.name == "table":
                    tbl = parse_any_table(t)
                    if tbl: obj["table"] = tbl
            ipo_data[matched_key] = obj
        elif matched_key == "kpis":
            k_data = ipo_data.get("kpis", {})
            idx = len(k_data) + 1
            for t in section_tags:
                if t.name == "table":
                    k_data[f"table_{idx}"] = parse_any_table(t)
                    idx += 1
            if k_data: ipo_data["kpis"] = k_data
        elif matched_key == "member_reviews":
            for t in section_tags:
                if t.name == "table":
                    revs = []
                    rows = t.find_all("tr")
                    j = 0
                    while j < len(rows):
                        tds = rows[j].find_all(["td", "th"])
                        if len(tds) >= 3:
                            r = {"name": clean_text(tds[1].get_text()), "rating": clean_text(tds[2].get_text())}
                            if j + 1 < len(rows):
                                next_cells = rows[j+1].find_all(["td", "th"])
                                if len(next_cells) == 1:
                                    r["comment"] = clean_text(next_cells[0].get_text())
                                    j += 2
                                else: j += 1
                            else: j += 1
                            revs.append(r)
                        else: j += 1
                    ipo_data["member_reviews"] = revs
                    break
        else:
            for t in section_tags:
                if t.name == "table":
                    if matched_key not in ipo_data: ipo_data[matched_key] = parse_any_table(t)
                    break

    if not ipo_data.get("subscription") or not ipo_data["subscription"].get("summary"):
        sub_match = re.search(r"IPO is subscribed [\d.]+ times.*?(?=\u003c|\n|$)", combined_content)
        if sub_match:
            if "subscription" not in ipo_data: ipo_data["subscription"] = {}
            ipo_data["subscription"]["summary"] = clean_text(sub_match.group(0))

    promoter_p = soup.find(string=re.compile(r"promoters of the Company are", re.IGNORECASE))
    if promoter_p: ipo_data["promoter_info"] = clean_text(promoter_p)

    # RHP & GMP Link Extraction
    rhp_link = ""
    gmp_link = ""
    links = soup.find_all("a", href=True)
    for a in links:
        text = a.get_text(strip=True).upper()
        if not rhp_link and ("RHP" == text or (len(text) < 10 and "RHP" in text)):
            rhp_link = a["href"]
        if not gmp_link and ("GMP" == text or (len(text) < 10 and "GMP" in text)):
            gmp_link = a["href"]
            
    if rhp_link:
        ipo_data["rhp_link"] = rhp_link
    if gmp_link:
        ipo_data["gmp_link"] = gmp_link

    return ipo_data


# --- RHP Download Logic ---

def download_rhp_locally(rhp_url, folder_name, session):
    """
    Downloads RHP files locally:
    - PDF: Downloads to local folder
    - ZIP: Downloads, extracts PDFs, and saves to local folder
    - HTML: Ignored
    """
    if not rhp_url:
        return []

    local_paths = []
    rhp_folder = os.path.join(BASE_OUTPUT_DIR, folder_name, "rhp_docs")
    ensure_dir(rhp_folder)

    try:
        if rhp_url.lower().endswith('.pdf'):
            file_name = rhp_url.split('/')[-1]
            local_path = os.path.join(rhp_folder, file_name)
            
            if os.path.exists(local_path):
                print(f"      - PDF already exists locally, skipping: {local_path}")
                local_paths.append(local_path)
                return local_paths

            print(f"      - Downloading PDF RHP: {rhp_url}")
            response = session.get(rhp_url, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"      - Saved PDF locally: {local_path}")
            local_paths.append(local_path)

        elif rhp_url.lower().endswith('.zip'):
            print(f"      - Downloading ZIP RHP: {rhp_url}")
            response = session.get(rhp_url, timeout=60)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for file_name in z.namelist():
                    if file_name.lower().endswith('.pdf'):
                        local_path = os.path.join(rhp_folder, os.path.basename(file_name))
                        
                        if os.path.exists(local_path):
                            print(f"        - Extracted file already exists locally, skipping: {local_path}")
                            local_paths.append(local_path)
                            continue

                        print(f"        - Extracting and saving PDF from ZIP: {file_name}")
                        with z.open(file_name) as pdf_file:
                            with open(local_path, 'wb') as f:
                                f.write(pdf_file.read())
                            print(f"        - Saved extracted PDF locally: {local_path}")
                            local_paths.append(local_path)
        
        elif rhp_url.lower().endswith('.html'):
            print(f"      - RHP is HTML, skipping download: {rhp_url}")
        
    except Exception as e:
        print(f"      - Error processing RHP document {rhp_url}: {e}")
        
    return local_paths


# --- Main Scraping Logic ---

def scrape_and_save_ipo_data():
    """
    1. Fetches IPO list from API
    2. For each IPO, fetches detailed information
    3. Saves each IPO's data to individual JSON files
    4. Downloads RHP documents to local folders
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/html, */*"
    }
    
    session = requests.Session()
    session.headers.update(headers)

    # Ensure base output directory exists
    ensure_dir(BASE_OUTPUT_DIR)

    print(f"Step 1: Fetching IPO list from {LIST_URL}...")
    try:
        response = session.get(LIST_URL, timeout=15)
        response.raise_for_status()
        ipo_data_raw = response.json()
        ipo_list_from_api = ipo_data_raw.get("ipoDropDownList", [])
        print(f"    - Found {len(ipo_list_from_api)} IPOs in API list")
    except Exception as e:
        print(f"Error fetching IPO list: {e}")
        return {"status": "error", "message": str(e)}

    # Get current time for open status check
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    total = len(ipo_list_from_api)
    processed_count = 0
    failed_count = 0

    print(f"Starting processing of {total} IPOs...")

    for index, ipo in enumerate(ipo_list_from_api, 1):
        ipo_id = str(ipo.get('id', ''))
        folder = ipo.get("urlrewrite_folder_name")
        ipo_name = ipo.get("ipo_news_title", "Unknown")
        ipo_period = ipo.get("ipo_period", "")
        order_date = ipo.get('orderdate', '')

        if not ipo_id or not folder:
            print(f"[{index}/{total}] Skipping invalid entry: {ipo_name}")
            continue

        is_open = order_date and order_date >= now_str
        print(f"[{index}/{total}] Processing IPO: {ipo_name} (ID: {ipo_id}, Open: {is_open})")

        # Parse dates from ipo_period
        open_date = ""
        close_date = ""
        if ipo_period and "-" in ipo_period:
            try:
                parts = [p.strip() for p in ipo_period.split("-")]
                if len(parts) == 2:
                    open_date = parts[0]
                    close_date = parts[1]
            except: pass

        target_url = HTML_BASE_URL.format(folder=folder, id=ipo_id)
        print(f"    Fetching: {target_url}")
        
        try:
            # Respectful throttle
            time.sleep(0.3)
            
            resp = session.get(target_url, timeout=20)
            resp.raise_for_status()
            
            # Parse HTML content
            parsed_details = parse_ipo_html_content(resp.text)
            
            # Check if subscription details are missing
            sub_missing = False
            used_playwright = False
            if not parsed_details.get('subscription'):
                sub_missing = True
            else:
                sub = parsed_details.get('subscription', {})
                if not sub.get('summary') and not sub.get('table'):
                    sub_missing = True

            # Try Playwright for subscription page if needed
            if sub_missing and folder:
                try:
                    subscription_url = f"https://www.chittorgarh.com/ipo_subscription/{folder}/{ipo_id}/"
                    print(f"    - Trying Playwright-rendered subscription URL: {subscription_url}")
                    rendered = get_rendered_html(subscription_url, wait_selector='table', timeout=20)
                    if rendered:
                        sub_parsed = parse_ipo_html_content(rendered)
                        if sub_parsed and 'subscription' in sub_parsed and sub_parsed['subscription'].get('summary'):
                            parsed_details['subscription'] = sub_parsed['subscription']
                            used_playwright = True
                            print(f"    - Subscription details obtained from subscription page via Playwright")
                        else:
                            combined_rendered = extract_combined_html(rendered)
                            sub_json = _extract_json_by_key(combined_rendered, 'subscriptionDataResponse')
                            if sub_json:
                                try:
                                    bids = sub_json.get('ipoBiddingDetails') or []
                                    if isinstance(bids, dict):
                                        bids = [bids]
                                    if bids and isinstance(bids, list):
                                        bid = bids[0]
                                        summary = {
                                            'total': bid.get('total') or bid.get('total_shares_bid_for'),
                                            'qib': bid.get('qib') or bid.get('qib_shares_bid_for'),
                                            'nii': bid.get('nii') or bid.get('nii_shares_bid_for'),
                                            'rii': bid.get('rii') or bid.get('rii_shares_bid_for'),
                                            'date_added': bid.get('date_added'),
                                            'total_application': bid.get('total_application') or bid.get('no_of_application')
                                        }
                                        table = [
                                            ['Qualified Institutional', summary.get('qib')],
                                            ['Non Institutional', summary.get('nii')],
                                            ['Retail Individual', summary.get('rii')],
                                            ['Total', summary.get('total')]
                                        ]
                                        parsed_details['subscription'] = {'summary': summary, 'table': table}
                                        used_playwright = True
                                        print(f"    - Subscription details obtained from embedded JSON via Playwright")
                                except Exception:
                                    pass
                except Exception as e:
                    print(f"    - Playwright subscription fetch failed: {e}")
            
            # Rename 'subscription' to 'subscription_details' for consistency
            if 'subscription' in parsed_details:
                parsed_details['subscription_details'] = parsed_details.pop('subscription')
            
            # Log RHP & GMP availability
            rhp_status = f"Found: {parsed_details['rhp_link']}" if "rhp_link" in parsed_details else "Not found"
            gmp_status = f"Found: {parsed_details['gmp_link']}" if "gmp_link" in parsed_details else "Not found"
            print(f"    - RHP Link: {rhp_status}")
            print(f"    - GMP Link: {gmp_status}")

            # Process and download RHP documents locally
            downloaded_rhp_files = []
            if "rhp_link" in parsed_details:
                downloaded_rhp_files = download_rhp_locally(parsed_details['rhp_link'], folder, session)
                if downloaded_rhp_files:
                    parsed_details["rhp_local_paths"] = downloaded_rhp_files
            
            # Create IPO folder structure
            ipo_folder = os.path.join(BASE_OUTPUT_DIR, folder)
            ensure_dir(ipo_folder)
            
            # Prepare comprehensive metadata JSON
            metadata = {
                'id': ipo_id,
                'name': ipo_name,
                'folder': folder,
                'ipo_period': ipo_period,
                'open_date': open_date,
                'close_date': close_date,
                'order_date': order_date,
                'is_open': is_open,
                'scraped_at': datetime.utcnow().isoformat(),
                'used_playwright': used_playwright
            }
            
            # Add all parsed details
            metadata.update(parsed_details)
            
            # Save metadata JSON
            metadata_file = os.path.join(ipo_folder, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"    - Saved metadata to: {metadata_file}")

            # --- Convert to Markdown ---
            slug = slugify(ipo_name)

            # 1. Metadata → structured markdown
            md_text = metadata_to_markdown(metadata)
            md_path = os.path.join(ipo_folder, f"{slug}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_text)
            print(f"    - Metadata markdown → {slug}.md")

            # 2. Extract URLs → <slug>.metadata.json
            urls_data = extract_urls_metadata(metadata)
            urls_path = os.path.join(ipo_folder, f"{slug}.metadata.json")
            with open(urls_path, 'w', encoding='utf-8') as f:
                json.dump(urls_data, f, indent=2, ensure_ascii=False)
            print(f"    - URL metadata → {slug}.metadata.json")

            # 3. RHP PDFs → markdown via pymupdf4llm
            rhp_dir = os.path.join(ipo_folder, "rhp_docs")
            if os.path.isdir(rhp_dir):
                for pdf_name in sorted(os.listdir(rhp_dir)):
                    if pdf_name.lower().endswith('.pdf'):
                        pdf_path = os.path.join(rhp_dir, pdf_name)
                        md_out = os.path.join(rhp_dir, os.path.splitext(pdf_name)[0] + ".md")
                        if os.path.exists(md_out):
                            print(f"    - [SKIP] PDF already converted: {pdf_name}")
                            continue
                        try:
                            convert_pdf_to_markdown(pdf_path, md_out)
                        except Exception as pdf_err:
                            print(f"    - [ERR] PDF conversion failed for {pdf_name}: {pdf_err}")

            processed_count += 1

        except Exception as e:
            print(f"    - Failed to process {ipo_name}: {e}")
            failed_count += 1

    # --- Assemble ipo_data_md folder ---
    print(f"\nAssembling final markdown folder: {MD_OUTPUT_DIR}")
    assemble_md_output()

    print(f"\n--- Scraping Completed ---")
    print(f"Total processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print(f"Markdown directory: {MD_OUTPUT_DIR}")
    return {
        "status": "success", 
        "total_processed": processed_count,
        "failed": failed_count,
        "output_directory": BASE_OUTPUT_DIR,
        "md_output_directory": MD_OUTPUT_DIR
    }


def assemble_md_output():
    """
    Assemble ipo_data_md/ from ipo_data/.
    For each IPO folder copies only:
      - <slug>.md              (metadata markdown)
      - <slug>.metadata.json   (URLs)
      - rhp_docs/*.md          (RHP markdowns)
    """
    src_base = Path(BASE_OUTPUT_DIR)
    dst_base = Path(MD_OUTPUT_DIR)

    if not src_base.is_dir():
        print(f"  [ERR] Source not found: {BASE_OUTPUT_DIR}")
        return

    # Clear and recreate
    if dst_base.exists():
        shutil.rmtree(dst_base)
    dst_base.mkdir(parents=True)

    copied = 0
    for folder in sorted(src_base.iterdir()):
        if not folder.is_dir():
            continue

        dst_folder = dst_base / folder.name
        has_content = False

        # Copy <slug>.md and <slug>.metadata.json
        for f in folder.iterdir():
            if f.is_file() and f.suffix == ".md" and f.stem != "metadata":
                dst_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst_folder / f.name)
                has_content = True
            elif f.is_file() and f.name.endswith(".metadata.json") and f.name != "metadata.json":
                dst_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst_folder / f.name)
                has_content = True

        # Copy rhp_docs/*.md
        rhp_src = folder / "rhp_docs"
        if rhp_src.is_dir():
            for md_file in sorted(rhp_src.glob("*.md")):
                rhp_dst = dst_folder / "rhp_docs"
                rhp_dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(md_file, rhp_dst / md_file.name)
                has_content = True

        if has_content:
            copied += 1

    print(f"  Assembled {copied} IPO folders into {MD_OUTPUT_DIR}")


if __name__ == "__main__":
    result = scrape_and_save_ipo_data()
    print(f"\nResult: {json.dumps(result, indent=2)}")
