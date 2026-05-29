import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

SAVE_DIR = "esg_reports"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Known direct PDF URLs for major company ESG/sustainability reports
DIRECT_PDFS = {
    "TotalEnergies_2023": "https://totalenergies.com/sites/g/files/nytnzq121/files/documents/2024-03/Sustainability-Climate-2024-Progress-Report-TotalEnergies.pdf",
    "BNP_Paribas_2023": "https://invest.bnpparibas/document/bnp-paribas-2023-universal-registration-document",
    "Danone_2023": "https://www.danone.com/content/dam/danone-corp/danone-com/investors/en-annual-reports/2024/Danone_URD_2023.pdf",
    "AXA_2023": "https://www.axa.com/en/newsroom/publications/axa-2023-climate-report",
    "Schneider_2023": "https://www.se.com/ww/en/assets/564/document/1234561/2023-sustainability-report-schneider-electric.pdf",
}

# Company sustainability report pages to auto-scrape PDF links from
SCRAPE_PAGES = [
    "https://totalenergies.com/sustainability/publications",
    "https://www.lvmh.com/news-documents/documents/",
    "https://www.airbus.com/en/sustainability/reporting-and-publications",
]


def download_pdf(url: str, filename: str, save_dir: str = SAVE_DIR) -> bool:
    """Download a PDF from a direct URL."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename if filename.endswith(".pdf") else filename + ".pdf")
    if os.path.exists(filepath):
        print(f"  [skip] already exists: {filepath}")
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = os.path.getsize(filepath) // 1024
        print(f"  [ok] {filename}.pdf  ({size_kb} KB)")
        return True
    except Exception as e:
        print(f"  [fail] {filename}: {e}")
        return False


def scrape_pdf_links(page_url: str) -> list:
    """Find all PDF links on a webpage."""
    try:
        r = requests.get(page_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.lower().endswith(".pdf"):
                full_url = urljoin(page_url, href)
                label = tag.get_text(strip=True) or os.path.basename(href)
                links.append((label, full_url))
        return links
    except Exception as e:
        print(f"  [fail] scraping {page_url}: {e}")
        return []


def scrape_from_pages(pages: list = SCRAPE_PAGES, save_dir: str = SAVE_DIR):
    """Scrape PDF links from sustainability report pages and download them."""
    for page_url in pages:
        domain = urlparse(page_url).netloc.replace("www.", "")
        print(f"\nScraping: {page_url}")
        links = scrape_pdf_links(page_url)
        if not links:
            print("  No PDF links found.")
            continue
        # Filter for ESG/sustainability relevant PDFs
        keywords = ["esg", "sustain", "climate", "report", "csrd", "extra-financ", "non-financial", "responsibility"]
        relevant = [(label, url) for label, url in links
                    if any(kw in label.lower() or kw in url.lower() for kw in keywords)]
        if not relevant:
            relevant = links[:3]  # fallback: take first 3 PDFs
        print(f"  Found {len(relevant)} relevant PDF(s)")
        for label, url in relevant:
            safe_name = re.sub(r'[^\w\-]', '_', f"{domain}_{label}")[:80]
            download_pdf(url, safe_name, save_dir)


def download_known_reports(save_dir: str = SAVE_DIR):
    """Download the hardcoded known ESG report PDFs."""
    print("=== Downloading known ESG reports ===")
    for name, url in DIRECT_PDFS.items():
        download_pdf(url, name, save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ESG Report Scraper")
    parser.add_argument("--mode", choices=["known", "scrape", "both"], default="known",
                        help="known: download hardcoded URLs | scrape: find PDFs on pages | both: do both")
    parser.add_argument("--url", help="Scrape a specific URL for PDF links")
    parser.add_argument("--dir", default=SAVE_DIR, help="Save directory")
    args = parser.parse_args()

    if args.url:
        print(f"\nScraping: {args.url}")
        links = scrape_pdf_links(args.url)
        for label, url in links:
            print(f"  {label}\n    {url}")
        download = input("\nDownload all? (y/n): ")
        if download.lower() == "y":
            for label, url in links:
                safe = re.sub(r'[^\w\-]', '_', label)[:80]
                download_pdf(url, safe, args.dir)
    elif args.mode in ("known", "both"):
        download_known_reports(args.dir)
        if args.mode == "both":
            scrape_from_pages(save_dir=args.dir)
    else:
        scrape_from_pages(save_dir=args.dir)

    print(f"\nDone. PDFs saved to: {os.path.abspath(args.dir)}/")
