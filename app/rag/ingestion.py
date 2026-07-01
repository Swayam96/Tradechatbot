"""Website crawling and content extraction."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from app.config import Config, RAW_DATA_DIR
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Tags whose content is typically boilerplate / navigation
SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript"}
CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".article-body",
    ".content",
    "#content",
    ".post-content",
    ".entry-content",
]


@dataclass
class PageDocument:
    """Represents a crawled web page."""

    url: str
    title: str
    text: str
    html_path: Optional[str] = None
    text_path: Optional[str] = None
    links: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class WebsiteIngestion:
    """Crawl a website within the same domain and extract article text."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
        crawl_delay: Optional[float] = None,
        user_agent: Optional[str] = None,
    ):
        self.base_url = (base_url or Config.TARGET_WEBSITE_BASE_URL).rstrip("/")
        parsed = urlparse(self.base_url)
        self.domain = parsed.netloc
        self.scheme = parsed.scheme or "https"
        self.max_pages = max_pages or Config.MAX_PAGES
        self.max_depth = max_depth or Config.MAX_DEPTH
        self.crawl_delay = crawl_delay or Config.CRAWL_DELAY_SECONDS
        self.user_agent = user_agent or Config.USER_AGENT
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._robots: Optional[RobotFileParser] = None

    def _load_robots(self) -> RobotFileParser:
        """Load and cache robots.txt for the target domain."""
        if self._robots is None:
            robots_url = f"{self.scheme}://{self.domain}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                logger.info("Loaded robots.txt from %s", robots_url)
            except Exception as exc:
                logger.warning("Could not read robots.txt (%s); allowing all URLs.", exc)
            self._robots = rp
        return self._robots

    def _can_fetch(self, url: str) -> bool:
        """Check robots.txt permission for a URL."""
        try:
            return self._load_robots().can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _normalize_url(self, url: str) -> Optional[str]:
        """Normalize URL and restrict to same domain."""
        url, _ = urldefrag(url)
        parsed = urlparse(url)
        if not parsed.netloc:
            url = urljoin(self.base_url + "/", url)
            parsed = urlparse(url)
        if parsed.netloc != self.domain:
            return None
        if parsed.scheme not in ("http", "https"):
            return None
        # Skip common non-content resources
        path_lower = parsed.path.lower()
        if re.search(r"\.(pdf|jpg|jpeg|png|gif|svg|zip|mp4|mp3|css|js)$", path_lower):
            return None
        return url.rstrip("/")

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract same-domain links from a page."""
        links: List[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            normalized = self._normalize_url(urljoin(current_url, href))
            if normalized:
                links.append(normalized)
        return links

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return "Untitled"

    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        """Extract main article/body text using heuristics."""
        for tag_name in SKIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        content_root = None
        for selector in CONTENT_SELECTORS:
            if selector.startswith(".") or selector.startswith("#"):
                content_root = soup.select_one(selector)
            else:
                content_root = soup.find(selector)
            if content_root:
                break

        if content_root is None:
            content_root = soup.body or soup

        paragraphs = []
        for element in content_root.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
            text = element.get_text(" ", strip=True)
            if len(text) > 30:
                paragraphs.append(text)

        if not paragraphs:
            text = content_root.get_text("\n", strip=True)
            return re.sub(r"\n{2,}", "\n\n", text)

        return "\n\n".join(paragraphs)

    def _save_raw(self, url: str, html: str, text: str) -> tuple[str, str]:
        """Persist raw HTML and extracted text to disk."""
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
        html_path = RAW_DATA_DIR / f"{url_hash}.html"
        text_path = RAW_DATA_DIR / f"{url_hash}.txt"
        html_path.write_text(html, encoding="utf-8")
        text_path.write_text(text, encoding="utf-8")
        return str(html_path), str(text_path)

    def fetch_page(self, url: str) -> Optional[PageDocument]:
        """Fetch a single page and return extracted content."""
        if not self._can_fetch(url):
            logger.debug("Blocked by robots.txt: %s", url)
            return None

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None

            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            title = self._extract_title(soup)
            text = self._extract_main_text(soup)

            if len(text.strip()) < 100:
                logger.debug("Skipping thin content: %s", url)
                return None

            links = self._extract_links(soup, url)
            html_path, text_path = self._save_raw(url, html, text)
            return PageDocument(
                url=url,
                title=title,
                text=text,
                html_path=html_path,
                text_path=text_path,
                links=links,
            )
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def crawl(self) -> List[PageDocument]:
        """
        Breadth-first crawl starting from base_url.

        Respects robots.txt, max depth, and max pages limits.
        """
        Config.ensure_directories()
        visited: Set[str] = set()
        queue: List[tuple[str, int]] = [(self.base_url, 0)]
        documents: List[PageDocument] = []

        logger.info(
            "Starting crawl of %s (max_pages=%d, max_depth=%d)",
            self.base_url,
            self.max_pages,
            self.max_depth,
        )

        while queue and len(documents) < self.max_pages:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            if depth > self.max_depth:
                continue

            page = self.fetch_page(url)
            if page:
                documents.append(page)
                logger.info("Crawled [%d/%d]: %s", len(documents), self.max_pages, url)

            if self.crawl_delay > 0:
                time.sleep(self.crawl_delay)

            if page and depth < self.max_depth:
                for link in page.links:
                    if link not in visited:
                        queue.append((link, depth + 1))

        manifest_path = RAW_DATA_DIR / "manifest.json"
        manifest_path.write_text(
            json.dumps([doc.to_dict() for doc in documents], indent=2),
            encoding="utf-8",
        )
        logger.info("Crawl complete. Saved %d documents.", len(documents))
        return documents
