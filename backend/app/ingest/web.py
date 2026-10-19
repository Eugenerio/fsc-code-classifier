"""Site crawler: homepage + the most product-relevant same-site pages, reduced to clean text.

All fetched content is untrusted. Every hop (including redirects) is checked against an SSRF guard.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from urllib.parse import ParseResult, urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

from app.config import get_settings
from app.schemas import SourceDoc

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FSCClassifierBot/0.1; +prototype research crawler)",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",  # some WAFs 403 requests that lack these
    "Accept-Language": "en-US,en;q=0.9",
}
MAX_BYTES = 2_000_000
MAX_PAGE_CHARS = 20_000
MAX_REDIRECTS = 5
CONCURRENCY = 4
BLOCKED = "blocked: "  # prefix of SSRF-guard refusals

FREE_MAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "mail.ru", "bk.ru", "list.ru", "inbox.ru", "rambler.ru", "me.com", "mac.com",
    "mail.com", "email.com", "web.de", "t-online.de", "qq.com", "163.com", "126.com", "sina.com", "naver.com",
    "daum.net", "ukr.net", "i.ua", "meta.ua", "pm.me", "hey.com", "tutanota.com", "tuta.com", "fastmail.com",
    "zoho.com", "comcast.net", "verizon.net", "att.net", "sbcglobal.net", "cox.net", "charter.net",
    "bellsouth.net", "earthlink.net", "rediffmail.com", "libero.it", "orange.fr", "free.fr",
}
# providers that exist under many country TLDs (yahoo.co.uk, outlook.de, ...): match on the second-level label
FREE_MAIL_LABELS = {"yahoo", "hotmail", "outlook", "live", "msn", "icloud", "aol", "proton", "protonmail", "gmx", "yandex"}

# second-level labels that sit under a 2-letter ccTLD, e.g. example.co.uk
_SLD = {"co", "com", "org", "net", "gov", "edu", "ac", "ltd", "plc"}
_HOST_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9¡-￿]([a-z0-9¡-￿-]{0,61}[a-z0-9¡-￿])?\.)+[a-z0-9¡-￿-]{2,63}$")

LINK_HINTS = (  # substrings of the URL path / anchor text
    "product", "service", "capabilit", "catalog", "about", "what-we-do", "whatwedo", "industr", "parts",
    "equipment", "solution", "offering", "manufactur", "inventory", "suppl", "repair", "overhaul", "fabricat",
    "machin", "company", "brands", "shop", "categor",
)
LINK_AVOID = {  # whole words ("press" must not reject "/pressure-control")
    "privacy", "terms", "legal", "cookie", "cookies", "login", "signin", "register", "account", "cart", "checkout",
    "career", "careers", "job", "jobs", "blog", "news", "press", "event", "events", "sitemap", "feed", "search",
    "subscribe", "contact", "faq", "disclaimer", "gdpr", "accessibility", "wp", "admin",
}
SKIP_EXT = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".zip", ".rar", ".doc", ".docx", ".xls",
    ".xlsx", ".ppt", ".pptx", ".mp3", ".mp4", ".mov", ".avi", ".css", ".js", ".xml", ".json", ".exe", ".dmg",
)
STRIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form", "iframe", "template", "button", "select"}
WRAPPER_TAGS = {"nav", "footer", "header", "form"}  # some sites (ASP.NET) wrap the whole page in one of these
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCK_TAGS = HEADINGS | {
    "p", "div", "section", "article", "main", "ul", "ol", "li", "tr", "table", "dl", "dt", "dd", "blockquote",
    "pre", "figure", "figcaption", "aside", "details", "summary", "hr", "br",
}
NOISE_CONTAINERS = {"div", "section", "aside", "ul"}
CLOSERS = {"a": " ", "td": " | ", "th": " | "}  # adjacent links must not fuse into one word; table cells stay on one row
NOISE_ATTR_RE = re.compile(r"cookie|consent|gdpr|popup|newsletter|breadcrumb", re.I)
NON_PRODUCT_ALT_RE = re.compile(r"logo|icon|banner|arrow|facebook|twitter|linkedin|instagram|youtube|spacer|placeholder|badge", re.I)


# --------------------------------------------------------------------------- URL helpers


def normalize_url(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    if "://" not in raw:
        raw = "https://" + raw.lstrip("/")
    try:
        p = urlparse(raw)
        host, port = (p.hostname or "").lower(), p.port
    except ValueError:  # malformed port / bracketed host
        return None
    if p.scheme not in ("http", "https") or p.username or not (_HOST_RE.match(host) or _is_ip(host)):
        return None
    netloc = f"[{host}]" if ":" in host else host
    return urlunparse((p.scheme, netloc + (f":{port}" if port else ""), p.path, "", p.query, ""))


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def registrable_domain(host: str) -> str:
    """Approximate eTLD+1 without a public-suffix list (good enough to keep a crawl on one company's site)."""
    labels = host.lower().removeprefix("www.").split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _SLD:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def guess_site_from_email_domain(domain: str | None) -> str | None:
    if not domain:
        return None
    domain = domain.strip().lower().rsplit("@", 1)[-1].strip(".")
    if not _HOST_RE.match(domain):
        return None
    root = registrable_domain(domain)
    if root in FREE_MAIL_DOMAINS or root.split(".")[0] in FREE_MAIL_LABELS:
        return None
    return f"https://{domain}"


# --------------------------------------------------------------------------- SSRF guard


async def _resolve(host: str) -> list[str]:
    if _is_ip(host):
        return [host]
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


async def check_public_url(url: str) -> None:
    """Raise ValueError unless url is http(s) and every address its host resolves to is publicly routable."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError(f"{BLOCKED}unsupported URL scheme {p.scheme!r}")
    try:
        addresses = await _resolve(p.hostname)
    except OSError as e:
        raise ValueError(f"could not resolve host {p.hostname!r}") from e
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        ip = getattr(ip, "ipv4_mapped", None) or ip  # ::ffff:127.0.0.1 must not slip through
        if not ip.is_global:
            raise ValueError(f"{BLOCKED}host resolves to a private, loopback or link-local address")


async def _get(client: httpx.AsyncClient, url: str, timeout: float) -> tuple[str, bytes]:
    """GET with manual redirects so each hop is re-checked; the body is capped at MAX_BYTES."""
    for _ in range(MAX_REDIRECTS + 1):
        await check_public_url(url)
        async with client.stream("GET", url, timeout=timeout, headers=HEADERS) as resp:
            if resp.is_redirect and (location := resp.headers.get("location")):
                url = urljoin(str(resp.url), location)
                continue
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "text/html").lower()
            if not any(t in ctype for t in ("html", "text/plain")):
                raise ValueError(f"not an HTML page ({ctype.split(';')[0]})")
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body += chunk
                if len(body) >= MAX_BYTES:
                    break
            return url, bytes(body[:MAX_BYTES])
    raise ValueError("too many redirects")


# --------------------------------------------------------------------------- HTML -> text


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _walk_json_ld(node: object, out: list[str]) -> None:
    if isinstance(node, list):
        for item in node:
            _walk_json_ld(item, out)
    elif isinstance(node, dict):
        name, desc = node.get("name"), node.get("description")
        if isinstance(name, str) and name.strip():
            out.append(_clean(name) + (f": {_clean(desc)}" if isinstance(desc, str) and desc.strip() else ""))
        for key in ("@graph", "itemListElement", "hasOfferCatalog", "makesOffer", "itemOffered"):
            _walk_json_ld(node.get(key), out)


def _skip(tag: Tag, total: int) -> bool:
    if tag.name in STRIP_TAGS:
        # some sites (ASP.NET) wrap the whole page in one <form>/<header>: unwrap those instead of dropping the page
        return not (tag.name in WRAPPER_TAGS and len(tag.get_text()) > 0.6 * total)
    if tag.name in NOISE_CONTAINERS:
        # short widgets only: a class like "cookie-enabled" on a big wrapper must not delete the page
        attrs = f"{tag.get('id', '')} {' '.join(tag.get('class') or [])}"
        return bool(NOISE_ATTR_RE.search(attrs)) and len(tag.get_text()) < 2000
    return False


def _walk(root: Tag, total: int) -> tuple[list[str], list[str]]:
    """Linearise the DOM into text lines (block elements break lines) and collect image alt texts.

    Iterative and mutation-free: inserting separators into the tree is quadratic on pages with thousands of siblings.
    """
    lines: list[str] = []
    alts: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            lines.append("".join(buf))
            buf.clear()

    stack = [(iter(root.children), "\n")]  # (children, what to emit when the element closes)
    while stack:
        children, closer = stack[-1]
        for child in children:
            if isinstance(child, Tag):
                if _skip(child, total):
                    continue
                name = child.name
                if name == "img" and 4 <= len(alt := _clean(str(child.get("alt") or ""))) <= 120 and not NON_PRODUCT_ALT_RE.search(alt):
                    alts.append(alt)
                if name in BLOCK_TAGS:
                    flush()
                if name in HEADINGS:
                    buf.append("# ")
                elif name == "li":
                    buf.append("- ")
                stack.append((iter(child.children), "\n" if name in BLOCK_TAGS else CLOSERS.get(name, "")))
                break
            if type(child) is NavigableString:  # not comments, CDATA, doctype, script/style strings
                buf.append(str(child))
        else:
            stack.pop()
            flush() if closer == "\n" else buf.append(closer)
    flush()
    return lines, alts


def parse_html(html: bytes | str, base_url: str = "") -> tuple[str, list[tuple[str, str]]]:
    """Return (clean main text, [(absolute link, anchor text)]). Links are taken before nav/footer are skipped."""
    soup = BeautifulSoup(html, "lxml")

    links = [(urljoin(base_url, str(a["href"])), _clean(a.get_text(" "))) for a in soup.find_all("a", href=True)]

    head: list[str] = []
    if soup.title and (title := _clean(soup.title.get_text())):
        head.append(f"Title: {title}")
    for meta in soup.find_all("meta", attrs={"name": "description"}) + soup.find_all("meta", attrs={"property": "og:description"}):
        if desc := _clean(str(meta.get("content", ""))):
            head.append(f"Description: {desc}")
    structured: list[str] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            _walk_json_ld(json.loads(script.string or ""), structured)
        except ValueError:
            continue
    head += [f"Structured data: {s}" for s in structured[:15]]
    # an <h1> inside <header> is skipped by the walk, but it is often the best one-line statement of what the site is
    head += [f"# {t}" for h in soup.find_all("h1") if h.find_parent("header") and (t := _clean(h.get_text(" ")))][:2]
    head = list(dict.fromkeys(head))  # meta and og:description are usually identical

    raw_lines, alts = _walk(soup.body or soup, len(soup.get_text()))
    lines: list[str] = []
    seen: set[str] = set()
    for raw in raw_lines:
        line = _clean(raw).strip("| ")
        if len(line) >= 2 and re.search(r"[A-Za-z0-9]", line) and line not in seen:
            seen.add(line)
            lines.append(line)
    if alts:
        lines.append("Image captions: " + "; ".join(dict.fromkeys(alts[:40])))

    return "\n".join(head + ["---"] * bool(head) + lines)[:MAX_PAGE_CHARS], links


def _page_key(p: ParseResult) -> str:
    """Identity of a page: www/apex, trailing slash and fragments do not make a different page."""
    query = f"?{p.query}" if len(p.query) < 40 else ""
    return (p.hostname or "").removeprefix("www.") + (p.path.rstrip("/") or "/") + query


def rank_links(links: list[tuple[str, str]], home: str, limit: int) -> list[str]:
    """Same-site links ordered by how likely they describe products (anchor text counts double vs. URL path)."""
    site = registrable_domain(urlparse(home).hostname or "")
    home_key = _page_key(urlparse(home))
    found: dict[str, tuple[str, tuple[int, int, int]]] = {}  # page key -> (url to fetch, sort key)
    for order, (href, anchor) in enumerate(links):
        p = urlparse(href)
        if p.scheme not in ("http", "https") or registrable_domain(p.hostname or "") != site:
            continue
        key, path, anchor = _page_key(p), p.path.lower(), anchor.lower()
        if path.endswith(SKIP_EXT) or key == home_key or key in found:
            continue
        if LINK_AVOID & set(re.findall(r"[a-z0-9]+", f"{path} {anchor}")):
            continue
        hits = sum(w in path for w in LINK_HINTS) + 2 * sum(w in anchor for w in LINK_HINTS)
        found[key] = (urlunparse((p.scheme, p.netloc.lower(), p.path or "/", "", p.query, "")), (-hits, path.count("/"), order))
    return [url for url, _ in sorted(found.values(), key=lambda v: v[1])[:limit]]


# --------------------------------------------------------------------------- public API


def _failed(url: str, error: str) -> SourceDoc:
    return SourceDoc(kind="website", label=url, ok=False, error=error)


async def _fetch_page(client: httpx.AsyncClient, url: str, timeout: float) -> tuple[SourceDoc, list[tuple[str, str]]]:
    final_url = url
    try:
        async with asyncio.timeout(2 * timeout):  # httpx timeouts are per-phase; this bounds a slow-drip response
            final_url, body = await _get(client, url, timeout)
        text, links = parse_html(body, final_url)
    except httpx.HTTPStatusError as e:
        return _failed(url, f"HTTP {e.response.status_code}"), []
    except TimeoutError:
        return _failed(url, "timed out"), []
    except httpx.HTTPError as e:
        return _failed(url, f"{type(e).__name__}: {e}"[:200]), []
    except Exception as e:  # SSRF refusals, parse errors, ...: a bad page must never break ingestion
        return _failed(url, str(e) or type(e).__name__), []
    if not text.strip():
        return _failed(final_url, "no readable text (page may need JavaScript)"), links
    return SourceDoc(kind="website", label=final_url, text=text), links


def _www_sibling(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.removeprefix("www.") if p.netloc.startswith("www.") else f"www.{p.netloc}"
    return urlunparse(p._replace(netloc=host))


async def _crawl(client: httpx.AsyncClient, url: str, max_pages: int, timeout: float) -> list[SourceDoc]:
    home, links = await _fetch_page(client, url, timeout)
    refused = (home.error or "").startswith(BLOCKED) or _is_ip(urlparse(url).hostname or "")
    if not home.ok and not links and not refused:
        # real-world sites often serve a broken certificate or a 403 on only one of www / apex
        alt, alt_links = await _fetch_page(client, _www_sibling(url), timeout)
        if alt.ok:
            home, links = alt, alt_links
    # the homepage may have redirected elsewhere (example.com -> www.example.com): "same site" means where we ended up
    targets = rank_links(links, home.label, max_pages - 1)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def fetch(target: str) -> SourceDoc:
        async with sem:
            return (await _fetch_page(client, target, timeout))[0]

    docs = [home, *await asyncio.gather(*(fetch(t) for t in targets))]
    seen: set[str] = set()  # sidebars/menus repeat on every page: keep each line once across the whole site
    for doc in docs:
        lines = [line for line in doc.text.split("\n") if line not in seen or line == "---"]
        seen.update(lines)
        doc.text = "\n".join(lines)
    return docs


async def fetch_site(
    url: str,
    *,
    max_pages: int | None = None,
    timeout: float = 15.0,
    client: httpx.AsyncClient | None = None,
) -> list[SourceDoc]:
    normalized = normalize_url(url)
    if normalized is None:
        return [_failed(url, "invalid URL")]
    pages = max_pages or get_settings().max_site_pages
    if client is not None:  # injected (tests): caller owns its lifecycle
        return await _crawl(client, normalized, pages, timeout)
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as own:
        return await _crawl(own, normalized, pages, timeout)
