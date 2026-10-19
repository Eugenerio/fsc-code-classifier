"""Crawler tests: httpx.MockTransport for HTTP, a fake resolver for DNS. No network."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from app.ingest import web
from app.ingest.web import (
    check_public_url,
    fetch_site,
    guess_site_from_email_domain,
    normalize_url,
    parse_html,
    rank_links,
    registrable_domain,
)

PUBLIC_IP = "93.184.216.34"
DNS = {"localhost": ["127.0.0.1"], "internal.example": ["10.0.0.5"], "rebind.example": [PUBLIC_IP, "192.168.1.9"]}

HOME = """<!doctype html><html><head><title>Acme Fasteners</title>
<meta name="description" content="Manufacturer of hex bolts and stainless steel fasteners.">
<script type="application/ld+json">{"@type": "Organization", "name": "Acme Fasteners", "description": "Bolts and nuts"}</script>
<style>.x{color:red}</style><script>var tracking = "do not keep";</script></head>
<body>
<header><nav><a href="/">Home</a><a href="/products">Products</a></nav><h1>Acme Fasteners Inc.</h1></header>
<div class="cookie-banner">We use cookies. Accept all.</div>
<main>
  <h2>Our products</h2>
  <ul><li>Hex head cap screws</li><li>Carriage bolts</li></ul>
  <p>We <b>manufacture</b> and stock <a href="/products/nuts">flange nuts</a><a href="/products/washers">lock washers</a>.</p>
  <table><tr><th>Item</th><th>Grade</th></tr><tr><td>Hex bolt</td><td>Grade 8</td></tr></table>
  <img src="/logo.png" alt="Acme logo"><img src="/p1.jpg" alt="Stainless steel U-bolt assortment">
  <a href="/about-us">About us</a> <a href="/privacy-policy">Privacy</a> <a href="/careers">Careers</a>
  <a href="/catalog.pdf">Catalog PDF</a> <a href="https://other-company.com/products">Partner</a>
  <a href="mailto:sales@acme.example">Mail</a> <a href="/products/pressure-control">Pressure control</a>
</main>
<footer>Copyright 2026 Acme. All rights reserved.</footer>
</body></html>"""


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolve(host: str) -> list[str]:
        if web._is_ip(host):
            return [host]
        return DNS.get(host, [PUBLIC_IP])

    monkeypatch.setattr(web, "_resolve", resolve)


def site(pages: dict[str, str], seen: list[str] | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        body = pages.get(request.url.path)
        if body is None:
            return httpx.Response(404, text="nope")
        return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def routed(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- URL helpers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("acme.com", "https://acme.com"),
        ("  HTTP://Acme.COM/Products?id=1#top ", "http://acme.com/Products?id=1"),
        ("https://www.acme.com:8443/a/b", "https://www.acme.com:8443/a/b"),
        ("//acme.com", "https://acme.com"),
        ("https://xn--bcher-kva.example/", "https://xn--bcher-kva.example/"),
        ("https://[2001:db8::1]/x", "https://[2001:db8::1]/x"),
        (None, None),
        ("", None),
        ("   ", None),
        ("file:///etc/passwd", None),
        ("ftp://acme.com", None),
        ("javascript:alert(1)", None),
        ("mailto:bob@acme.com", None),
        ("https://user:pw@acme.com", None),
        ("https://acme.com:notaport", None),
        ("localhost", None),
        ("not a url", None),
    ],
)
def test_normalize_url(raw: str | None, expected: str | None) -> None:
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("hrpartsco.com", "https://hrpartsco.com"),
        ("@Loosco.com", "https://loosco.com"),
        ("bob@acme-industrial.co.uk", "https://acme-industrial.co.uk"),
        ("shop.acme.com", "https://shop.acme.com"),
        ("gmail.com", None),
        ("GMAIL.COM", None),
        ("yahoo.com", None),
        ("yahoo.co.uk", None),
        ("outlook.com", None),
        ("outlook.de", None),
        ("hotmail.fr", None),
        ("live.com", None),
        ("msn.com", None),
        ("icloud.com", None),
        ("aol.com", None),
        ("proton.me", None),
        ("protonmail.com", None),
        ("gmx.net", None),
        ("yandex.ru", None),
        ("mail.ru", None),
        ("sbcglobal.net", None),
        ("liveoak.com", "https://liveoak.com"),  # "live" only matches the whole second-level label
        ("nodot", None),
        ("", None),
        (None, None),
    ],
)
def test_guess_site_from_email_domain(domain: str | None, expected: str | None) -> None:
    assert guess_site_from_email_domain(domain) == expected


def test_registrable_domain() -> None:
    assert registrable_domain("www.acme.com") == "acme.com"
    assert registrable_domain("shop.eu.acme.com") == "acme.com"
    assert registrable_domain("www.acme.co.uk") == "acme.co.uk"
    assert registrable_domain("acme.com") == "acme.com"


# --------------------------------------------------------------------------- SSRF guard


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:8000/admin",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.1.2.3/",
        "http://172.16.0.1/",
        "http://192.168.0.10/",
        "http://100.64.0.1/",  # carrier-grade NAT
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[fe80::1]/",
        "http://internal.example/",  # public-looking name, private A record
        "http://rebind.example/",  # one private address among several is enough to refuse
        "ftp://example.com/",
        "file:///etc/passwd",
        "gopher://example.com/",
    ],
)
async def test_ssrf_guard_blocks_internal_targets(url: str) -> None:
    with pytest.raises(ValueError, match="blocked"):
        await check_public_url(url)


async def test_ssrf_guard_allows_public_addresses() -> None:
    await check_public_url("https://example.com/")
    await check_public_url("http://93.184.216.34/")
    await check_public_url("http://[2606:2800:220:1:248:1893:25c8:1946]/")


async def test_ssrf_guard_reports_unresolvable_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(host: str) -> list[str]:
        raise OSError("nxdomain")

    monkeypatch.setattr(web, "_resolve", fail)
    with pytest.raises(ValueError, match="could not resolve"):
        await check_public_url("https://nowhere.example/")


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1/", "http://localhost/", "http://169.254.169.254/latest/", "http://10.0.0.1/", "http://internal.example/"],
)
async def test_fetch_site_never_sends_requests_to_internal_targets(url: str) -> None:
    seen: list[str] = []
    async with site({"/": HOME}, seen) as client:
        docs = await fetch_site(url, client=client)
    assert seen == []
    assert len(docs) == 1 and not docs[0].ok
    assert docs[0].error and ("blocked" in docs[0].error or "invalid URL" in docs[0].error)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://acme.com/", "javascript:alert(1)", "", "http://"])
async def test_fetch_site_rejects_bad_schemes_and_garbage(url: str) -> None:
    seen: list[str] = []
    async with site({"/": HOME}, seen) as client:
        docs = await fetch_site(url, client=client)
    assert seen == [] and [d.ok for d in docs] == [False] and docs[0].error == "invalid URL"


@pytest.mark.parametrize(
    "target", ["http://169.254.169.254/latest/meta-data/", "http://10.0.0.7/admin", "file:///etc/passwd", "http://localhost:9200/"]
)
async def test_redirect_to_internal_target_is_blocked(target: str) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "www.acme.example":
            return httpx.Response(302, headers={"location": target})
        return httpx.Response(200, text="secret", headers={"content-type": "text/html"})

    async with routed(handler) as client:
        docs = await fetch_site("https://www.acme.example/", client=client)

    assert not docs[0].ok and "blocked" in (docs[0].error or "")
    assert not any(u.startswith(target) for u in seen)  # the internal URL was never requested
    assert seen[0] == "https://www.acme.example/"


async def test_redirects_to_public_hosts_are_followed_and_label_is_final_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "acme.example":
            return httpx.Response(301, headers={"location": "https://www.acme.example/home"})
        return httpx.Response(200, text="<html><body><p>Hello bolts catalog page text.</p></body></html>", headers={"content-type": "text/html"})

    async with routed(handler) as client:
        docs = await fetch_site("https://acme.example", client=client, max_pages=1)
    assert docs[0].ok and docs[0].label == "https://www.acme.example/home"


async def test_redirect_loop_fails_cleanly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    async with routed(handler) as client:
        docs = await fetch_site("https://loop.example/", client=client)
    assert not docs[0].ok and docs[0].error == "too many redirects"


# --------------------------------------------------------------------------- parsing


def test_parse_html_keeps_product_text_and_drops_chrome() -> None:
    text, _ = parse_html(HOME, "https://acme.example/")

    assert text.startswith("Title: Acme Fasteners\nDescription: Manufacturer of hex bolts")
    assert "Structured data: Acme Fasteners: Bolts and nuts" in text
    assert "# Acme Fasteners Inc." in text  # h1 rescued from the stripped <header>
    assert "# Our products" in text
    assert "- Hex head cap screws" in text and "- Carriage bolts" in text
    assert "We manufacture and stock flange nuts lock washers ." in text  # links do not fuse into one word
    assert "Item | Grade" in text and "Hex bolt | Grade 8" in text
    assert "Image captions: Stainless steel U-bolt assortment" in text and "logo" not in text.lower()
    for noise in ("tracking", "color:red", "Copyright", "cookies", "Home"):
        assert noise not in text


def test_parse_html_collects_links_from_navigation_before_it_is_stripped() -> None:
    _, links = parse_html(HOME, "https://acme.example/")
    assert ("https://acme.example/products", "Products") in links
    assert ("https://acme.example/products/nuts", "flange nuts") in links


def test_parse_html_keeps_pages_wrapped_in_one_big_form_or_cookie_class() -> None:
    html = (
        '<html><body class="cookie-enabled"><form id="aspnetForm"><div class="cookie-enabled">'
        + "<p>We machine precision gaskets and seals for pumps. </p>" * 60
        + "</div></form></body></html>"
    )
    text, _ = parse_html(html, "https://asp.example/")
    assert "precision gaskets" in text


def test_parse_html_dedupes_repeated_lines_and_caps_size() -> None:
    html = "<html><body>" + "<p>Repeated line about valves.</p>" * 50 + "".join(f"<p>Unique line {i} " + "x" * 50 + "</p>" for i in range(2000)) + "</body></html>"
    text, _ = parse_html(html)
    assert text.count("Repeated line about valves.") == 1
    assert len(text) <= web.MAX_PAGE_CHARS


def test_parse_html_survives_garbage() -> None:
    assert parse_html(b"\x00\xff not html <<<", "https://x.example/")[0] is not None
    assert parse_html("", "https://x.example/") == ("", [])


def test_rank_links_prefers_product_pages_and_filters_the_rest() -> None:
    links = [
        ("https://acme.example/privacy-policy", "Privacy"),
        ("https://acme.example/blog/2024/post", "Our blog"),
        ("https://acme.example/contact", "Contact us"),
        ("https://acme.example/misc", "Misc"),
        ("https://acme.example/about-us", "About us"),
        ("https://acme.example/products/pressure-control", "Pressure control"),
        ("https://acme.example/products", "Products & Services"),
        ("https://acme.example/products#top", "Products"),  # same page once the fragment is gone
        ("https://acme.example/catalog.pdf", "Catalog"),
        ("https://other.example/products", "Partner"),
        ("mailto:sales@acme.example", "Mail"),
        ("javascript:void(0)", "Menu"),
        ("https://shop.acme.example/capabilities", "Capabilities"),
        ("https://acme.example/", "Home"),
    ]
    ranked = rank_links(links, "https://www.acme.example/", limit=10)

    assert ranked[0] == "https://acme.example/products"
    assert set(ranked[:4]) == {
        "https://acme.example/products",
        "https://acme.example/products/pressure-control",
        "https://shop.acme.example/capabilities",
        "https://acme.example/about-us",
    }
    assert ranked[-1] == "https://acme.example/misc"
    assert not any(bad in u for u in ranked for bad in ("privacy", "blog", "contact", ".pdf", "other.example", "mailto", "javascript"))
    assert "https://acme.example/" not in ranked
    assert len(ranked) == len(set(ranked))
    assert len(rank_links(links, "https://www.acme.example/", limit=2)) == 2


# --------------------------------------------------------------------------- crawling


async def test_fetch_site_crawls_homepage_plus_ranked_internal_pages() -> None:
    pages = {
        "/": HOME,
        "/products": "<html><body><h1>Products</h1><p>Hex bolts, carriage bolts and U-bolts in stock.</p></body></html>",
        "/products/nuts": "<html><body><p>Flange nuts and hex nuts in grades 5 and 8 available.</p></body></html>",
        "/products/washers": "<html><body><p>Lock washers and flat washers.</p></body></html>",
        "/products/pressure-control": "<html><body><p>Pressure control equipment.</p></body></html>",
        "/about-us": "<html><body><p>Family owned since 1955 and located in Texas.</p></body></html>",
    }
    seen: list[str] = []
    async with site(pages, seen) as client:
        docs = await fetch_site("acme.example", max_pages=4, client=client)

    assert len(docs) == 4 and all(d.kind == "website" and d.ok for d in docs)
    assert docs[0].label == "https://acme.example" and "Hex head cap screws" in docs[0].text
    assert len(seen) == 4
    assert not any("privacy" in u or "careers" in u or "other-company" in u or ".pdf" in u for u in seen)
    assert all("Hex head cap screws" not in d.text for d in docs[1:])


async def test_fetch_site_reports_failed_pages_without_raising() -> None:
    pages = {"/": '<html><body><a href="/products">Products</a><a href="/services">Services</a><p>Homepage text here.</p></body></html>', "/products": "<html><body><p>Widgets.</p></body></html>"}
    async with site(pages) as client:
        docs = await fetch_site("https://acme.example/", client=client)

    by_label = {d.label: d for d in docs}
    assert by_label["https://acme.example/products"].ok
    missing = by_label["https://acme.example/services"]
    assert not missing.ok and missing.error == "HTTP 404"


async def test_homepage_failure_returns_single_failed_doc() -> None:
    async with site({}) as client:
        docs = await fetch_site("https://acme.example/", client=client)
    assert len(docs) == 1 and not docs[0].ok and docs[0].error == "HTTP 404"


async def test_falls_back_to_www_sibling_when_one_host_is_broken() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.broken.example":
            raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]")
        return httpx.Response(200, text="<html><body><p>We sell industrial pumps.</p></body></html>", headers={"content-type": "text/html"})

    async with routed(handler) as client:
        docs = await fetch_site("https://www.broken.example/", client=client)
    assert docs[0].ok and docs[0].label == "https://broken.example/"


async def test_connection_errors_become_failed_docs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    async with routed(handler) as client:
        docs = await fetch_site("https://down.example/", client=client)
    assert not docs[0].ok and "ConnectTimeout" in (docs[0].error or "")


async def test_non_html_and_empty_pages_are_flagged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/data":
            return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})
        return httpx.Response(200, text="<html><body><script>app()</script></body></html>", headers={"content-type": "text/html"})

    async with routed(handler) as client:
        pdf = await fetch_site("https://acme.example/data", client=client)
        spa = await fetch_site("https://acme.example/", client=client)
    assert pdf[0].error and "not an HTML page" in pdf[0].error
    assert spa[0].error and "JavaScript" in spa[0].error


async def test_page_size_is_capped() -> None:
    big = "<html><body>" + "<p>Line of valves and pumps text.</p>" * 200_000 + "</body></html>"
    assert len(big) > 3 * web.MAX_BYTES
    async with site({"/": big}) as client:
        docs = await fetch_site("https://acme.example/", client=client)
    assert docs[0].ok and 0 < len(docs[0].text) <= web.MAX_PAGE_CHARS


async def test_sidebar_lines_repeated_across_pages_are_kept_once() -> None:
    sidebar = "<ul><li>Hydraulic hose assemblies</li><li>Ball valves</li></ul>"
    pages = {
        "/": f'<html><body><a href="/products">Products</a><p>Welcome to our shop.</p>{sidebar}</body></html>',
        "/products": f"<html><body><p>Everything we stock.</p>{sidebar}</body></html>",
    }
    async with site(pages) as client:
        docs = await fetch_site("https://acme.example/", client=client)
    assert "- Ball valves" in docs[0].text and "- Ball valves" not in docs[1].text
