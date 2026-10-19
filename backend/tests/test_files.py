"""File extraction: real pypdf on a hand-built PDF, fake Gemini for OCR."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from app.ingest.files import extract_file
from app.schemas import UploadedFile

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


class FakeGemini:
    def __init__(self, reply: str = "OCR TEXT", fail: bool = False) -> None:
        self.reply, self.fail = reply, fail
        self.calls: list[tuple[bytes, str, str]] = []

    async def ocr(self, data: bytes, mime_type: str, filename: str) -> str:
        self.calls.append((data, mime_type, filename))
        if self.fail:
            raise RuntimeError("quota")
        return self.reply


def make_pdf(lines: list[str]) -> bytes:
    """Minimal valid single-page PDF (Helvetica) with an exact xref table."""
    escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    stream = "BT /F1 12 Tf 14 TL 50 740 Td " + " T* ".join(f"({line}) Tj" for line in escaped) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def upload(name: str, data: bytes, content_type: str | None = None) -> UploadedFile:
    return UploadedFile(filename=name, content_type=content_type, data=data)


LONG_LINES = [f"Line {i}: we manufacture wireline pressure control equipment and downhole tools for oil wells." for i in range(6)]


# --------------------------------------------------------------------------- PDF


async def test_text_pdf_is_read_with_pypdf_and_never_sent_to_ocr() -> None:
    gemini = FakeGemini()
    doc = await extract_file(upload("catalog.pdf", make_pdf(LONG_LINES), "application/pdf"), gemini)  # type: ignore[arg-type]

    assert doc.ok and doc.kind == "file" and doc.label == "catalog.pdf"
    assert "wireline pressure control equipment" in doc.text and "Line 5" in doc.text
    assert doc.meta == {"ocr": False, "pages": 1}
    assert gemini.calls == []


async def test_text_pdf_works_without_gemini() -> None:
    doc = await extract_file(upload("catalog.pdf", make_pdf(LONG_LINES), "application/pdf"), None)
    assert doc.ok and "Line 0" in doc.text


async def test_pdf_type_is_detected_from_content_not_declared_type() -> None:
    doc = await extract_file(upload("download", make_pdf(LONG_LINES), "application/octet-stream"), None)
    assert doc.ok and doc.meta["pages"] == 1


async def test_scanned_pdf_goes_to_ocr_when_gemini_is_available() -> None:
    gemini = FakeGemini("Scanned: downhole pumps")
    pdf = make_pdf(["Page 1"])  # far below 200 chars per page => treated as scanned
    doc = await extract_file(upload("scan.pdf", pdf, "application/pdf"), gemini)  # type: ignore[arg-type]

    assert doc.ok and doc.text == "Scanned: downhole pumps" and doc.meta == {"ocr": True, "pages": 1}
    assert gemini.calls == [(pdf, "application/pdf", "scan.pdf")]


async def test_scanned_pdf_without_gemini_keeps_whatever_text_exists() -> None:
    doc = await extract_file(upload("scan.pdf", make_pdf(["Hex bolts"]), "application/pdf"), None)
    assert doc.ok and doc.text == "Hex bolts" and doc.meta["ocr"] is False


async def test_ocr_failure_falls_back_to_text_layer() -> None:
    doc = await extract_file(upload("scan.pdf", make_pdf(["Hex bolts"]), "application/pdf"), FakeGemini(fail=True))  # type: ignore[arg-type]
    assert doc.ok and doc.text == "Hex bolts"


async def test_pdf_without_text_layer_and_without_gemini_explains_why() -> None:
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    buffer = io.BytesIO()
    writer.write(buffer)

    doc = await extract_file(upload("blank.pdf", buffer.getvalue(), "application/pdf"), None)
    assert not doc.ok and "GEMINI_API_KEY" in (doc.error or "")


async def test_pdf_without_text_layer_and_failing_ocr_is_an_error() -> None:
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    buffer = io.BytesIO()
    writer.write(buffer)

    doc = await extract_file(upload("blank.pdf", buffer.getvalue(), "application/pdf"), FakeGemini(fail=True))  # type: ignore[arg-type]
    assert not doc.ok and "OCR failed" in (doc.error or "")


async def test_encrypted_pdf_is_reported_not_raised() -> None:
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    writer.encrypt("s3cret")
    buffer = io.BytesIO()
    writer.write(buffer)

    doc = await extract_file(upload("locked.pdf", buffer.getvalue(), "application/pdf"), None)
    assert not doc.ok and "password" in (doc.error or "")


@pytest.mark.parametrize("data", [b"%PDF-1.7\nthis is not a real pdf", b"%PDF-", make_pdf(LONG_LINES)[:200]])
async def test_corrupt_pdf_is_reported_not_raised(data: bytes) -> None:
    doc = await extract_file(upload("bad.pdf", data, "application/pdf"), None)
    assert not doc.ok and doc.error


# --------------------------------------------------------------------------- images


@pytest.mark.parametrize(
    ("name", "data", "declared", "mime"),
    [
        ("label.png", PNG, "image/png", "image/png"),
        ("photo.jpg", JPEG, "image/jpeg", "image/jpeg"),
        ("blob", PNG, "application/octet-stream", "image/png"),  # sniffed
        ("mislabeled.txt", JPEG, "text/plain", "image/jpeg"),  # content beats the declared type
        ("x.webp", b"RIFF\x00\x00\x00\x00WEBPVP8 ", None, "image/webp"),
        ("x.gif", b"GIF89a" + b"\x00" * 16, "image/gif", "image/gif"),
    ],
)
async def test_images_are_ocrd(name: str, data: bytes, declared: str | None, mime: str) -> None:
    gemini = FakeGemini("Nameplate: centrifugal pump 3 HP")
    doc = await extract_file(upload(name, data, declared), gemini)  # type: ignore[arg-type]

    assert doc.ok and doc.text == "Nameplate: centrifugal pump 3 HP" and doc.meta == {"ocr": True, "pages": 1}
    assert gemini.calls == [(data, mime, name)]


async def test_image_without_gemini_needs_api_key() -> None:
    doc = await extract_file(upload("label.png", PNG, "image/png"), None)
    assert not doc.ok and doc.error == "needs GEMINI_API_KEY for OCR"


async def test_image_with_empty_ocr_is_an_error() -> None:
    doc = await extract_file(upload("label.png", PNG, "image/png"), FakeGemini("  \n"))  # type: ignore[arg-type]
    assert not doc.ok and "no text" in (doc.error or "")


async def test_image_ocr_exception_never_escapes() -> None:
    doc = await extract_file(upload("label.png", PNG, "image/png"), FakeGemini(fail=True))  # type: ignore[arg-type]
    assert not doc.ok and "RuntimeError" in (doc.error or "")


# --------------------------------------------------------------------------- text-ish and unsupported


@pytest.mark.parametrize(
    ("name", "content_type", "raw"),
    [
        ("notes.txt", "text/plain", "We sell hex bolts."),
        ("readme.md", "text/markdown", "# Products\n- hex bolts"),
        ("export.csv", "text/csv", "sku,name\n1,hex bolt"),
        ("data.json", "application/json", '{"products": ["hex bolt"]}'),
        ("catalog.md", "application/octet-stream", "hex bolt"),  # extension decides when the browser gives up
    ],
)
async def test_text_files_are_decoded(name: str, content_type: str, raw: str) -> None:
    doc = await extract_file(upload(name, raw.encode(), content_type), None)
    assert doc.ok and doc.text == raw and doc.meta == {"ocr": False, "pages": 1}


async def test_untyped_blob_is_refused_rather_than_guessed() -> None:
    doc = await extract_file(upload("blob", b"hex bolt", "application/octet-stream"), None)
    assert not doc.ok and "unsupported" in (doc.error or "")


async def test_text_decoding_is_lenient() -> None:
    bom = await extract_file(upload("a.txt", b"\xef\xbb\xbfhex bolts", "text/plain"), None)
    bad = await extract_file(upload("b.txt", b"hex \xff\xfe bolts\x00", "text/plain"), None)
    assert bom.text == "hex bolts"
    assert bad.ok and bad.text.startswith("hex ") and bad.text.endswith(" bolts") and "\x00" not in bad.text


async def test_html_files_are_reduced_to_text() -> None:
    html = b"<html><head><title>Acme</title><script>x()</script></head><body><h1>Pumps</h1><p>Centrifugal pumps.</p></body></html>"
    doc = await extract_file(upload("page.html", html, "text/html"), None)
    assert doc.ok and "# Pumps" in doc.text and "Centrifugal pumps." in doc.text and "<" not in doc.text and "x()" not in doc.text


async def test_empty_file_is_an_error() -> None:
    doc = await extract_file(upload("empty.txt", b"  \n", "text/plain"), None)
    assert not doc.ok and doc.error == "file is empty"


@pytest.mark.parametrize(
    ("name", "data", "content_type"),
    [
        ("setup.exe", b"MZ\x90\x00", "application/x-msdownload"),
        ("archive.zip", b"PK\x03\x04", "application/zip"),
        ("sheet.xlsx", b"PK\x03\x04", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("movie.mp4", b"\x00\x00\x00\x18ftyp", "video/mp4"),
    ],
)
async def test_unsupported_types_are_clear_errors(name: str, data: bytes, content_type: str) -> None:
    doc = await extract_file(upload(name, data, content_type), FakeGemini())  # type: ignore[arg-type]
    assert not doc.ok and (doc.error or "").startswith("unsupported file type")
    assert doc.label == name
