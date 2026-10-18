"""Turn a file into a short digest: metadata plus its FIRST and LAST page only.

Never reads a whole document into the prompt. "Page" means a real page for PDFs and an
equal-sized slice of text (page_chars) from the start and the end for everything else.
"""
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTS = {
    ".txt", ".md", ".rst", ".csv", ".tsv", ".log", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".xml", ".html", ".htm", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
    ".go", ".rs", ".rb", ".php", ".sh", ".ps1", ".bat", ".sql", ".r", ".m", ".tex", ".ipynb",
}


@dataclass
class FileDigest:
    path: Path
    name: str
    ext: str
    size: int
    modified: str
    sha256: str
    first_page: str
    last_page: str
    note: str  # how the content was read, or why it wasn't

    def as_prompt(self, budget_chars: int) -> str:
        """Compact text for the LLM, trimmed so first+last page fit in budget_chars."""
        half = max(0, budget_chars // 2)
        first = self.first_page[:half].strip()
        last = self.last_page[-half:].strip() if self.last_page else ""
        lines = [
            f"name: {self.name}",
            f"type: {self.ext or '(none)'}   size: {self.size} bytes   modified: {self.modified}",
        ]
        if first:
            lines += ["first page:", first]
        if last and last != first:
            lines += ["last page:", last]
        if not first:
            lines.append(f"(content not read: {self.note})")
        return "\n".join(lines)

    def text(self) -> str:
        return f"{self.name} {self.first_page} {self.last_page}"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _head_tail_text(path: Path, n: int):
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(n)
        tail = b""
        if size > 2 * n:
            f.seek(size - n)
            tail = f.read(n)
        elif size > n:
            tail = f.read()
    dec = lambda b: b.decode("utf-8", errors="replace")
    return dec(head), dec(tail)


def _pdf_pages(path: Path, n: int):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = reader.pages
    if not pages:
        return "", "", "pdf has no pages"
    first = (pages[0].extract_text() or "")[:n]
    last = (pages[-1].extract_text() or "")[:n] if len(pages) > 1 else ""
    title = (reader.metadata.title if reader.metadata else None) or ""
    note = f"pdf, {len(pages)} pages" + (f", title: {title}" if title else "")
    return (f"[title: {title}]\n" if title else "") + first, last, note


def _docx_pages(path: Path, n: int):
    import docx

    paras = [p.text for p in docx.Document(str(path)).paragraphs if p.text.strip()]
    text = "\n".join(paras)
    return text[:n], (text[-n:] if len(text) > n else ""), f"docx, {len(paras)} paragraphs"


def _xlsx_pages(path: Path, n: int):
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    parts = [f"sheets: {', '.join(wb.sheetnames)}"]
    ws = wb[wb.sheetnames[0]]
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i >= 8:
            break
        parts.append(" | ".join("" if v is None else str(v) for v in row))
    wb.close()
    return "\n".join(parts)[:n], "", "xlsx, first sheet's first rows"


def make_digest(path: Path, page_chars: int) -> FileDigest:
    st = path.stat()
    ext = path.suffix.lower()
    first = last = ""
    note = "metadata only"
    try:
        if ext == ".pdf":
            first, last, note = _pdf_pages(path, page_chars)
        elif ext == ".docx":
            first, last, note = _docx_pages(path, page_chars)
        elif ext in (".xlsx", ".xlsm"):
            first, last, note = _xlsx_pages(path, page_chars)
        elif ext in TEXT_EXTS:
            first, last = _head_tail_text(path, page_chars)
            note = "text, first and last page"
    except Exception as e:  # unreadable/corrupt/encrypted: fall back to metadata
        note = f"could not read content ({type(e).__name__}: {e})"
    return FileDigest(
        path=path,
        name=path.name,
        ext=ext,
        size=st.st_size,
        modified=time.strftime("%Y-%m-%d", time.localtime(st.st_mtime)),
        sha256=sha256(path),
        first_page=first,
        last_page=last,
        note=note,
    )


def iter_files(root: Path, recursive: bool):
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for f in filenames:
                yield Path(dirpath) / f
    else:
        for p in root.iterdir():
            if p.is_file():
                yield p
