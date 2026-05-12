import asyncio
from io import BytesIO
from pathlib import Path

from starlette.datastructures import UploadFile

from app.storage_uploads import persist_ocr_uploads, sniff_is_pdf


def test_sniff_pdf_by_magic_when_untyped() -> None:
    raw = b"%PDF-1.4\n%eof"
    assert sniff_is_pdf(raw, "blob", "application/octet-stream")


def test_persist_writes_unique_disk_names_when_same_origin_name(tmp_path: Path) -> None:
    async def go() -> None:
        f1 = UploadFile(BytesIO(b"aa"), filename="a.pdf")
        f2 = UploadFile(BytesIO(b"bb"), filename="a.pdf")
        _, batch_dir, saved = await persist_ocr_uploads(tmp_path, [f1, f2])
        assert len(saved) == 2
        p0, _, _ = saved[0]
        p1, _, _ = saved[1]
        assert p0.name == "a.pdf"
        assert p1.read_bytes() == b"bb"
        assert batch_dir.samefile(tmp_path / batch_dir.name)
        assert sorted(p.name for p in batch_dir.iterdir()) == ["a.pdf", "a_2.pdf"]

    asyncio.run(go())
