import hashlib
import zipfile

from alpr.build.sources import PolishPlates
from alpr.download import cached_md5, extract, missing_files, verify_sums


def make_zip(path, names):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.writestr(name, name.encode())
    return path


def test_extract_mapuje_prefiksy(tmp_path):
    zip_path = make_zip(tmp_path / "a.zip", ["photos/photos/x/1.jpg", "annotations.xml", "inne.txt"])
    extract(zip_path, tmp_path / "out", {"photos/photos/": "photos/", "annotations.xml": "annotations.xml"})
    files = sorted(p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert files == ["annotations.xml", "photos/x/1.jpg"]
    assert (tmp_path / "out/photos/x/1.jpg").read_text() == "photos/photos/x/1.jpg"


def test_extract_wszystko_z_podniesieniem_ocr(tmp_path):
    zip_path = make_zip(tmp_path / "a.zip", ["plates.csv", "ocr/ocr/test/a.png", "detection/images/test/b.jpg"])
    extract(zip_path, tmp_path / "out", {"ocr/ocr/": "ocr/", "": ""})
    assert all((tmp_path / "out" / p).exists() for p in ["plates.csv", "ocr/test/a.png", "detection/images/test/b.jpg"])


def test_cached_md5(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    assert cached_md5(path) == hashlib.md5(b"abc").hexdigest()
    (tmp_path / "a.bin.md5").write_text("zapamietane")
    assert cached_md5(path) == "zapamietane"


def test_verify_sums(tmp_path):
    (tmp_path / "ok.jpg").write_bytes(b"1")
    (tmp_path / "zly.jpg").write_bytes(b"2")
    sums = tmp_path / "SUMS"
    sums.write_text(
        f"{hashlib.sha256(b'1').hexdigest()}  ok.jpg\n"
        f"{hashlib.sha256(b'1').hexdigest()}  zly.jpg\n"
        f"{hashlib.sha256(b'3').hexdigest()}  brak.jpg\n"
    )
    assert verify_sums(sums, tmp_path) == ["zly.jpg", "brak.jpg"]


def test_source_verify(tmp_path):
    source = PolishPlates(tmp_path)
    assert source.verify() is None
    (source.photos / "a").mkdir(parents=True)
    (source.photos / "a" / "b c.jpg").write_bytes(b"x")
    (source.dir / "SHA256SUMS").write_text(f"{hashlib.sha256(b'x').hexdigest()}  a/b c.jpg\n")
    assert source.has_photos()
    assert source.verify() == []


def test_missing_files(tmp_path):
    rows = ["sample_type,image,crop", "full_photo,img/a.jpg,ocr/a.png", "plate_crop,,ocr/b.png"]
    (tmp_path / "plates.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "ocr").mkdir()
    (tmp_path / "ocr" / "a.png").touch()
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.jpg").touch()
    df, missing = missing_files(tmp_path)
    assert (len(df), missing) == (2, ["ocr/b.png"])
