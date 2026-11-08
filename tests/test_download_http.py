"""Pobieranie przez HTTP na lokalnym serwerze z obsługą Range (bez internetu)."""

import hashlib
import http.server
import os
import struct
import threading
import time
import urllib.error
import zipfile

import pytest

from alpr import download as dl
from alpr.download import Archive, RemoteFile, _target, download, fetch_member, file_hash, range_get


class Handler(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    ranges = True
    fail = 0
    retry_after: str | None = None
    log: list[tuple[str, str, str | None]] = []

    def log_message(self, *args):
        pass

    def _error(self) -> bool:
        if Handler.fail <= 0:
            return False
        Handler.fail -= 1
        self.send_response(503)
        if Handler.retry_after:
            self.send_header("Retry-After", Handler.retry_after)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def do_HEAD(self):
        Handler.log.append(("HEAD", self.path, None))
        data = Handler.files.get(self.path)
        self.send_response(200 if data is not None else 404)
        self.send_header("Content-Length", str(len(data or b"")))
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        Handler.log.append(("GET", self.path, rng))
        if self._error():
            return
        if self.path == "/przekierowanie":
            self.send_response(302)
            self.send_header("Location", "/duzy.bin")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = Handler.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        if rng and Handler.ranges:
            start, end = (int(v) for v in rng.removeprefix("bytes=").split("-"))
            end = min(end, len(data) - 1)
            body = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    Handler.files, Handler.ranges, Handler.fail, Handler.retry_after, Handler.log = {}, True, 0, None, []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, args=(0.01,), daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


def payload(n: int) -> bytes:
    return bytes((i * 7 + i // 256) % 256 for i in range(n))


def test_range_get_wlacznie_z_koncem(server):
    Handler.files["/a"] = payload(100)
    assert range_get(f"{server}/a", 10, 19) == payload(100)[10:20]


def test_range_get_ponawia_z_retry_after(server, no_sleep):
    Handler.files["/a"] = b"0123456789"
    Handler.fail, Handler.retry_after = 2, "3"
    assert range_get(f"{server}/a", 0, 3) == b"0123"
    assert no_sleep == [3, 3]


def test_range_get_backoff_wykladniczy_bez_retry_after(server, no_sleep):
    Handler.files["/a"] = b"0123456789"
    Handler.fail = 3
    assert range_get(f"{server}/a", 0, 0) == b"0"
    assert no_sleep == [1, 2, 4]


def test_range_get_retry_after_ograniczone_do_120s(server, no_sleep):
    Handler.files["/a"] = b"x"
    Handler.fail, Handler.retry_after = 1, "9999"
    range_get(f"{server}/a", 0, 0)
    assert no_sleep == [120]


def test_range_get_poddaje_sie_po_limicie(server, no_sleep):
    Handler.files["/a"] = b"x"
    Handler.fail = 10
    with pytest.raises(urllib.error.HTTPError):
        range_get(f"{server}/a", 0, 0, retries=3)
    assert len(no_sleep) == 2


def test_download_maly_plik(server, tmp_path):
    Handler.files["/maly.bin"] = payload(5000)
    download(f"{server}/maly.bin", tmp_path / "x" / "maly.bin")
    assert (tmp_path / "x" / "maly.bin").read_bytes() == payload(5000)
    assert not (tmp_path / "x" / "maly.part").exists()


def test_download_serwer_bez_range(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CHUNK", 1000)
    Handler.files["/duzy.bin"] = payload(5500)
    Handler.ranges = False
    download(f"{server}/duzy.bin", tmp_path / "duzy.bin")
    assert (tmp_path / "duzy.bin").read_bytes() == payload(5500)


def test_download_rownolegle_kawalki(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CHUNK", 1000)
    Handler.files["/duzy.bin"] = payload(5500)
    download(f"{server}/duzy.bin", tmp_path / "duzy.bin")
    assert (tmp_path / "duzy.bin").read_bytes() == payload(5500)
    assert not (tmp_path / "duzy.chunks").exists() and not (tmp_path / "duzy.part").exists()
    chunk_requests = {r for m, p, r in Handler.log if m == "GET" and r != "bytes=0-0"}
    assert chunk_requests == {f"bytes={s}-{min(s + 1000, 5500) - 1}" for s in range(0, 5500, 1000)}


def test_download_wznawia_pobrane_kawalki(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CHUNK", 1000)
    data = payload(3500)
    Handler.files["/duzy.bin"] = data
    chunks = tmp_path / "duzy.chunks"
    chunks.mkdir()
    (chunks / f"{0:013d}").write_bytes(data[:1000])
    (chunks / f"{2000:013d}").write_bytes(data[2000:3000])
    (chunks / f"{1000:013d}.tmp").write_bytes(b"przerwany")
    download(f"{server}/duzy.bin", tmp_path / "duzy.bin")
    assert (tmp_path / "duzy.bin").read_bytes() == data
    fetched = sorted(r for m, p, r in Handler.log if m == "GET" and r != "bytes=0-0")
    assert fetched == ["bytes=1000-1999", "bytes=3000-3499"]


def test_download_po_przekierowaniu_uzywa_koncowego_url(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CHUNK", 1000)
    Handler.files["/duzy.bin"] = payload(2500)
    download(f"{server}/przekierowanie", tmp_path / "d.bin")
    assert (tmp_path / "d.bin").read_bytes() == payload(2500)
    assert all(p == "/duzy.bin" for m, p, r in Handler.log if m == "GET" and r != "bytes=0-0")


def test_download_404_to_blad(server, tmp_path):
    with pytest.raises(urllib.error.HTTPError):
        download(f"{server}/brak", tmp_path / "brak.bin")
    assert not (tmp_path / "brak.bin").exists()


def make_zip(path, files: dict[str, bytes], compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path


FILES = {"a/1.txt": b"jeden" * 100, "a/2.bin": payload(3000), "b.json": b'{"x": 1}'}


@pytest.fixture
def remote_zip(server, tmp_path):
    path = make_zip(tmp_path / "z.zip", FILES)
    Handler.files["/z.zip"] = path.read_bytes()
    return f"{server}/z.zip", path


def test_remote_file_jak_lokalny(remote_zip):
    url, path = remote_zip
    f = RemoteFile(url)
    assert f.size == path.stat().st_size
    assert f.seek(-10, 2) == f.size - 10
    assert f.read(100) == path.read_bytes()[-10:]
    assert f.read(5) == b""
    f.seek(3)
    f.seek(2, 1)
    assert f.tell() == 5 and f.read(4) == path.read_bytes()[5:9]


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_fetch_member(server, tmp_path, compression):
    path = make_zip(tmp_path / "z.zip", FILES, compression)
    Handler.files["/z.zip"] = path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            assert fetch_member(f"{server}/z.zip", info) == FILES[info.filename]


def test_fetch_member_duze_pole_extra_dociaga_reszte(server, tmp_path):
    info = zipfile.ZipInfo("x.txt")
    info.extra = struct.pack("<HH", 0xCAFE, 4000) + bytes(4000)
    with zipfile.ZipFile(tmp_path / "z.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(info, b"dane" * 50)
    Handler.files["/z.zip"] = (tmp_path / "z.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "z.zip") as zf:
        assert fetch_member(f"{server}/z.zip", zf.getinfo("x.txt")) == b"dane" * 50
    assert len([1 for m, _, _ in Handler.log if m == "GET"]) == 2


def test_fetch_member_zla_suma_crc(remote_zip):
    url, path = remote_zip
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo("b.json")
    info.CRC ^= 1
    with pytest.raises(ValueError, match="CRC"):
        fetch_member(url, info)


def test_fetch_member_nieobslugiwana_kompresja(remote_zip):
    url, path = remote_zip
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo("b.json")
    info.compress_type = zipfile.ZIP_BZIP2
    with pytest.raises(ValueError, match="kompresja"):
        fetch_member(url, info)


def test_archive_zdalne_i_lokalne_daja_to_samo(remote_zip):
    url, path = remote_zip
    remote, local = Archive(url), Archive(url, path)
    assert sorted(remote.zf.namelist()) == sorted(FILES)
    infos = remote.zf.infolist()
    done = []
    assert list(remote.read_many(infos, lambda: done.append(1))) == [FILES[i.filename] for i in infos]
    assert list(local.read_many(local.zf.infolist())) == [FILES[i.filename] for i in local.zf.infolist()]
    assert len(done) == len(infos)


@pytest.mark.parametrize(
    ("name", "members", "expected"),
    [
        ("annotations.xml", {"annotations.xml": "a.xml"}, "a.xml"),
        ("photos/photos/x/1.jpg", {"photos/photos/": "photos/"}, "photos/x/1.jpg"),
        ("inne.txt", {"photos/photos/": "photos/"}, None),
        ("cokolwiek/x", {"": "out/"}, "out/cokolwiek/x"),
        ("ocr/ocr/a.png", {"ocr/ocr/": "ocr/", "": ""}, "ocr/a.png"),
        ("photos", {"photos/": "p/"}, None),
        ("annotations.xml.bak", {"annotations.xml": "a.xml"}, None),
    ],
)
def test_target(name, members, expected):
    assert _target(name, members) == expected


def test_file_hash(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    assert file_hash(tmp_path / "a") == hashlib.sha256(b"abc").hexdigest()
    assert file_hash(tmp_path / "a", "md5") == hashlib.md5(b"abc").hexdigest()


def test_cached_md5_przelicza_po_zmianie_pliku(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    (tmp_path / "a.bin.md5").write_text("stare")
    old = path.stat().st_mtime
    os.utime(tmp_path / "a.bin.md5", (old - 10, old - 10))
    assert dl.cached_md5(path) == hashlib.md5(b"abc").hexdigest()
    assert (tmp_path / "a.bin.md5").read_text() == hashlib.md5(b"abc").hexdigest()
