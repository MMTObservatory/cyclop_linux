import json
import logging
import struct
import urllib.request
import zlib

import numpy as np

from cyclop import config
from cyclop.camera import SimCamera
from cyclop.monitor import Monitor
from cyclop.web import LogBuffer, WebServer, WebState, png_gray, read_history

LINES = """\
10/1/2026 12:56:49 PM | 10/1/2026 5:56:49 AM | 2461315.0394607 | 1284.9 | 0.91 | 125.1
10/1/2026 12:57:41 PM | 10/1/2026 5:57:41 AM | 2461315.0400601 | 1221.6 | 0.76 | 148.3
"""


def decode_png(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        assert zlib.crc32(kind + body) == struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0]
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    w, h = ihdr[:2]
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, w + 1)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:]


def test_png_roundtrip():
    img = np.random.default_rng(1).integers(0, 256, (37, 53), dtype=np.uint8)
    assert (decode_png(png_gray(img)) == img).all()


def test_read_history(tmp_path):
    f = tmp_path / "Seeing_Data.txt"
    f.write_text(LINES + "garbage line\n")
    now = (2461315.0400601 - 2440587.5) * 86400 + 3600
    rows = read_history(f, days=1, now=now)
    assert len(rows) == 2
    t, zen, flux, r0 = rows[1]
    assert (zen, flux, r0) == (0.76, 1221.6, 148.3)
    assert abs(t - (now - 3600)) < 1e-3
    assert read_history(f, days=1, now=now + 2 * 86400) == []
    assert read_history(tmp_path / "missing.txt") == []


def run_monitor(tmp_path):
    cfg = config.load()
    cfg['measurement']['n_samples'] = 300
    cam = SimCamera(seed=3)
    mon = Monitor(cfg, lambda: cam, sleep=lambda s: None)
    (tmp_path / "Seeing_Data.txt").write_text(LINES)
    logbuf = LogBuffer()
    logbuf.emit(logging.makeLogRecord({'msg': 'hello', 'levelname': 'INFO'}))
    # the same file twice counts once; a missing file is skipped
    files = [tmp_path / "Seeing_Data.txt", tmp_path / "Seeing_Data.txt", tmp_path / "none.txt"]
    state = WebState(mon, logbuf, history_files=files, history_days=100000)
    mon.run(ignore_sun=True, max_results=2)
    mon.camera = cam          # run() closes the camera on exit; keep its state for the endpoints
    return mon, state


def test_state_endpoints(tmp_path):
    mon, state = run_monitor(tmp_path)
    st = json.loads(json.dumps(state.status(), allow_nan=False))
    assert st['state'] == "measurements pending"
    assert st['n_results'] == 2
    assert st['frame']['width'] == 640 and sum(st['frame']['histogram']) == 640 * 480
    assert st['last']['accepted'] and st['last']['r0_local'] < st['last']['r0']
    assert abs(st['star']['x'] - cam_x(mon)) < 2

    h = state.history_rows(days=100000)
    assert len(h['t']) == 4 and h['seeing'][:2] == [0.91, 0.76]

    m = state.motion()
    assert len(m['t']) == 300 and m['t'][0] == 0

    png, factor = state.frame_png()
    assert factor == 1 and decode_png(png).shape == (480, 640)
    zoom, origin = state.zoom_png(8)
    assert decode_png(zoom).shape == (17, 17)
    assert state.logbuf.since(-1)[0]['msg'] == 'hello'


def cam_x(mon):
    return mon.camera.true_position()[0]


def test_http_server(tmp_path):
    mon, state = run_monitor(tmp_path)
    web = WebServer(state, host="127.0.0.1", port=0).start()
    try:
        def get(path):
            with urllib.request.urlopen(web.url + path, timeout=5) as r:
                return r.status, r.headers.get_content_type(), r.read()
        code, ctype, body = get("")
        assert code == 200 and ctype == "text/html" and b"Latest Seeing Motion Data" in body
        assert json.loads(get("api/status")[2])['n_results'] == 2
        assert len(json.loads(get("api/history?days=100000")[2])['t']) == 4
        assert get("api/frame.png?stretch=linear")[1] == "image/png"
        assert json.loads(get("api/log?after=-1")[2])[0]['msg'] == 'hello'
        try:
            get("nope")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        web.stop()
