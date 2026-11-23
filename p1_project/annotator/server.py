#!/usr/bin/env python3
"""Local web annotator for the P1 gold set (Praat-TextGrid correction).

Run:  python server.py   (or double-click "Start Annotator.bat"), then open http://127.0.0.1:8624
Only the Python standard library is used.

Folder layout (relative to gold/):
    pack/<song_id>/{vocals.wav, mix.wav, meta.json, <song_id>.TextGrid}   never modified
    corrected/<song_id>.TextGrid                                          your saved output
    .annotator/drafts/<song_id>.json          autosaved work in progress
    .annotator/history/<song_id>/<time>.TextGrid   backup of every previously saved version
    .annotator/state.json                     per song: done flag, notes, listened coverage, time
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.dirname(HERE)
PACK = os.path.join(GOLD, "pack")
CORR = os.path.join(GOLD, "corrected")
WORK = os.path.join(GOLD, ".annotator")
DRAFTS = os.path.join(WORK, "drafts")
HIST = os.path.join(WORK, "history")
STATE = os.path.join(WORK, "state.json")
STATIC = os.path.join(HERE, "static")
PRAAT = os.path.join(GOLD, "Praat.exe")
PORT = int(os.environ.get("ANNOTATOR_PORT", "8624"))
REQUIRED_TIERS = ["words", "syllables", "vocables", "singers"]
LOCK = threading.Lock()


def ensure_dirs():
    for d in (CORR, DRAFTS, HIST):
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------- TextGrid IO
def read_text_any(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


_TOK = re.compile(r'"((?:[^"]|"")*)"|(<exists>|<absent>)|\[[^\]]*\]|'
                  r'(?<![\w.])(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)(?![\w])|[A-Za-z_][\w?]*|\S')


def parse_textgrid(text):
    """Praat long *and* short text TextGrids ->
    {xmin, xmax, tiers:[{name, kind:'interval'|'point', xmin, xmax, items}]}"""
    toks = []
    for m in _TOK.finditer(text):
        if m.group(1) is not None:
            toks.append(("s", m.group(1).replace('""', '"')))
        elif m.group(2) is not None:
            toks.append(("f", m.group(2)))
        elif m.group(3) is not None:
            toks.append(("n", float(m.group(3))))
    pos = 0

    def nxt(kind):
        nonlocal pos
        while pos < len(toks) and toks[pos][0] != kind:
            pos += 1
        if pos >= len(toks):
            raise ValueError("TextGrid ended unexpectedly")
        v = toks[pos][1]
        pos += 1
        return v

    if nxt("s") != "ooTextFile" or nxt("s") != "TextGrid":
        raise ValueError("not a TextGrid text file")
    xmin, xmax = nxt("n"), nxt("n")
    if pos < len(toks) and toks[pos] == ("f", "<absent>"):
        return {"xmin": xmin, "xmax": xmax, "tiers": []}
    ntiers = int(nxt("n"))
    tiers = []
    for _ in range(ntiers):
        cls, name = nxt("s"), nxt("s")
        txmin, txmax, n = nxt("n"), nxt("n"), int(nxt("n"))
        items = []
        if cls == "IntervalTier":
            for _ in range(n):
                a, b, t = nxt("n"), nxt("n"), nxt("s")
                items.append({"a": a, "b": b, "t": t})
            tiers.append({"name": name, "kind": "interval", "xmin": txmin, "xmax": txmax, "items": items})
        else:
            for _ in range(n):
                t, mark = nxt("n"), nxt("s")
                items.append({"t": t, "mark": mark})
            tiers.append({"name": name, "kind": "point", "xmin": txmin, "xmax": txmax, "items": items})
    return {"xmin": xmin, "xmax": xmax, "tiers": tiers}


def fnum(x):
    s = "%.6f" % x
    return "0" if float(s) == 0 else s


def q(s):
    return '"' + s.replace('"', '""') + '"'


def write_textgrid(tg):
    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
         "xmin = %s " % fnum(tg["xmin"]), "xmax = %s " % fnum(tg["xmax"]), "tiers? <exists> ",
         "size = %d " % len(tg["tiers"]), "item []: "]
    for i, t in enumerate(tg["tiers"], 1):
        L.append("    item [%d]:" % i)
        if t["kind"] == "interval":
            L += ['        class = "IntervalTier" ', "        name = %s " % q(t["name"]),
                  "        xmin = %s " % fnum(t["xmin"]), "        xmax = %s " % fnum(t["xmax"]),
                  "        intervals: size = %d " % len(t["items"])]
            for j, it in enumerate(t["items"], 1):
                L += ["        intervals [%d]:" % j, "            xmin = %s " % fnum(it["a"]),
                      "            xmax = %s " % fnum(it["b"]), "            text = %s " % q(it["t"])]
        else:
            L += ['        class = "TextTier" ', "        name = %s " % q(t["name"]),
                  "        xmin = %s " % fnum(t["xmin"]), "        xmax = %s " % fnum(t["xmax"]),
                  "        points: size = %d " % len(t["items"])]
            for j, it in enumerate(t["items"], 1):
                L += ["        points [%d]:" % j, "            number = %s " % fnum(it["t"]),
                      "            mark = %s " % q(it["mark"])]
    return "\n".join(L) + "\n"


def check_textgrid(tg):
    if not tg["xmax"] > tg["xmin"]:
        raise ValueError("TextGrid xmax must be > xmin")
    for t in tg["tiers"]:
        it = t["items"]
        if t["kind"] == "point":
            for k in range(1, len(it)):
                if not it[k]["t"] > it[k - 1]["t"]:
                    raise ValueError("point tier %s: points not in increasing time order at %.3f s" % (t["name"], it[k]["t"]))
            continue
        if not it:
            raise ValueError("tier %s has no intervals" % t["name"])
        if abs(it[0]["a"] - t["xmin"]) > 1e-6 or abs(it[-1]["b"] - t["xmax"]) > 1e-6:
            raise ValueError("tier %s does not cover its full time range" % t["name"])
        for k, x in enumerate(it):
            if not x["b"] > x["a"]:
                raise ValueError("tier %s interval %d has zero/negative length at %.3f s" % (t["name"], k + 1, x["a"]))
            if k and abs(x["a"] - it[k - 1]["b"]) > 1e-6:
                raise ValueError("tier %s has a gap/overlap at %.3f s" % (t["name"], x["a"]))


# ----------------------------------------------------------------- helpers
def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {}


def song_ids():
    if not os.path.isdir(PACK):
        return []
    return sorted(d for d in os.listdir(PACK) if os.path.isfile(os.path.join(PACK, d, d + ".TextGrid")))


def valid_id(sid):
    return bool(re.fullmatch(r"[\w\-. ]+", sid or "")) and sid in song_ids()


def mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else None


def song_meta(sid):
    try:
        return json.load(open(os.path.join(PACK, sid, "meta.json"), encoding="utf-8"))
    except Exception:
        return {"song_id": sid}


def song_summary(sid, st):
    tc, td = mtime(os.path.join(CORR, sid + ".TextGrid")), mtime(os.path.join(DRAFTS, sid + ".json"))
    m, s = song_meta(sid), st.get(sid, {})
    return {"id": sid, "title": m.get("title") or sid, "year": sid[:4] if sid[:4].isdigit() else "",
            "saved_at": tc, "draft_at": td, "draft_newer": bool(td and (not tc or td > tc + 0.5)),
            "done": bool(s.get("done")), "issues": s.get("issues"), "listened": s.get("listened_pct"),
            "seconds": s.get("seconds", 0)}


# ----------------------------------------------------------------- HTTP
class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=STATIC, **k)

    def log_message(self, fmt, *args):
        if self.command == "POST" and "/api/save" in (self.path or ""):
            sys.stderr.write("[%s] saved %s\n" % (time.strftime("%H:%M:%S"), unquote(self.path.split("/")[-1])))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        try:
            if p == "/api/songs":
                st = load_state()
                return self.send_json({"songs": [song_summary(s, st) for s in song_ids()],
                                       "corrected_dir": CORR, "praat": os.path.exists(PRAAT)})
            m = re.fullmatch(r"/api/song/(.+)", p)
            if m:
                return self.get_song(m.group(1))
            m = re.fullmatch(r"/audio/(.+)/(vocals|mix)\.wav", p)
            if m:
                if not valid_id(m.group(1)):
                    return self.send_error(404)
                data = open(os.path.join(PACK, m.group(1), m.group(2) + ".wav"), "rb").read()
                n, a, b = len(data), 0, len(data) - 1
                rng = re.fullmatch(r"bytes=(\d*)-(\d*)", (self.headers.get("Range") or "").strip())
                if rng and (rng.group(1) or rng.group(2)):
                    if rng.group(1):
                        a = int(rng.group(1)); b = min(int(rng.group(2)), n - 1) if rng.group(2) else n - 1
                    else:
                        a = max(0, n - int(rng.group(2)))
                    if a >= n or a > b:
                        self.send_response(416)
                        self.send_header("Content-Range", "bytes */%d" % n)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, n))
                else:
                    self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(b - a + 1))
                self.end_headers()
                try:
                    self.wfile.write(data[a:b + 1])
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                return
        except Exception as e:
            return self.send_json({"error": str(e)}, 500)
        return super().do_GET()

    def get_song(self, sid):
        if not valid_id(sid):
            return self.send_json({"error": "unknown song"}, 404)
        orig_p = os.path.join(PACK, sid, sid + ".TextGrid")
        corr_p = os.path.join(CORR, sid + ".TextGrid")
        draft_p = os.path.join(DRAFTS, sid + ".json")
        out = {"id": sid, "meta": song_meta(sid), "state": load_state().get(sid, {}),
               "original": parse_textgrid(read_text_any(orig_p)),
               "corrected": None, "corrected_at": mtime(corr_p), "draft": None, "draft_at": mtime(draft_p)}
        if os.path.exists(corr_p):
            try:
                out["corrected"] = parse_textgrid(read_text_any(corr_p))
            except Exception as e:
                out["corrected_error"] = str(e)
        if os.path.exists(draft_p):
            try:
                out["draft"] = json.load(open(draft_p, encoding="utf-8"))
            except Exception:
                pass
        self.send_json(out)

    def do_POST(self):
        p = unquote(urlparse(self.path).path)
        m = re.fullmatch(r"/api/(draft|save|state|discard_draft|praat|open_folder)/(.+)", p)
        if not m or not valid_id(m.group(2)):
            return self.send_json({"error": "bad request"}, 400)
        act, sid = m.groups()
        try:
            data = self.body()
            ensure_dirs()
            with LOCK:
                if act == "draft":
                    atomic_write(os.path.join(DRAFTS, sid + ".json"), json.dumps(data, ensure_ascii=False))
                    return self.send_json({"ok": True, "at": time.time()})
                if act == "discard_draft":
                    f = os.path.join(DRAFTS, sid + ".json")
                    if os.path.exists(f):
                        os.remove(f)
                    return self.send_json({"ok": True})
                if act == "state":
                    st = load_state()
                    st.setdefault(sid, {}).update(data)
                    atomic_write(STATE, json.dumps(st, ensure_ascii=False, indent=1))
                    return self.send_json({"ok": True})
                if act == "save":
                    return self.save(sid, data)
                if act == "open_folder":
                    if hasattr(os, "startfile"):
                        os.startfile(CORR)
                    return self.send_json({"ok": True})
                if act == "praat":
                    f = os.path.join(CORR, sid + ".TextGrid")
                    if not os.path.exists(f):
                        return self.send_json({"error": "save first"}, 400)
                    exe = PRAAT if os.path.exists(PRAAT) else "praat"
                    subprocess.Popen([exe, "--open", os.path.join(PACK, sid, "vocals.wav"),
                                      os.path.join(PACK, sid, "mix.wav"), f])
                    return self.send_json({"ok": True, "file": f})
        except Exception as e:
            return self.send_json({"error": str(e)}, 500)

    def save(self, sid, tg):
        tiers = []
        for t in tg["tiers"]:
            kind = t.get("kind", "interval")
            if kind == "interval":
                items = [{"a": round(float(i["a"]), 6), "b": round(float(i["b"]), 6),
                          "t": " ".join(str(i["t"]).split())} for i in t["items"]]
            else:
                items = [{"t": round(float(i["t"]), 6), "mark": " ".join(str(i["mark"]).split())} for i in t["items"]]
            tiers.append({"name": str(t["name"]), "kind": kind, "xmin": float(t["xmin"]), "xmax": float(t["xmax"]), "items": items})
        tg = {"xmin": float(tg["xmin"]), "xmax": float(tg["xmax"]), "tiers": tiers}
        check_textgrid(tg)
        names = [t["name"] for t in tiers]
        missing = [n for n in REQUIRED_TIERS if n not in names]
        if missing:
            raise ValueError("required tier(s) missing: %s (the pipeline reads these names)" % ", ".join(missing))
        text = write_textgrid(tg)
        if parse_textgrid(text) != tg:
            raise ValueError("round-trip check failed; nothing was saved")
        dst = os.path.join(CORR, sid + ".TextGrid")
        if os.path.exists(dst):
            hd = os.path.join(HIST, sid)
            os.makedirs(hd, exist_ok=True)
            shutil.copy2(dst, os.path.join(hd, time.strftime("%Y%m%d-%H%M%S") + ".TextGrid"))
        atomic_write(dst, text)
        dp = os.path.join(DRAFTS, sid + ".json")
        if os.path.exists(dp):
            os.remove(dp)
        return self.send_json({"ok": True, "path": dst, "at": mtime(dst)})


def main():
    ensure_dirs()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = "http://127.0.0.1:%d/" % PORT
    print("P1 gold annotator running at", url)
    print("  songs:     ", PACK)
    print("  saving to: ", CORR)
    print("Keep this window open while you work. Press Ctrl+C to stop.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
