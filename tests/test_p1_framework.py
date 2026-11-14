"""P1 framework: atomic state, stage invalidation, claims, resumable workers,
manifest ids and duplicate detection (offline, no models)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saregama.p1 import clean, manifest  # noqa: E402
from saregama.p1.state import SongState, atomic_json, read_json  # noqa: E402
from saregama.p1.worker import Context, Skip, Stage, claim, release, run_worker, status_table  # noqa: E402


def _tmp() -> Path:
    """A fresh temporary folder."""
    return Path(tempfile.mkdtemp(prefix="p1_"))


def test_manifest_ids():
    """Readable, stable ids; files without a YouTube id get a name hash."""
    info = manifest.parse_name("2013 - Tum Hi Ho [Umqb9KENgmk]")
    assert info == {"year": "2013", "title": "Tum Hi Ho", "video_id": "Umqb9KENgmk"}
    assert manifest.make_id("2013", "Tum Hi Ho", "Umqb9KENgmk", "x") == "2013_tum_hi_ho__Umqb9KENgmk"
    assert manifest.make_id("", "Song", "", "a.mp3") != manifest.make_id("", "Song", "", "b.mp3")
    d = _tmp()
    for n in ("2013 - A [x1].opus", "2014 - B (Lyrical) [y2].m4a", "notes.txt"):
        (d / n).write_bytes(b"0")
    songs = manifest.build(d)
    assert [s.song_id for s in songs] == ["2013_a__x1", "2014_b_lyrical__y2"]
    manifest.write(songs, d / "songs.tsv")
    assert manifest.read(d / "songs.tsv") == songs


def test_state_versions_and_config():
    """A stage is done only for the same version and config hash."""
    st = SongState(_tmp())
    st.mark_done("f0", 1, "aaa", 1.0, n=3)
    assert st.is_done("f0", 1, "aaa")
    assert not st.is_done("f0", 2, "aaa") and not st.is_done("f0", 1, "bbb")
    assert st.mark_failed("asr", 1, "c", "boom") == 1 and not st.is_parked("asr")
    assert st.mark_failed("asr", 1, "c", "boom") == 2 and st.is_parked("asr")
    st.reset("asr")
    assert st.get("asr") is None and st.get("f0")["n"] == 3
    atomic_json(st.dir / "x.json", {"a": 1})
    assert read_json(st.dir / "x.json") == {"a": 1} and not list(st.dir.glob("*.tmp.*"))


def test_claims_and_stale_takeover():
    """A claim blocks others; a claim of a dead owner is taken over."""
    w = _tmp()
    assert claim(w, "s1") and claim(w, "s1")            # our own claim
    release(w, "s1")
    d = w / "claims" / "s2"
    d.mkdir(parents=True)
    (d / "owner").write_text("nosuchhost:1")           # unknown host -> assumed alive
    assert not claim(w, "s2")
    import socket
    (d / "owner").write_text(f"{socket.gethostname()}:999999999")   # dead local pid
    assert claim(w, "s2")


def test_worker_resumes_and_isolates_failures():
    """Failures park after 2 tries without stopping other songs; a second run
    only does what is left; changing a version re-runs that stage."""
    w = _tmp()
    calls = []

    def a(ctx, song, sdir):
        """Stage A: records the call; fails for song s2."""
        calls.append(("a", song.song_id))
        if song.song_id == "s2":
            raise RuntimeError("bad file")
        return {"ok": 1}

    def b(ctx, song, sdir):
        """Stage B: skipped for s3."""
        calls.append(("b", song.song_id))
        if song.song_id == "s3":
            raise Skip("duplicate")

    songs = [manifest.Song(f"s{i}", "", "", "", "") for i in (1, 2, 3)]
    sa, sb = Stage("a", 1, a), Stage("b", 1, b, depends=("a",))
    allst = {"a": sa, "b": sb}
    ctx = Context({}, w)
    r = run_worker(ctx, songs, [sa, sb], allst, budget_s=60)
    assert r == {"done": 3, "skipped": 1, "failed": 1, "songs": 3}
    assert ("b", "s2") not in calls
    calls.clear()
    run_worker(ctx, songs, [sa, sb], allst, budget_s=60)
    assert calls == [("a", "s2")]                       # only the failed one is retried
    calls.clear()
    run_worker(ctx, songs, [sa, sb], allst, budget_s=60)
    assert calls == []                                  # parked after 2 failures
    sb2 = Stage("b", 2, b, depends=("a",))              # new version of B
    run_worker(ctx, songs, [sa, sb2], {"a": sa, "b": sb2}, budget_s=60)
    assert sorted(calls) == [("b", "s1"), ("b", "s3")]
    tab = status_table(songs, [sa, sb2], w, {})
    assert tab["a"] == {"done": 2, "skipped": 0, "failed": 0, "parked": 1, "pending": 0}
    assert tab["b"]["skipped"] == 1 and tab["b"]["pending"] == 1


def test_duplicate_detection():
    """Two copies of one fingerprint (one shifted, a few bits flipped) group
    together; an unrelated one stays alone; the plainer title is kept."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2 ** 32, 1500, dtype=np.uint64).astype(np.uint32)
    noisy = base[10:].copy()
    noisy[::7] ^= np.uint32(1)
    other = rng.integers(0, 2 ** 32, 1500, dtype=np.uint64).astype(np.uint32)
    w = _tmp()
    songs = [manifest.Song("a_lyrical", "", "2013", "Song (Lyrical)", "x"),
             manifest.Song("a_plain", "", "2013", "Song", "y"),
             manifest.Song("b", "", "2014", "Other", "z")]
    for s, fp in zip(songs, (base, noisy, other)):
        d = w / "songs" / s.song_id
        d.mkdir(parents=True)
        np.save(d / "fingerprint.npy", fp)
        atomic_json(d / "inventory.json", {"bit_rate": 128000, "duration_s": 200})
    res = clean.find_duplicates(songs, w)
    assert res["groups"] == [["a_plain", "a_lyrical"]]
    assert res["duplicate_of"] == {"a_lyrical": "a_plain"}
    assert clean.ber(base, other)[0] > 0.4


def test_dependency_rerun_invalidates_dependants():
    """Re-running a stage (new version) makes stages that depend on it stale."""
    w = _tmp()
    calls = []

    def mk(name):
        """A stage function that records its calls."""
        def fn(ctx, song, sdir):
            """Record the call."""
            calls.append(name)
        return fn

    songs = [manifest.Song("s1", "", "", "", "")]
    a1, b1, c1 = Stage("a", 1, mk("a")), Stage("b", 1, mk("b"), depends=("a",)), Stage("c", 1, mk("c"))
    ctx = Context({}, w)
    run_worker(ctx, songs, [a1, b1, c1], {"a": a1, "b": b1, "c": c1}, budget_s=60)
    assert calls == ["a", "b", "c"]
    calls.clear()
    a2 = Stage("a", 2, mk("a"))
    run_worker(ctx, songs, [a2], {"a": a2, "b": b1, "c": c1}, budget_s=60)     # only A re-run now
    assert calls == ["a"]
    calls.clear()
    run_worker(ctx, songs, [a2, b1, c1], {"a": a2, "b": b1, "c": c1}, budget_s=60)
    assert calls == ["b"]                                                      # B is stale, C is not
