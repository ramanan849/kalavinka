"""Tests for the take history v2 (kalavinka.deploy.history): atomic saves, labels in an append-only log, soft delete
and restore, search, the event log, rebuild from sidecars, heavy info kept out of history.jsonl, old files still read,
and lyrics with line separators (U+2028) that used to break the log."""
import json

import numpy as np
import pytest

from kalavinka.deploy import history as H


def make(h, kind="sing", lyrics="ठंडी हवा", model="kalavinka", sung=None):
    """Add one take through the cache."""
    info = {"took_s": 1.0, **({"sung": sung} if sung else {})}
    req = {"lyrics": lyrics, "model": model, "seed": 0}
    return h.cached(kind, req, lambda: (np.zeros(2400, np.float32), 24000, info), lyrics)


def test_add_find_and_heavy_info(tmp_path):
    """The light log has no sung curve; the sidecar and a cache hit do."""
    h = H.History(tmp_path)
    *_, hit = make(h, sung=[1, 2, 3])
    assert not hit
    line = json.loads(h.path.read_text(encoding="utf-8").split("\n")[0])
    assert "sung" not in line["info"]
    wav, sr, info, path, hit = make(h, sung=[1, 2, 3])
    assert hit and info["sung"] == [1, 2, 3]
    assert not list(tmp_path.glob(".*.tmp"))


def test_labels_delete_restore_search(tmp_path):
    """Title, tags, notes, star; deleted takes leave the cache and come back on restore."""
    h = H.History(tmp_path)
    make(h, lyrics="पहला")
    make(h, lyrics="दूसरा", model="base")
    h.annotate(1, "title", "first try")
    h.annotate(1, "tags", "slow, sad")
    h.annotate(1, "star", True)
    h.annotate(2, "notes", "too fast")
    assert [r["id"] for r in h.search(starred=True)] == [1]
    assert [r["id"] for r in h.search("sad")] == [1] and [r["id"] for r in h.search("too fast")] == [2]
    assert [r["id"] for r in h.search(model="base")] == [2]
    assert h.get(1)["tags"] == ["slow", "sad"] and "★" in H.describe(h.get(1))
    h.delete(1)
    assert (tmp_path / "trash").exists() and h.search() and all(r["id"] != 1 for r in h.search())
    *_, hit = make(h, lyrics="पहला")                     # deleted: the cache sings again (new take 3)
    assert not hit
    h.restore(1)
    assert any(r["id"] == 1 for r in h.search()) and not h.get(1)["missing"]
    with pytest.raises(ValueError):
        h.annotate(1, "colour", "red")


def test_event_log_and_errors(tmp_path):
    """Requests, hits and failures are logged."""
    h = H.History(tmp_path)
    make(h)
    make(h)
    with pytest.raises(RuntimeError):
        h.cached("sing", {"lyrics": "x"}, lambda: (_ for _ in ()).throw(RuntimeError("boom")), "x")
    ev = [e["event"] for e in reversed(h.events())]
    assert ev[:4] == ["sing", "take_saved", "cache_hit", "sing"] and ev[-1] == "sing_failed"
    assert "boom" in h.events(1)[0]["error"]


def test_rebuild_from_sidecars_and_line_separators(tmp_path):
    """history.jsonl can be lost: rebuild() re-creates it; lyrics with U+2028 no longer break the log."""
    h = H.History(tmp_path)
    make(h, lyrics="एक दो")
    make(h, lyrics="तीन")
    assert len(h.records()) == 2
    h.path.unlink()
    assert h.rebuild() == 2
    assert [r["id"] for r in h.records()] == [1, 2]
    assert make(h, lyrics="तीन")[-1]                      # cache still works after the rebuild


def test_old_format_still_reads(tmp_path):
    """A history.jsonl written by the first version (no annotations, no log) is read as before."""
    import soundfile as sf
    sf.write(tmp_path / "a.wav", np.zeros(100, np.float32), 24000)
    rec = {"id": 7, "time": "2026-11-25 10:00:00", "key": "k", "kind": "sargam", "model": "base",
           "request": {"title": "t", "model": "base"}, "wav": "a.wav", "info": {}}
    (tmp_path / "history.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    h = H.History(tmp_path)
    assert h.recent()[0]["id"] == 7 and h.find("k")["id"] == 7


def test_export_zip(tmp_path):
    """Export writes the chosen takes' wav and json into one zip."""
    import zipfile
    h = H.History(tmp_path)
    make(h)
    z = h.export([1])
    assert sorted(n.rsplit(".", 1)[1] for n in zipfile.ZipFile(z).namelist()) == ["json", "wav"]
