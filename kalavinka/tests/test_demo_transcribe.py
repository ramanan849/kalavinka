"""``annot.demo_transcribe``: swara naming, note segmentation, phrase fitting and fixing on synthetic pitch curves."""
import numpy as np

from kalavinka.annot.demo_transcribe import (FPS, apply_fixes, compare, designed_semis, fit_phrase, segment_notes,
                                             swara_text, vibrato_of)


def test_swara_text_octaves_and_komal():
    """Semitones above Sa -> the app's notation, in all three octaves."""
    assert swara_text(0) == "Sa" and swara_text(12) == "Sa'" and swara_text(-2) == ".ni(k)"
    assert swara_text(11) == "Ni" and swara_text(10) == "ni(k)" and swara_text(-3) == ".Dha"
    assert swara_text(5) == "ma" and swara_text(6) == "Ma(t)"


def steps(seq, sec=0.3, wobble=0.0):
    """A cents curve of held notes (semitones, seconds each) with a small optional vibrato."""
    out = []
    for s in seq:
        n = int(sec * FPS)
        t = np.arange(n) / FPS
        out.append(s * 100 + wobble * np.sin(2 * np.pi * 5.5 * t))
    return np.concatenate(out)


def test_segment_notes_splits_repeats_at_onsets():
    """Ni Sa' Ni Sa' -> four notes; a repeated pitch is split only where an onset says so."""
    c = steps([11, 12, 11, 12])
    notes = segment_notes(c, [0, 2, 4, 5, 7, 9, 10, 11], on_frames=[])
    assert [n["semi"] for n in notes] == [11, 12, 11, 12]
    same = steps([12, 12])
    assert len(segment_notes(same, [0, 11], on_frames=[])) == 1
    assert len(segment_notes(same, [0, 11], on_frames=[15])) == 2


def test_vibrato_detected_on_held_note_only():
    """A held note with a 5.5 Hz, 40-cent wobble gets a vibrato ornament; a flat one does not."""
    v = vibrato_of(steps([7], sec=1.2, wobble=40))
    assert v is not None and 4.5 <= v["rate"] <= 6.5 and 20 <= v["depth"] <= 60
    assert vibrato_of(steps([7], sec=1.2)) is None


def test_fit_phrase_finds_offset_and_wrong_note(tmp_path):
    """A designed phrase with one wrong swara is found at its offset; compare flags exactly that note, and
    apply_fixes replaces it (and the swara-name syllable)."""
    sung = np.concatenate([np.full(40, np.nan), steps([7, 9, 7, 9, 4, 5, 7, 9]), np.full(40, np.nan)])
    comp = {"tempo_bpm": 100, "phrases": [{"notes": [
        {"syl": "प", "swara": "Pa", "sec": 0.3}, {"syl": "ध", "swara": "Dha", "sec": 0.3},
        {"syl": "प", "swara": "Pa", "sec": 0.3}, {"syl": "ध", "swara": "Dha", "sec": 0.3},
        {"syl": "ग", "swara": "Ga", "sec": 0.3}, {"syl": "म", "swara": "ma", "sec": 0.3},
        {"syl": "प", "swara": "Pa", "sec": 0.3}, {"syl": "नि", "swara": "ni(k)", "sec": 0.3}]}]}
    fit = fit_phrase(sung, designed_semis(comp)[0], 30, search=(-20, 20), stretch=(1.0,))
    assert fit["start"] == 40 and fit["octave"] == 0
    hz = np.where(np.isfinite(sung), 200.0 * 2 ** (np.nan_to_num(sung) / 1200), 0.0)
    (tmp_path / "f0.npz").write_bytes(b"")
    np.savez(tmp_path / "f0.npz", f0=hz, agree=np.ones_like(hz), fps=FPS)
    (tmp_path / "analysis.json").write_text('{"tonic_hz": 200.0, "audio": "x.opus"}')
    rep = compare(tmp_path, comp, start=0.6)
    wrong = [n for n in rep["phrases"][0]["notes"] if n.get("ok") is False]
    assert [(n["i"], n["designed"], n["sung"]) for n in wrong] == [(8, "ni(k)", "Dha")]
    fixed = apply_fixes(comp, rep)
    last = fixed["phrases"][0]["notes"][-1]
    assert last["swara"] == "Dha" and last["syl"] == "ध"


def test_syllable_starts_restart_after_pause_and_pick_strong_onsets():
    """Four sung syllables (tone bursts), the third after a 0.5 s pause: the first voiced frame and the voice
    resuming after the pause are always starts; the others come from the strongest onsets."""
    from kalavinka.annot.demo_transcribe import SR, syllable_starts
    t_on = [0.2, 0.7, 1.7, 2.2]                         # burst starts (s); a pause between 1.2 and 1.7
    wav = np.zeros(int(3.0 * SR), np.float32)
    voiced = np.zeros(int(3.0 * FPS), bool)
    for k, t0 in enumerate(t_on):
        n = int(0.45 * SR) if k != 1 else int(0.5 * SR)
        tt = np.arange(n) / SR
        wav[int(t0 * SR):int(t0 * SR) + n] = (0.5 * np.sin(2 * np.pi * 220 * tt) * np.minimum(1, tt / 0.005)
                                               * np.minimum(1, (tt[-1] - tt) / 0.03))
        voiced[int(t0 * FPS):int(t0 * FPS) + n * FPS // SR] = True
    st = syllable_starts(wav, voiced, 0.0, 3.0, 4)
    assert len(st) == 4
    assert abs(st[0] - 0.2) < 0.03 and abs(st[2] - 1.7) < 0.03
    assert abs(st[1] - 0.7) < 0.06 and abs(st[3] - 2.2) < 0.06
