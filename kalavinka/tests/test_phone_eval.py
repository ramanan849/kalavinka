"""Tests for kalavinka.eval.phone_eval (CPU, no model download)."""
from kalavinka.eval.phone_eval import align, base, clip_counts, is_asp, is_retro, pooled, target_phones


def test_ipa_covers_all_phones():
    """Every Indic phone has an IPA value the target builder can use."""
    from saregama.phoneset import PHONES
    assert all(p.ipa for p in PHONES) and len(PHONES) == 62


def test_target_phones_skips_nasal_mark_and_non_indic():
    """``nas`` and en_/<SP> tokens are dropped; names map to IPA."""
    t = target_phones("in_Th-aa-nas <SP> en_HH in_d-a")
    assert [n for n, _ in t] == ["Th", "aa", "d", "a"]
    assert t[0][1] == "ʈʰ"


def test_feature_predicates_and_base():
    """Retroflex / aspiration detection and the base form used for the cheap substitution."""
    assert is_retro("ʈʰ") and is_retro("ɖ") and not is_retro("t̪")
    assert is_asp("ʈʰ") and is_asp("th") and not is_asp("t")
    assert base("ʈʰ") == base("t̪") == base("th") == "t"


def test_counts_hit_and_false_alarm():
    """A retroflex aspirate sung as dental plain: miss on both features; dental sung retroflex: false alarm."""
    target = [("Th", "ʈʰ"), ("aa", "aː"), ("t", "t̪"), ("a", "ə")]
    c = clip_counts(target, ["t", "aː", "ʈ", "ə"])
    assert c["retro_n"] == 1 and c["retro_hit"] == 0 and c["retro_fa"] == 1
    assert c["asp_n"] == 1 and c["asp_hit"] == 0
    assert c["held_n"] == 1 and c["errors"] == 2
    good = clip_counts(target, ["ʈʰ", "aː", "t̪", "ə"])
    assert good["retro_hit"] == 1 and good["asp_hit"] == 1 and good["errors"] == 0
    s = pooled([c, good])
    assert s["retro_recall"] == 0.5 and s["held_asp_recall"] == 0.5


def test_align_handles_insertions():
    """An inserted phone does not shift the alignment of the rest."""
    pairs = align(["a", "b", "c"], ["a", "x", "b", "c"])
    assert (1, 2) in pairs and (2, 3) in pairs and (None, 1) in pairs
