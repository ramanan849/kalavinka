"""``deploy.compose_matrix``: voice priority order, file names, and the job plan (no model needed)."""
from pathlib import Path

import pytest

from kalavinka.deploy.compose_matrix import pick_voices, plan_jobs, take_name

TABLE = [{"no": 9, "gender": "male", "singer": "Arijit Singh"},
         {"no": 17, "gender": "female", "singer": "Anuradha Paudwal"},
         {"no": 2, "gender": "female", "singer": "Lata Mangeshkar"},
         {"no": 5, "gender": "male", "singer": "Udit Narayan"}]


def test_voices_in_priority_order_each_once():
    """``17,9,all``: the named pair first, then every other voice once, in table order."""
    assert [v["no"] for v in pick_voices("17,9,all", TABLE)] == [17, 9, 2, 5]
    assert [v["no"] for v in pick_voices("female", TABLE)] == [17, 2]
    with pytest.raises(SystemExit):
        pick_voices("99", TABLE)


def test_take_name_is_filesystem_safe():
    """Voice number, gender and singer, with spaces and punctuation as underscores."""
    assert take_name({"no": 17, "gender": "female", "singer": "Anuradha Paudwal"}) == "v17_female_Anuradha_Paudwal"
    assert take_name({"no": 3, "gender": "male", "singer": "K. S. Chithra?"}) == "v03_male_K_S_Chithra"


def test_plan_puts_first_voices_first_for_every_composition(tmp_path):
    """Jobs are ordered voice first, so a round-robin over workers finishes the first voices everywhere first."""
    comps = [Path("a.json"), Path("b.json")]
    jobs = plan_jobs(comps, pick_voices("17,9,all", TABLE), ["kalavinka", "base"], tmp_path)
    assert len(jobs) == 2 * 4 * 2
    assert {j["voice"]["no"] for j in jobs[:8]} == {17, 9}
    assert jobs[0]["wav"] == str(tmp_path / "a" / "kalavinka" / "v17_female_Anuradha_Paudwal.wav")
