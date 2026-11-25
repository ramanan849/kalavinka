"""Regression tests for the G2P fixes of 25 Nov 2026 (saregama.g2p, saregama.numbers_hi): nasalised medial schwa,
final schwa after r/v clusters, numbers, ॐ and symbols. These words occur in the training lyrics (चाँदनी 33 times)."""
import pytest

from saregama.g2p import line_to_syllables
from saregama.numbers_hi import number_words, spell_numbers


def syl(text):
    """Syllable texts of a line."""
    return [s.text for s in line_to_syllables(text, "hi")]


def phones(text):
    """Phones per syllable."""
    return ["-".join(s.phones) for s in line_to_syllables(text, "hi")]


@pytest.mark.parametrize("word,n", [("चाँदनी", 2), ("हँसना", 2), ("हंसना", 2), ("उँगली", 2), ("माँगना", 2),
                                    ("झाँकना", 2), ("पूर्व", 1), ("गर्व", 1), ("विश्व", 1)])
def test_fixed_words(word, n):
    """Medial schwa deletion now looks through vowel nasalisation; no extra final schwa after r/v clusters."""
    assert len(syl(word)) == n


@pytest.mark.parametrize("word,n", [("सत्य", 2), ("मित्र", 2), ("कार्य", 2), ("अंगना", 3), ("चंदा", 2), ("बंधन", 2),
                                    ("जिंदगी", 3), ("मेरा", 2), ("करना", 2), ("कहाँ", 2), ("आँसू", 2)])
def test_unchanged_words(word, n):
    """Clusters ending in य/र keep their schwa; an anusvara that becomes a nasal consonant still closes a syllable."""
    assert len(syl(word)) == n


def test_chandni_phones():
    """चाँदनी = chaand-nii."""
    assert phones("चाँदनी") == ["c-aa-nas-d", "n-ii"]


def test_numbers_are_spelled():
    """ASCII and Devanagari digits become Hindi words; leading zeros are read digit by digit."""
    assert number_words(2) == "दो" and number_words(21) == "इक्कीस" and number_words(99) == "निन्यानवे"
    assert number_words(105) == "एक सौ पाँच" and number_words(2025) == "दो हज़ार पच्चीस"
    assert number_words(150000) == "एक लाख पचास हज़ार"
    assert spell_numbers("007").split() == ["शून्य", "शून्य", "सात"]
    assert syl("हम 2 हैं") == ["हम", "दो", "हैं"]
    assert syl("१२ बजे")[:2] == ["बा", "रह"]


def test_symbols_and_om():
    """Symbols and curly quotes are ignored instead of raising; ॐ is sung as ओम्."""
    assert syl("तू ही ♪ मेरा") == ["तू", "ही", "मे", "रा"]
    assert syl("‘हाँ’ ’") == ["हाँ"]
    assert phones("ॐ")[0] == "oo-m"
    assert syl("♪ ♪") == []
