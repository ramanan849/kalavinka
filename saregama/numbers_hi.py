"""Spell out numbers in Hindi lyrics, so the G2P sings them instead of dropping or rejecting them.

Why: lyric lines sometimes contain numbers ("हम 2 हैं", "१२ बजे"). Before this module the G2P raised an error on
ASCII digits and silently dropped Devanagari digits. ``spell_numbers`` replaces every run of digits (ASCII 0-9 or
Devanagari ०-९) by its Hindi words:

* 0-99 from the table (Hindi numbers below 100 are irregular, so each has its own word);
* larger numbers with सौ (100), हज़ार (1,000), लाख (1,00,000) and करोड़ (1,00,00,000), Indian grouping;
* a run that starts with 0 and has more digits ("007"), or is longer than 9 digits, is read digit by digit.

Separators inside numbers (1,000 or 2.5) are not interpreted: "2.5" is read as "दो" "पाँच".
"""
from __future__ import annotations

import re

UNITS = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
         "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
         "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
         "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
         "पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
         "साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
         "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी "
         "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
         "नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()
assert len(UNITS) == 100

_DIGITS = re.compile(r"[0-9०-९]+")
_DEV_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")
SCALES = ((10 ** 7, "करोड़"), (10 ** 5, "लाख"), (1000, "हज़ार"), (100, "सौ"))


def number_words(n: int) -> str:
    """Hindi words for a non-negative integer (Indian grouping)."""
    if n < 0:
        raise ValueError("negative numbers are not spelled")
    if n < 100:
        return UNITS[n]
    out = []
    for size, word in SCALES:
        if n >= size:
            out += [number_words(n // size), word]
            n %= size
    if n:
        out.append(UNITS[n])
    return " ".join(out)


def spell_digits(run: str) -> str:
    """One run of digits -> Hindi words (digit by digit for leading zeros or very long runs)."""
    s = run.translate(_DEV_TO_ASCII)
    if (len(s) > 1 and s[0] == "0") or len(s) > 9:
        return " ".join(UNITS[int(c)] for c in s)
    return number_words(int(s))


def spell_numbers(text: str) -> str:
    """Replace every digit run in ``text`` by its Hindi words (padded with spaces)."""
    return _DIGITS.sub(lambda m: f" {spell_digits(m.group(0))} ", text)
