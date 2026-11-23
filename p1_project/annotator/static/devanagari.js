// Devanagari helpers: written word -> spoken syllables (Hindi schwa deletion),
// and romanized (Hinglish / ITRANS-ish) -> Devanagari. Browser: window.Deva; node: module.exports.
(function (root) {
  'use strict';
  const HALANT = '्', NUKTA = '़';
  const isCons = c => (c >= 'क' && c <= 'ह') || (c >= 'क़' && c <= 'य़') || (c >= 'ॸ' && c <= 'ॿ');
  const isIndepVowel = c => (c >= 'ऄ' && c <= 'औ') || c === 'ॠ' || c === 'ॡ' || c === 'ॲ';
  const isMatra = c => (c >= 'ा' && c <= 'ौ') || c === 'ॢ' || c === 'ॣ' || c === 'ॎ' || c === 'ॏ' || c === 'ॕ' || c === 'ॖ' || c === 'ॗ';
  const isMod = c => c === 'ऀ' || c === 'ँ' || c === 'ं' || c === 'ः';

  function aksharas(w) {
    const units = [];
    let i = 0;
    while (i < w.length) {
      const c = w[i];
      const u = { cons: [], vowel: null, inherent: false, start: i, cut: [] };
      if (isIndepVowel(c)) { u.vowel = c; i++; }
      else if (isCons(c)) {
        while (i < w.length && isCons(w[i])) {
          let cc = w[i]; i++;
          if (w[i] === NUKTA) { cc += NUKTA; i++; }
          u.cons.push(cc);
          if (w[i] === HALANT && i + 1 < w.length && isCons(w[i + 1])) { i++; u.cut.push(i); continue; }
          if (w[i] === HALANT) { i++; u.vowel = null; break; }
          if (i < w.length && isMatra(w[i])) { u.vowel = w[i]; i++; }
          else { u.vowel = 'a'; u.inherent = true; }
          break;
        }
      } else { u.other = true; i++; }
      while (i < w.length && (isMod(w[i]) || w[i] === NUKTA)) i++;
      u.end = i; u.text = w.slice(u.start, u.end);
      units.push(u);
    }
    return units;
  }

  function schwaDelete(units) {
    const n = units.length;
    const hasV = u => u.vowel !== null && !u.deleted && !u.other;
    if (n > 1 && units[n - 1].inherent) units[n - 1].deleted = true;
    for (let k = n - 2; k >= 1; k--) {
      const u = units[k];
      if (!u.inherent || u.cons.length !== 1) continue;
      const p = units[k - 1], q = units[k + 1];
      if (!hasV(p) || !hasV(q) || q.cons.length !== 1) continue;
      u.deleted = true;
    }
    return units;
  }

  // Spoken syllables as substrings that concatenate back to the written word.
  function syllabify(word) {
    if (!word) return [];
    if (!/[ऀ-ॿ]/.test(word) || /[A-Za-z]/.test(word)) return [word];
    const units = schwaDelete(aksharas(word)), syl = [];
    for (const u of units) {
      const nucleus = u.vowel !== null && !u.deleted && !u.other;
      if (!nucleus) { if (syl.length) syl[syl.length - 1] += u.text; else syl.push(u.text); continue; }
      if (u.cons.length > 1 && syl.length && u.cut.length) {
        const cutAt = u.cut[u.cut.length - 1] - u.start;
        syl[syl.length - 1] += u.text.slice(0, cutAt);
        syl.push(u.text.slice(cutAt));
      } else syl.push(u.text);
    }
    return syl.filter(s => s.length);
  }

  // ---------- romanized -> Devanagari ----------
  const V = [['aa', 'आ', 'ा'], ['A', 'आ', 'ा'], ['ai', 'ऐ', 'ै'], ['au', 'औ', 'ौ'], ['ee', 'ई', 'ी'], ['ii', 'ई', 'ी'], ['I', 'ई', 'ी'],
    ['oo', 'ऊ', 'ू'], ['uu', 'ऊ', 'ू'], ['U', 'ऊ', 'ू'], ['RRi', 'ऋ', 'ृ'], ['R^i', 'ऋ', 'ृ'], ['.o', 'ऑ', 'ॉ'],
    ['a', 'अ', ''], ['i', 'इ', 'ि'], ['u', 'उ', 'ु'], ['e', 'ए', 'े'], ['o', 'ओ', 'ो']];
  const C = [['chh', 'छ'], ['Ch', 'छ'], ['ksh', 'क्ष'], ['GY', 'ज्ञ'], ['gy', 'ज्ञ'], ['shh', 'ष'], ['Sh', 'ष'],
    ['kh', 'ख'], ['gh', 'घ'], ['~N', 'ङ'], ['ch', 'च'], ['jh', 'झ'], ['~n', 'ञ'], ['Th', 'ठ'], ['Dh', 'ढ'], ['.Dh', 'ढ़'], ['.D', 'ड़'],
    ['th', 'थ'], ['dh', 'ध'], ['ph', 'फ'], ['bh', 'भ'], ['sh', 'श'], ['T', 'ट'], ['D', 'ड'], ['N', 'ण'], ['R', 'ड़'],
    ['k', 'क'], ['g', 'ग'], ['c', 'च'], ['j', 'ज'], ['t', 'त'], ['d', 'द'], ['n', 'न'], ['p', 'प'], ['f', 'फ़'],
    ['b', 'ब'], ['m', 'म'], ['y', 'य'], ['r', 'र'], ['l', 'ल'], ['v', 'व'], ['w', 'व'], ['s', 'स'], ['h', 'ह'],
    ['q', 'क़'], ['z', 'ज़'], ['x', 'क्स'], ['K', 'ख़'], ['G', 'ग़']];
  const MODS = [['.n', 'ं'], ['M', 'ं'], ['~', 'ँ'], ['.N', 'ँ'], ['H', 'ः'], ['.h', '्'], ['|', '।']];
  const sortLen = a => a.slice().sort((x, y) => y[0].length - x[0].length);
  const Vs = sortLen(V), Cs = sortLen(C), Ms = sortLen(MODS);
  const match = (s, i, tbl) => { for (const e of tbl) if (s.startsWith(e[0], i)) return e; return null; };
  const STOPS = /^(k|kh|g|gh|ch|chh|j|jh|T|Th|D|Dh|t|th|d|dh|p|ph|b|bh)$/;

  const COMMON_LIST = `main मैं mai मैं mein में me में haan हाँ han हाँ hain हैं hai है nahin नहीं nahi नहीं kyun क्यों kyon क्यों kyu क्यूँ
    tum तुम hum हम yeh ये ye ये wo वो woh वो vo वो toh तो to तो na ना naa ना ho हो jo जो se से ke के ki की ka का ko को de दे le ले
    ek एक yahi यही maine मैंने leke लेके bhi भी aur और pe पे par पर ne ने tu तू mera मेरा meri मेरी mere मेरे tera तेरा teri तेरी
    tere तेरे dil दिल pyar प्यार pyaar प्यार ishq इश्क़ jaan जान yaar यार yaara यारा mujhe मुझे tujhe तुझे apna अपना apni अपनी apne अपने
    koi कोई kahin कहीं kabhi कभी abhi अभी sab सब baat बात aaj आज kal कल zindagi ज़िंदगी sapna सपना sapne सपने re रे o ओ oye ओए
    balle बल्ले mahi माही soniye सोणिए dholna ढोलना dekho देखो chal चल chalo चलो aaja आजा ha हा raha रहा rahi रही rahe रहे tha था
    thi थी the थे hi ही sanam सनम sajna सजना jana जाना dard दर्द kya क्या humko हमको tumko तुमको aankhen आँखें ankhiyan अँखियाँ
    kuch कुछ kuchh कुछ jab जब tab तब ab अब phir फिर fir फिर saath साथ mann मन man मन jaane जाने jaana जाना hoon हूँ hun हूँ hu हूँ
    hui हुई hua हुआ yun यूँ yoon यूँ`.split(/\s+/).filter(Boolean);
  const COMMON = {};
  for (let i = 0; i + 1 < COMMON_LIST.length; i += 2) COMMON[COMMON_LIST[i]] = COMMON_LIST[i + 1];

  function roman2word(s) {
    let out = '', i = 0, prevCons = false;
    while (i < s.length) {
      let e;
      if ((e = match(s, i, Ms))) { out += e[1]; i += e[0].length; prevCons = false; continue; }
      if ((e = match(s, i, Cs))) {
        if (e[0] === 'n' && !prevCons) { // Hinglish: n before a stop = anusvara (zindagi -> ज़िंदगी)
          const nx = match(s, i + 1, Cs);
          if (nx && STOPS.test(nx[0])) { out += 'ं'; i += 1; continue; }
        }
        if (prevCons) out += HALANT;
        out += e[1]; i += e[0].length; prevCons = true; continue;
      }
      if ((e = match(s, i, Vs))) {
        let m = e[2];
        const final = !/[A-Za-z.~]/.test(s[i + e[0].length] || '');
        if (final && prevCons && e[0].length === 1) { if (e[0] === 'a') m = 'ा'; else if (e[0] === 'i') m = 'ी'; else if (e[0] === 'u') m = 'ू'; }
        out += prevCons ? m : e[1]; i += e[0].length; prevCons = false; continue;
      }
      out += s[i]; i++; prevCons = false;
    }
    return out;
  }
  function roman2deva(s) {
    return s.split(/(\s+)/).map(w => COMMON[w.toLowerCase()] || (w.trim() ? roman2word(w) : w)).join('');
  }

  const api = { syllabify, aksharas, roman2deva };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Deva = api;
})(typeof window !== 'undefined' ? window : globalThis);
