P1 gold set: correcting the pipeline's predictions
=================================================

Install Praat (https://www.praat.org).  For each folder:
  1. Praat > Open > Read from file: vocals.wav AND <song_id>.TextGrid
     (mix.wav is the original song, to listen when the vocal is unclear).
  2. Select both > View & Edit.  Play a stretch with Tab / click the bars.
  3. Correct the tiers (boundaries: drag; text: click the interval, type):
       words      every sung word, in the order sung, spelled as sung
                  (Devanagari for Hindi, Latin for English words).  Repeats and
                  half lines are written as sung.
       syllables  boundaries at syllable onsets; you only need to fix the TEXT
                  where you changed the word.
       vocables   every aakar / humming stretch that is not a word:
                  आ / ओ / ई / ए / ऊ / म्म (the vowel heard), or  mel  when it is a
                  long vowel continuing the previous syllable.
       singers    who sings: A, B, C ... (same letter = same person within this
                  song), AB when two or more voices sing at once, empty = nobody.
  4. Save: File > Save TextGrid as text file, into  ../corrected/<song_id>.TextGrid
     (same file name).  Copy the corrected/ folder back to Prajna:
        P1/gold/corrected/

Rough effort: 15-25 min per 75 s window.  Precision matters more than speed:
when you can't tell, leave the prediction as it is and add a word "?" in words.
