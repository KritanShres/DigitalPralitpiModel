"""
src/vocabulary.py — Nepali vocabulary loading and BK-tree spell correction

Vocabulary sources:
  VOCAB_DIR/vocabulary-dictionary  — nepali-bhasa/nepali-spell (~75k forms)
  VOCAB_DIR/vocabulary-corpus      — frequency corpus

BK-tree nearest-neighbour search on Unicode code-points (Levenshtein).
"""

import logging
import os
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_DEFAULT_VOCAB_DIR    = Path(__file__).resolve().parent.parent / "data"
VOCAB_DIR             = Path(os.getenv("NEPALI_VOCAB_DIR", str(_DEFAULT_VOCAB_DIR)))
VOCAB_DICTIONARY_FILE = VOCAB_DIR / "vocabulary-dictionary"
VOCAB_CORPUS_FILE     = VOCAB_DIR / "vocabulary-corpus"

# ── Module-level singletons ────────────────────────────────────────────────
_VOCAB:   set[str] | None = None
_BK_TREE: object          = None

_RE_NON_WORD   = re.compile(r'^[०-९0-9।,.!?\s\u0964\u0965]+$')
_RE_DEVANAGARI = re.compile(r'[\u0900-\u097F]')

# Built-in seed vocabulary (used when no external files are found)
_SEED_WORDS = {
    "म","मेरो","हाम्रो","तिमी","तिम्रो","उ","उसको","यो","त्यो",
    "हामी","तपाई","आफ्नो","छ","छन्","हो","हुन्","गर्छ","गयो",
    "आयो","भयो","गर्नु","आउनु","जानु","खानु","पिउनु","पढ्नु",
    "लेख्नु","हेर्नु","बोल्नु","सुन्नु","भन्नु","हुन्छ","गर्छु",
    "जान्छु","आउँछु","पर्छ","नाम","घर","देश","नेपाल","मान्छे",
    "आमा","बाबा","दाजु","भाइ","दिदी","बहिनी","साथी","स्कुल",
    "किताब","कलम","पानी","खाना","दूध","चिया","काम","पैसा",
    "समय","दिन","रात","बिहान","साँझ","सहर","गाउँ","जीवन",
    "संसार","मन","राम्रो","नराम्रो","ठूलो","सानो","धेरै",
    "थोरै","नया","खुसी","दुखी","सुन्दर","रमाइलो",
    "मलाई","हामीलाई","तिमीलाई","उसलाई",
    "मा","को","का","की","लाई","बाट","देखि","सम्म","साथ",
    "पनि","नै","र","तर","वा","भने","भनेर","छु","छौ","छन",
    "थियो","थिए","थिएन","छैन","गरे","गरेको","भएको",
    "।","?","!",",",".",
}


# ============================================================
# UTILITIES
# ============================================================

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _load_vocab_file(path: Path) -> set[str]:
    words: set[str] = set()
    if not path.exists():
        logger.warning(f"Vocabulary file not found: {path}")
        return words
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                w = _nfc(raw.strip())
                if w and not w.startswith("#"):
                    words.add(w)
        logger.info(f"Loaded {len(words):,} words from {path.name}")
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return words


# ============================================================
# VOCABULARY
# ============================================================

def _get_vocab() -> set[str]:
    global _VOCAB
    if _VOCAB is None:
        logger.info("Loading Nepali vocabulary …")
        _VOCAB = (_load_vocab_file(VOCAB_DICTIONARY_FILE) |
                  _load_vocab_file(VOCAB_CORPUS_FILE))
        if not _VOCAB:
            logger.warning("No external vocabulary files — using built-in seed.")
            _VOCAB = {_nfc(w) for w in _SEED_WORDS}
        logger.info(f"Total vocabulary size: {len(_VOCAB):,} words")
    return _VOCAB


# ============================================================
# BK-TREE
# ============================================================

class _BKTree:
    """
    Minimal BK-tree over Unicode strings using Levenshtein distance.
    Supports efficient nearest-neighbour queries in sub-linear time.
    """
    __slots__ = ("word", "children")

    def __init__(self, word: str):
        self.word     = word
        self.children: dict[int, "_BKTree"] = {}

    @staticmethod
    def _lev(a: str, b: str) -> int:
        if a == b:  return 0
        if not a:   return len(b)
        if not b:   return len(a)
        if len(a) < len(b): a, b = b, a
        prev = list(range(len(b) + 1))
        for ca in a:
            curr = [prev[0] + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j+1]+1, curr[j]+1,
                                prev[j] + (0 if ca == cb else 1)))
            prev = curr
        return prev[-1]

    def insert(self, word: str) -> None:
        node = self
        while True:
            d = self._lev(node.word, word)
            if d == 0: return
            if d not in node.children:
                node.children[d] = _BKTree(word)
                return
            node = node.children[d]

    def search(self, query: str, max_dist: int) -> list[tuple[int, str]]:
        results, stack = [], [self]
        while stack:
            node = stack.pop()
            d    = self._lev(node.word, query)
            if d <= max_dist:
                results.append((d, node.word))
            lo, hi = d - max_dist, d + max_dist
            for k, child in node.children.items():
                if lo <= k <= hi:
                    stack.append(child)
        return results


def _get_bktree() -> _BKTree | None:
    global _BK_TREE
    if _BK_TREE is None:
        vocab = _get_vocab()
        if not vocab:
            return None
        logger.info(f"Building BK-tree for {len(vocab):,} words …")
        words = list(vocab)
        root  = _BKTree(words[0])
        for w in words[1:]:
            root.insert(w)
        _BK_TREE = root
        logger.info("BK-tree ready.")
    return _BK_TREE


# ============================================================
# SPELL CORRECTION
# ============================================================

def spell_correct(word: str, max_dist: int = 1) -> tuple[str, bool]:
    """
    Return (corrected_word, was_changed).

    Strategy:
      1. NFC-normalise.
      2. Skip short, non-Devanagari, or punctuation tokens.
      3. O(1) set-membership → already correct.
      4. BK-tree search (max_dist+1 for words len>=6, capped at 2).
      5. Prefer dictionary words over corpus-only; tie-break alphabetically.
    """
    word = _nfc(word)
    if not word or len(word) < 3:         return word, False
    if _RE_NON_WORD.match(word):          return word, False
    if not _RE_DEVANAGARI.search(word):   return word, False

    vocab = _get_vocab()
    if word in vocab: return word, False

    effective_dist = min(max_dist + (1 if len(word) >= 6 else 0), 2)
    tree = _get_bktree()
    if tree is None: return word, False

    candidates = tree.search(word, effective_dist)
    if not candidates: return word, False

    min_d = min(c[0] for c in candidates)
    best  = sorted(
        [w for d, w in candidates if d == min_d],
        key=lambda w: (0 if w in vocab else 1, w)
    )
    corrected = best[0]
    return (corrected, True) if corrected != word else (word, False)