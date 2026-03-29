"""
Vocabulary Overlap Analysis
============================
Compares IIIT-HW-Dev test split word labels against the Open Multilingual
Wordnet (OMW) to classify each word as:
  - Nepali-only      : found in Nepali Wordnet, not Hindi
  - Hindi-only       : found in Hindi Wordnet, not Nepali
  - Both (shared)    : found in both (Devanagari shared vocabulary)
  - Neither          : not found in either (names, rare words, numerals, etc.)

Requirements:
    pip install nltk pandas tabulate
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

Usage:
    Set LABELS_FILE_PATH below to your .txt label file, then run:
        python vocabulary_overlap_analysis.py
"""

import re
import unicodedata
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from tabulate import tabulate

# =============================================================================
# >>>  FILL THIS IN  <<<
LABELS_FILE_PATH = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\train.txt"
NEPALI_DICT_PATH = "D:\\DigitalPratilipiTrainingScriptv0.1\\data\\vocabulary-dictionary"# e.g. "path/to/iiit_hw_dev_test_labels.txt"
FILE_TYPE = "txt"
WORD_COLUMN = 1 
# =============================================================================

# ── NLTK data check ──────────────────────────────────────────────────────────
def ensure_nltk_data():
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            print(f"[setup] Downloading NLTK package: {pkg}")
            nltk.download(pkg, quiet=True)

# ── Label file loader ─────────────────────────────────────────────────────────
def load_labels(path: str) -> pd.DataFrame:
    """
    Reads the label file and returns a DataFrame with one column: 'word'.
    Handles three common IIIT-HW-Dev formats automatically:

      Format A  — one word per line:
            नमस्ते
            धन्यवाद

      Format B  — image filename TAB word:
            img_001.png\tनमस्ते
            img_002.png\tधन्यवाद

      Format C  — image filename SPACE word (single space):
            img_001.png नमस्ते
    """
    with open(path, encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f if l.strip()]

    words = []
    for line in raw_lines:
        if "\t" in line:
            parts = line.split("\t", 1)
            words.append(parts[1].strip())
        elif " " in line and not _is_devanagari_only(line.split(" ")[0]):
            parts = line.split(" ", 1)
            words.append(parts[1].strip())
        else:
            words.append(line.strip())

    df = pd.DataFrame({"word": words})
    df = df[df["word"].str.strip() != ""]
    return df

def _is_devanagari_only(text: str) -> bool:
    return all(
        unicodedata.category(ch) in ("Lo", "Mn", "Mc", "Nd")
        and "\u0900" <= ch <= "\u097F"
        for ch in text if not ch.isspace()
    )

# ── Wordnet lookup ────────────────────────────────────────────────────────────
def build_wordnet_sets() -> tuple[set, set]:
    """
    Nepali  : loaded from the nepali-bhasa/nepali-spell vocabulary file.
              Clone the repo once and point NEPALI_DICT_PATH to
              data/vocabulary-dictionary inside it.

    Hindi   : loaded from OMW via NLTK (lang code 'hin').
              If Hindi is also missing from your omw-1.4 build,
              the function falls back to an empty set and warns you.
    """
    # ── Nepali: from the local spell-checker dictionary ───────────────────────
    if not NEPALI_DICT_PATH:
        raise ValueError(
            "NEPALI_DICT_PATH is empty. "
            "Clone https://github.com/nepali-bhasa/nepali-spell and set "
            "NEPALI_DICT_PATH to the path of data/vocabulary-dictionary inside it."
        )
    print(f"[wordnet] Loading Nepali vocabulary from: {NEPALI_DICT_PATH}")
    with open(NEPALI_DICT_PATH, encoding="utf-8") as f:
        nepali_lemmas = {
            unicodedata.normalize("NFC", line.strip())
            for line in f
            if line.strip()
        }
    print(f"          Nepali dictionary words loaded : {len(nepali_lemmas):,}")

    # ── Hindi: from Open Multilingual Wordnet ─────────────────────────────────
    print("[wordnet] Loading Hindi lemmas from OMW ...")
    try:
        hindi_lemmas = set(wn.all_lemma_names(lang="hin"))
        print(f"          Hindi  OMW lemmas loaded      : {len(hindi_lemmas):,}")
    except Exception as e:
        print(f"          [warning] Hindi OMW failed ({e}). Hindi-only column will be empty.")
        hindi_lemmas = set()

    return nepali_lemmas, hindi_lemmas

# ── Word normalisation ────────────────────────────────────────────────────────
def normalise(word: str) -> str:
    """NFC-normalise and strip punctuation/numerals for lookup."""
    word = unicodedata.normalize("NFC", word)
    word = re.sub(r"[।॥,\.!?\-\d०-९]+", "", word)
    return word.strip()

def is_numeric_or_punctuation(word: str) -> bool:
    cleaned = re.sub(r"[\s।॥,\.!?\-]+", "", word)
    return all(ch in "0123456789०१२३४५६७८९" for ch in cleaned) or cleaned == ""

# ── Classification ────────────────────────────────────────────────────────────
def classify_word(word: str, nepali_set: set, hindi_set: set) -> str:
    if is_numeric_or_punctuation(word):
        return "numeral/punctuation"
    norm = normalise(word)
    if not norm:
        return "numeral/punctuation"
    in_nep = norm in nepali_set
    in_hin = norm in hindi_set
    if in_nep and in_hin:
        return "shared (both)"
    elif in_nep:
        return "nepali-only"
    elif in_hin:
        return "hindi-only"
    else:
        return "neither"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not LABELS_FILE_PATH:
        raise ValueError(
            "LABELS_FILE_PATH is empty. "
            "Please set it to the path of your label .txt file at the top of this script."
        )

    ensure_nltk_data()

    print(f"\n[data] Loading labels from: {LABELS_FILE_PATH}")
    df = load_labels(LABELS_FILE_PATH)
    print(f"[data] Total label entries loaded: {len(df):,}")

    total_entries = len(df)
    df_unique = df.drop_duplicates(subset="word").copy()
    total_unique = len(df_unique)
    print(f"[data] Unique word types: {total_unique:,}")

    nepali_set, hindi_set = build_wordnet_sets()

    print("\n[analysis] Classifying words ...")
    df_unique["category"] = df_unique["word"].apply(
        lambda w: classify_word(w, nepali_set, hindi_set)
    )
    df["category"] = df["word"].apply(
        lambda w: classify_word(w, nepali_set, hindi_set)
    )

    # ── Summary tables ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VOCABULARY OVERLAP ANALYSIS — RESULTS")
    print("=" * 60)

    # Token-level (with repetitions, i.e. total occurrences)
    token_counts = df["category"].value_counts()
    token_pct    = (token_counts / total_entries * 100).round(2)
    token_table  = pd.DataFrame({
        "Category":   token_counts.index,
        "Tokens":     token_counts.values,
        "% of Total": token_pct.values,
    })

    # Type-level (unique words only)
    type_counts = df_unique["category"].value_counts()
    type_pct    = (type_counts / total_unique * 100).round(2)
    type_table  = pd.DataFrame({
        "Category":       type_counts.index,
        "Unique Words":   type_counts.values,
        "% of Unique":    type_pct.values,
    })

    print("\n── Token-level (total label occurrences) ──")
    print(tabulate(token_table, headers="keys", tablefmt="rounded_outline", showindex=False))
    print(f"\n   Total tokens analysed : {total_entries:,}")

    print("\n── Type-level (unique word forms) ──")
    print(tabulate(type_table, headers="keys", tablefmt="rounded_outline", showindex=False))
    print(f"\n   Total unique types    : {total_unique:,}")

    # ── Key metrics for the report ────────────────────────────────────────────
    def safe_pct(cat, df_col):
        cnt = (df_col == cat).sum()
        return cnt, round(cnt / len(df_col) * 100, 1)

    nep_tok, nep_tok_p = safe_pct("nepali-only",   df["category"])
    hin_tok, hin_tok_p = safe_pct("hindi-only",    df["category"])
    both_tok,both_tok_p= safe_pct("shared (both)", df["category"])
    nei_tok, nei_tok_p = safe_pct("neither",        df["category"])

    nep_typ, nep_typ_p = safe_pct("nepali-only",   df_unique["category"])
    hin_typ, hin_typ_p = safe_pct("hindi-only",    df_unique["category"])

    print("\n── Report-ready summary ──")
    print(f"""
  Nepali-only   tokens : {nep_tok:>6,}  ({nep_tok_p}%)
  Hindi-only    tokens : {hin_tok:>6,}  ({hin_tok_p}%)
  Shared        tokens : {both_tok:>6,}  ({both_tok_p}%)
  Neither       tokens : {nei_tok:>6,}  ({nei_tok_p}%)

  Nepali-only   types  : {nep_typ:>6,}  ({nep_typ_p}% of unique words)
  Hindi-only    types  : {hin_typ:>6,}  ({hin_typ_p}% of unique words)
""")

    # ── Sample words per category ─────────────────────────────────────────────
    print("── Sample words per category (up to 10 each) ──\n")
    for cat in ["nepali-only", "hindi-only", "shared (both)", "neither"]:
        samples = df_unique[df_unique["category"] == cat]["word"].head(10).tolist()
        print(f"  {cat:20s}: {',  '.join(samples) if samples else '(none)'}")

    # ── Save detailed results ─────────────────────────────────────────────────
    out_path = "vocabulary_overlap_results.csv"
    df_unique[["word", "category"]].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[output] Full word-level results saved to: {out_path}")
    print("         Use this CSV to manually verify edge cases before citing numbers.\n")

if __name__ == "__main__":
    main()