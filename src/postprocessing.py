"""
src/postprocessing.py — Text postprocessing and Text-to-Speech
"""

import logging
from src.vocabulary import _nfc, spell_correct

logger = logging.getLogger(__name__)


def postprocess(lines: list[list[str]]) -> tuple[str, list[dict]]:
    """
    NFC-normalise every word, run spell correction, and join lines.

    Returns:
      final_text  — newline-joined corrected text
      corrections — list of {"original": str, "corrected": str} dicts
    """
    corrections: list[dict] = []
    result_lines: list[str] = []

    for line in lines:
        out_words: list[str] = []
        for word in line:
            word = _nfc(word)
            corrected, changed = spell_correct(word)
            if changed:
                corrections.append({"original": word, "corrected": corrected})
            out_words.append(corrected)
        result_lines.append(" ".join(out_words))

    return "\n".join(result_lines), corrections


def speak_nepali(text: str, path: str = "output_audio.mp3") -> str | None:
    """
    Convert Nepali text to speech using gTTS and save to path.
    Returns the output path on success, None on failure.
    """
    if not text.strip():
        return None
    try:
        from gtts import gTTS
        gTTS(text=text, lang="ne", slow=False).save(path)
        return path
    except Exception as e:
        logger.error(f"TTS: {e}")
        return None