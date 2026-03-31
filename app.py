"""
app.py — Digital Pratilipi: Nepali Handwritten OCR
Kantipur Engineering College — CT 755 Major Project  v12.1

Flask API entry point.  All logic lives in src/:
  src/preprocessing.py  — image preprocessing pipeline (v12)
  src/detection.py      — line + word detection (CC + VPP v3)
  src/ocr.py            — TrOCR model loader and inference
  src/vocabulary.py     — Nepali vocabulary + BK-tree spell correction
  src/postprocessing.py — NFC normalisation, spell correction, TTS
  src/pipeline.py       — end-to-end orchestration + visualisation

Routes:
  GET  /              → serve index.html
  GET  /health        → service status JSON
  POST /ocr           → run full OCR pipeline
  POST /ocr_full      → alias for /ocr
  POST /tts           → text-to-speech (POST JSON {"text": "…"})
  GET  /audio         → download last generated audio
  POST /detect_viz    → detection visualisation only (no OCR)
  POST /vocab_check   → debug: vocabulary membership + BK neighbours
"""

import os
import logging

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image

from src.pipeline       import run_pipeline, _render_detection_viz
from src.preprocessing  import preprocess
from src.detection      import detect_words
from src.ocr            import get_trocr
from src.vocabulary     import _get_vocab, _get_bktree, _nfc
from src.postprocessing import speak_nepali

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    from src.detection import _easyocr_reader
    p, _ = get_trocr()
    vocab = _get_vocab()
    return jsonify({
        "status":       "ok",
        "trocr":        p is not None,
        "easyocr":      _easyocr_reader is not None,
        "vocab_size":   len(vocab),
        "bktree_ready": _get_bktree() is not None,
    })


@app.route("/ocr",      methods=["POST"])
@app.route("/ocr_full", methods=["POST"])
def ocr_endpoint():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400
        img   = Image.open(request.files["image"].stream).convert("RGB")
        speak = request.form.get("speak", "0") == "1"
        return jsonify(run_pipeline(img, speak=speak))
    except Exception as e:
        logger.exception("Pipeline error")
        return jsonify({"error": str(e)}), 500


@app.route("/tts", methods=["POST"])
def tts_endpoint():
    try:
        text = request.get_json(force=True).get("text", "")
        if not text:
            return jsonify({"error": "No text"}), 400
        path = speak_nepali(text)
        if not path:
            return jsonify({"error": "TTS failed"}), 500
        return send_file(path, mimetype="audio/mpeg", download_name="speech.mp3")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio")
def get_audio():
    if not os.path.exists("output_audio.mp3"):
        return jsonify({"error": "No audio"}), 404
    return send_file("output_audio.mp3", mimetype="audio/mpeg",
                     download_name="speech.mp3")


@app.route("/detect_viz", methods=["POST"])
def detect_viz():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400
        pil = Image.open(request.files["image"].stream).convert("RGB")
        img_bgr, binary = preprocess(pil)
        boxes, line_groups = detect_words(img_bgr, binary=binary)
        image_b64 = _render_detection_viz(img_bgr, line_groups, binary=binary)
        return jsonify({
            "image_b64":  image_b64,
            "word_count": len(boxes),
            "line_count": len(line_groups),
            "boxes": [
                {"x": int(b["x"]), "y": int(b["y"]),
                 "w": int(b["w"]), "h": int(b["h"])}
                for b in boxes
            ],
        })
    except Exception as e:
        logger.exception("detect_viz error")
        return jsonify({"error": str(e)}), 500


@app.route("/vocab_check", methods=["POST"])
def vocab_check():
    """Debug: POST {"word":"…"} → vocabulary membership + BK-tree neighbours."""
    try:
        data     = request.get_json(force=True)
        word     = _nfc(data.get("word", "").strip())
        if not word:
            return jsonify({"error": "No word"}), 400
        vocab    = _get_vocab()
        in_vocab = word in vocab
        tree     = _get_bktree()
        neighbours = []
        if tree:
            hits = tree.search(word, max_dist=2)
            hits.sort()
            neighbours = [{"dist": d, "word": w} for d, w in hits[:10]]
        return jsonify({"word": word, "in_vocab": in_vocab,
                        "neighbours": neighbours})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Digital Pratilipi OCR  v12.1  [Modular Architecture]")
    print("=" * 60)
    print(f"  TROCR_MODEL : {os.getenv('TROCR_MODEL', 'aayushpuri01/TrOCR-Devanagari')}")
    print(f"  VOCAB_DIR   : {os.getenv('NEPALI_VOCAB_DIR', 'data/')}")
    print(f"  URL         : http://localhost:5000")
    print("=" * 60 + "\n")

    # Warm up all singletons at startup
    get_trocr()
    _get_vocab()
    _get_bktree()

    app.run(host="0.0.0.0", port=5000, debug=False)