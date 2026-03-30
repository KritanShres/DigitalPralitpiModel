"""
crop_inspector.py — Digital Pratilipi: Crop Inspector
Kantipur Engineering College — CT 755 Major Project

A standalone debug tool that visualises every word crop sent to TrOCR.
Imports and reuses the exact same functions from app.py so you see
precisely what the model receives — no approximations.

Run:
    python crop_inspector.py
    → http://localhost:5001
"""

import base64, io, sys, os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# ── Reuse functions directly from app.py ──────────────────────────────────
# Insert the project root on sys.path so we can import from app.py
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import (
    preprocess,
    detect_words,
    crop_word,
)

# ─────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    html_path = PROJECT_ROOT / "crop_inspector.html"
    if not html_path.exists():
        return (
            f"<pre>crop_inspector.html not found.\n"
            f"Expected it at: {html_path}\n"
            f"Make sure both files are in the same folder.</pre>",
            404,
        )
    return send_from_directory(str(PROJECT_ROOT), "crop_inspector.html")


@app.route("/inspect", methods=["POST"])
def inspect():
    """
    POST an image → returns every word crop (post-preprocessing) as base64,
    plus the detection viz and the preprocessed binary, so you can see
    exactly what TrOCR receives for each word.

    Response JSON:
    {
      "binary_b64"  : "<PNG, preprocessed binary, black ink on white>",
      "viz_b64"     : "<JPEG, bounding boxes drawn on binary>",
      "line_count"  : int,
      "word_count"  : int,
      "crops"       : [
        {
          "line"    : int,          # 1-indexed line number
          "word"    : int,          # 1-indexed word number within line
          "box"     : {x, y, w, h},
          "img_b64" : "<PNG, the exact crop sent to TrOCR>",
          "width"   : int,
          "height"  : int,
        },
        ...
      ]
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        pil_image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    try:
        # ── Step 1: preprocess (same as main pipeline) ────────────────────
        import cv2, numpy as np
        img_bgr, binary = preprocess(pil_image)

        # ── Step 2: detect words (same as main pipeline) ──────────────────
        boxes, line_groups = detect_words(img_bgr, binary=binary)

        # ── Step 3: encode detection viz (boxes on binary) ────────────────
        import cv2
        vis = cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)
        COLORS = [(60,200,100),(60,160,220),(220,80,60),
                  (220,160,40),(180,60,220),(40,200,200)]
        for li, line in enumerate(line_groups):
            col = COLORS[li % len(COLORS)]
            for wi, box in enumerate(line):
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                cv2.rectangle(vis, (x, y), (x+w, y+h), col, 2)
                cv2.putText(vis, f"L{li+1}W{wi+1}", (x, max(y-4, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                            cv2.LINE_AA)
        _, vbuf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        viz_b64 = base64.b64encode(vbuf.tobytes()).decode()

        # ── Step 4: encode preprocessed binary for display ────────────────
        binary_display = cv2.bitwise_not(binary)
        _, bbuf = cv2.imencode(".png", binary_display)
        binary_b64 = base64.b64encode(bbuf.tobytes()).decode()

        # ── Step 5: generate every crop (same as main pipeline) ───────────
        crops = []
        for li, line in enumerate(line_groups):
            for wi, box in enumerate(line):
                # crop_word returns a PIL RGB image — exactly what TrOCR gets
                crop_pil = crop_word(img_bgr, box)

                buf = io.BytesIO()
                crop_pil.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()

                crops.append({
                    "line":    li + 1,
                    "word":    wi + 1,
                    "box":     {k: int(v) for k, v in box.items()},
                    "img_b64": img_b64,
                    "width":   int(crop_pil.width),
                    "height":  int(crop_pil.height),
                })

        return jsonify({
            "binary_b64": binary_b64,
            "viz_b64":    viz_b64,
            "line_count": len(line_groups),
            "word_count": len(boxes),
            "crops":      crops,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Digital Pratilipi — Crop Inspector")
    print("="*55)
    print(f"  URL        : http://localhost:5001")
    print(f"  Script dir : {PROJECT_ROOT}")
    print(f"  HTML file  : {PROJECT_ROOT / 'crop_inspector.html'}")
    print(f"  HTML found : {(PROJECT_ROOT / 'crop_inspector.html').exists()}")
    print(f"  app.py found: {(PROJECT_ROOT / 'app.py').exists()}")
    print("  This tool shows the exact crops sent to TrOCR.")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=False)