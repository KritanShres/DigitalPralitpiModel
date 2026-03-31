"""
src/ocr.py — TrOCR model loader and word-level inference
"""

import logging
import os
from PIL import Image

os.environ.setdefault("TRANSFORMERS_VERBOSITY",        "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM",        "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")   

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TROCR_MODEL = os.getenv("TROCR_MODEL", "aayushpuri01/TrOCR-Devanagari")

_processor = None
_model     = None


def get_trocr():
    """Lazily load the TrOCR processor and model (cached after first call)."""
    global _processor, _model
    if _processor is None:
        try:
            from transformers import (
                VisionEncoderDecoderModel, AutoTokenizer,
                ViTImageProcessor, TrOCRProcessor,
            )
            print(f"Loading TrOCR mode: from ./model", flush=True)
            feature_extractor = ViTImageProcessor.from_pretrained(TROCR_MODEL)
            tokenizer         = AutoTokenizer.from_pretrained(TROCR_MODEL)
            _processor        = TrOCRProcessor(
                image_processor=feature_extractor, tokenizer=tokenizer)
            _model            = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
            _model.eval()
            print("✓ TrOCR loaded successfully.")
        except Exception as e:
            logger.error(f"TrOCR load failed: {e}")
            _processor = _model = None
    return _processor, _model


def trocr_predict(word_img: Image.Image) -> str:
    """Run a single word image through TrOCR and return the decoded string."""
    import torch
    proc, model = get_trocr()
    if not proc:
        return ""
    try:
        pixel_values = proc(
            images=word_img.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, max_new_tokens=48, num_beams=5,
                early_stopping=True, repetition_penalty=1.2, length_penalty=1.0,
            )
        return proc.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        logger.warning(f"TrOCR predict error: {e}")
        return ""