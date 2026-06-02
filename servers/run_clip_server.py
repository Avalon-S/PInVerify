#!/usr/bin/env python3
"""
CLIP / SigLIP2 Similarity Server.

Provides a REST endpoint for computing similarity between
an image and a list of text descriptions.

- CLIP: normalized cosine similarity (range [-1, 1])
- SigLIP/SigLIP2: sigmoid probability (range [0, 1])

Model type is auto-detected from the loaded checkpoint.

Usage:
    python servers/run_clip_server.py --port 12184
    python servers/run_clip_server.py --port 12184 --model openai/clip-vit-large-patch14
    python servers/run_clip_server.py --port 12184 --model /path/to/siglip2-so400m-patch14-384
"""

import base64
import io
import argparse

import torch
from PIL import Image
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoProcessor

app = Flask(__name__)
model = None
processor = None
device = None
is_siglip = False


def decode_image_base64(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


@app.route("/clip-score", methods=["POST"])
def clip_score():
    """Compute similarity between image and text queries.

    Input JSON:
        image: base64-encoded image
        texts: list of text descriptions

    Output JSON:
        scores: list of similarities (one per text)
        max_score: maximum similarity
    """
    try:
        data = request.get_json(force=True) or {}
        image_b64 = data.get("image")
        texts = data.get("texts", [])

        if not image_b64:
            return jsonify({"error": "Missing 'image'"}), 400
        if not texts:
            return jsonify({"error": "Missing 'texts'"}), 400

        img = decode_image_base64(image_b64)

        padding = "max_length" if is_siglip else True
        inputs = processor(text=texts, images=img, return_tensors="pt", padding=padding)
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            if is_siglip:
                # SigLIP/SigLIP2: sigmoid on logits (includes learned temperature + bias)
                scores = torch.sigmoid(outputs.logits_per_image)[0].cpu().tolist()
            else:
                # CLIP: normalized cosine similarity
                img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                scores = (img_emb @ txt_emb.T)[0].cpu().tolist()

        return jsonify({"scores": scores, "max_score": max(scores)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    scoring = "sigmoid" if is_siglip else "cosine"
    return jsonify({
        "status": "ok",
        "model": model.config._name_or_path if model else "not loaded",
        "scoring": scoring,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP / SigLIP2 Similarity Server")
    parser.add_argument("--port", type=int, default=12184, help="Server port")
    parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14",
                        help="CLIP or SigLIP2 model name or path")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    # Auto-detect SigLIP/SigLIP2
    model_class = type(model).__name__.lower()
    is_siglip = "siglip" in model_class
    scoring = "sigmoid" if is_siglip else "cosine"
    print(f"Model type: {type(model).__name__} (scoring: {scoring})")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Model loaded on GPU")
    else:
        device = None
        print(f"Model loaded on CPU")

    print(f"Starting server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port)
