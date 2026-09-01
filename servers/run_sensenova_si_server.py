"""
SenseNova-SI-1.2-InternVL3-8B Server
Drop-in replacement for the Qwen server: same HTTP interface.

Activate the SenseNova conda environment first:
    conda activate sensenova
    export PYTHONPATH=./SenseNova-SI:$PYTHONPATH

Launch:
    python run_sensenova_si_server.py

Port: 12182 (same as Qwen; run only one of them)
"""

# ============ Offline mode (must run before any transformers import) ============
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import base64
import io
import torch
from PIL import Image
from flask import Flask, request, jsonify

# SenseNova-SI uses its own API
from sensenova_si import get_model

# ================= Model loading =================
MODEL_PATH = "./models/SenseNova-SI-1.2-InternVL3-8B"

print(f"Loading SenseNova-SI model from: {MODEL_PATH}")
print("Note: OFFLINE mode enabled - loading from local files only")

# load through SenseNova-SI's get_model
model = get_model(MODEL_PATH)

print("SenseNova-SI-1.2-InternVL3-8B model loaded")

app = Flask(__name__)

def decode_image_base64(b64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def sensenova_generate(image_path_or_pil: Image.Image, prompt: str) -> str:
    """
    SenseNova-SI image + text generation.
    Uses the official model.generate() API.
    """
    import tempfile
    import os
    
    # SenseNova-SI wants a file path, so a PIL image is written to a temp file first
    if isinstance(image_path_or_pil, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image_path_or_pil.save(f, format="JPEG")
            image_path = f.name
        try:
            # make sure the prompt carries an <image> token
            if "<image>" not in prompt:
                prompt = "<image>\n" + prompt
            response = model.generate(prompt, images=[image_path])
            return response.strip()
        finally:
            os.unlink(image_path)  # clean up the temp file
    else:
        # already a path
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt
        response = model.generate(prompt, images=[image_path_or_pil])
        return response.strip()


def sensenova_generate_text_only(prompt: str) -> str:
    """
    Text-only generation (no image).
    """
    response = model.generate(prompt, images=[])
    return response.strip()



# ============ Text only: /qwen-text (name kept for compatibility) ============
@app.route("/qwen-text", methods=["POST"])
def qwen_text():
    """Text-only endpoint, compatible with existing client."""
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            return jsonify({"error": "Missing 'prompt'"}), 400

        text = sensenova_generate_text_only(prompt)
        return jsonify({"text": text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============ Image + text: /qwen-vl (name kept for compatibility) ============
@app.route("/qwen-vl", methods=["POST"])
def qwen_vl():
    """Vision-Language endpoint, compatible with existing client."""
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        image_b64 = data.get("image", None)
        
        if not prompt or not image_b64:
            return jsonify({"error": "Missing 'prompt' or 'image' (base64)"}), 400

        image_pil = decode_image_base64(image_b64)
        text = sensenova_generate(image_pil, prompt)
        return jsonify({"text": text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ================ Native endpoint names (optional) ================
@app.route("/internvl-vl", methods=["POST"])
def internvl_vl():
    """Native InternVL endpoint name."""
    return qwen_vl()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "SenseNova-SI-1.2-InternVL3-8B"})


# ================ Multi-image: /qwen-vl-multi ================
@app.route("/qwen-vl-multi", methods=["POST"])
def qwen_vl_multi():
    """
    Multi-image inference endpoint for SenseNova-SI.
    Input: {"prompt": "...", "images": ["base64_1", "base64_2", ...]}
    """
    import tempfile
    import os
    
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        images_b64 = data.get("images", [])
        
        if not prompt or not images_b64:
            return jsonify({"error": "Missing 'prompt' or 'images' (list of base64)"}), 400
        
        # Save all images to temp files
        temp_paths = []
        try:
            for img_b64 in images_b64:
                image_pil = decode_image_base64(img_b64)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    image_pil.save(f, format="JPEG")
                    temp_paths.append(f.name)
            
            # Build prompt with multiple <image> tokens
            image_tokens = "<image>\n" * len(temp_paths)
            full_prompt = image_tokens + prompt
            
            response = model.generate(full_prompt, images=temp_paths)
            return jsonify({"text": response.strip()})
        finally:
            # Clean up temp files
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182, help="Server port")
    args = parser.parse_args()

    print(f"Starting SenseNova-SI Server on port {args.port}...")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


