# -*- coding: utf-8 -*-
"""
HTTP wrapper around GroundingDINO, used by PInVerify as the detection frontend
(method.bbox_mode=dino).

It imports GroundingDINO's own modules and reads the config and checkpoint by
relative path, so run it from inside a GroundingDINO checkout:

    git clone https://github.com/IDEA-Research/GroundingDINO
    cd GroundingDINO && pip install -e .
    mkdir -p weights && cd weights
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    cd ..
    cp /path/to/PInVerify/servers/run_groundingdino_server.py .
    python run_groundingdino_server.py --port 12183

The multi-GPU launchers in scripts/ expect one instance per GPU, on port
12183 + i*100.
"""
import torch
# Monkey patch for PyTorch 2.2 compatibility with transformers 4.57
if not hasattr(torch, 'compiler') or not hasattr(torch.compiler, 'is_compiling'):
    if not hasattr(torch, 'compiler'):
        torch.compiler = type('compiler', (), {})()
    torch.compiler.is_compiling = lambda: False

import io, os, base64, json, argparse
from PIL import Image
from flask import Flask, request, jsonify

from groundingdino.datasets import transforms as T
from demo.inference_on_a_image import load_model, get_grounding_output

# === Model paths, relative to the GroundingDINO checkout ===
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
CKPT_PATH   = "weights/groundingdino_swint_ogc.pth"

print("Loading GroundingDINO ...")
model = load_model(CONFIG_PATH, CKPT_PATH)
print("GroundingDINO loaded")

app = Flask(__name__)

def decode_base64_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def load_image_from_pil(image_pil: Image.Image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

@app.route("/groundingdino", methods=["POST"])
def grounding_infer():
    try:
        data = request.get_json(force=True)
        image_b64      = data["image"]
        prompt         = str(data.get("prompt", ""))
        image_name     = str(data.get("image_name", "unknown.png"))
        box_threshold  = float(data.get("box_threshold", 0.3))
        text_threshold = float(data.get("text_threshold", 0.25))

        # decode and preprocess
        image_pil = decode_base64_image(image_b64)
        W, H = image_pil.size
        _, image_tensor = load_image_from_pil(image_pil)

        # inference
        boxes, labels, scores = get_grounding_output(
            model, image_tensor, prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # === Convert (cx,cy,w,h) in [0,1] to pixel xyxy, with fallbacks for other formats ===
        results = []
        boxes_list  = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes)
        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)

        def to_pixel_xyxy(b):
            x1 = y1 = x2 = y2 = None
            a, b1, c, d = map(float, b)

            # 1) try cxcywh first, which is what GroundingDINO returns
            if 0.0 <= a <= 1.2 and 0.0 <= b1 <= 1.2 and 0.0 <= c <= 1.2 and 0.0 <= d <= 1.2:
                # the four values are either cx,cy,w,h or normalized xyxy; decide by whether w,h look like extents
                if c >= 0 and d >= 0 and (a - c/2) <= (a + c/2) and (b1 - d/2) <= (b1 + d/2):
                    # as cxcywh
                    cx, cy, w, h = a, b1, c, d
                    x1, y1 = (cx - w/2) * W, (cy - h/2) * H
                    x2, y2 = (cx + w/2) * W, (cy + h/2) * H
                else:
                    # as normalized xyxy
                    x1, y1, x2, y2 = a * W, b1 * H, c * W, d * H
            else:
                # 2) fallback: treat as pixel xyxy
                x1, y1, x2, y2 = a, b1, c, d

            # sort and clip to the image
            x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
            return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

        for b, label, score in zip(boxes_list, labels, scores_list):
            results.append({
                "box":   to_pixel_xyxy(b),
                "label": str(label),
                "score": float(round(float(score), 4)),
            })

        return jsonify({
            "status": "ok",
            "image_name": image_name,
            "prompt": prompt,
            "image_size": [W, H],
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "num_detections": len(results),
            "results": results,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12183, help="Server port")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
