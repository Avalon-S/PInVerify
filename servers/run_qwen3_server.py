import base64
import io
import argparse
from PIL import Image
from flask import Flask, request, jsonify
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

# Monkey patch for PyTorch 2.2 compatibility with transformers 4.57
if not hasattr(torch, 'compiler') or not hasattr(torch.compiler, 'is_compiling'):
    if not hasattr(torch, 'compiler'):
        torch.compiler = type('compiler', (), {})()
    torch.compiler.is_compiling = lambda: False

# ================= Model loading =================
MODEL_PATH = "./models/Qwen3-VL-4B-Instruct"  # adjust to your local path

print("Loading Qwen3-VL model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # enable FlashAttention
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)  # disable the fast processor
print("Qwen3-VL model loaded")

app = Flask(__name__)

def decode_image_base64(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def qwen3_generate(messages, max_new_tokens=512):
    """
    Qwen3-VL inference.
    messages: [{"role": "user", "content": [...]}]
    """
    with torch.inference_mode():
        # Qwen3-VL uses the newer apply_chat_template API
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy decoding, equivalent to temperature=0
            temperature=None,       # unused when do_sample=False
            top_p=None,             # unused when do_sample=False
            top_k=None,             # unused when do_sample=False
            repetition_penalty=1.0,
        )
        
        # keep only the generated part, dropping the prompt
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()


# ================ Text only: /qwen-text =================
@app.route("/qwen-text", methods=["POST"])
def qwen_text():
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            return jsonify({"error": "Missing 'prompt'"}), 400

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }]
        text = qwen3_generate(messages)
        return jsonify({"text": text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ================ Image + text: /qwen-vl =================
@app.route("/qwen-vl", methods=["POST"])
def qwen_vl():
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        image_b64 = data.get("image", None)
        if not prompt or not image_b64:
            return jsonify({"error": "Missing 'prompt' or 'image' (base64)"}), 400

        image_pil = decode_image_base64(image_b64)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": prompt}
            ]
        }]
        text = qwen3_generate(messages)
        return jsonify({"text": text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ================ Multi-image: /qwen-vl-multi =================
@app.route("/qwen-vl-multi", methods=["POST"])
def qwen_vl_multi():
    """
    Multi-image inference endpoint.
    Input: {"prompt": "...", "images": ["base64_1", "base64_2", ...]}
    """
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        images_b64 = data.get("images", [])
        
        if not prompt or not images_b64:
            return jsonify({"error": "Missing 'prompt' or 'images' (list of base64)"}), 400
        
        # Build content with multiple images
        content = []
        for img_b64 in images_b64:
            image_pil = decode_image_base64(img_b64)
            content.append({"type": "image", "image": image_pil})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        text = qwen3_generate(messages)
        return jsonify({"text": text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182, help="Server port")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)

