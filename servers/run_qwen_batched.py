
import base64
import io
import torch
import time
import threading
import queue
import uuid
from typing import List, Dict, Any

# Monkey patch for PyTorch 2.2 compatibility
if not hasattr(torch, 'compiler') or not hasattr(torch.compiler, 'is_compiling'):
    if not hasattr(torch, 'compiler'):
        torch.compiler = type('compiler', (), {})()
    torch.compiler.is_compiling = lambda: False

from PIL import Image
from flask import Flask, request, jsonify
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ================= Configuration =================
MODEL_PATH = "./models/Qwen3-VL-4B-Instruct"
ADAPTER_PATH = None      # Set via --adapter flag for LoRA models
BATCH_SIZE = 8           # Optimized for 4B model
BATCH_TIMEOUT = 0.05     # 50ms wait
MAX_NEW_TOKENS = 2048

# ================= Model Loading =================
def load_model(model_path, adapter_path=None):
    print(f"Loading Qwen3-VL from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("LoRA adapter merged successfully.")

    print("Qwen3-VL Loaded Successfully.")
    return model, processor

try:
    model, processor = load_model(MODEL_PATH, ADAPTER_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

app = Flask(__name__)

# ================= Dynamic Batching Logic =================

class RequestItem:
    def __init__(self, messages):
        self.id = str(uuid.uuid4())
        self.messages = messages
        self.result_event = threading.Event()
        self.result_text = None
        self.error = None

request_queue = queue.Queue()

def batch_worker():
    print("Batch Processing Worker Started.")
    while True:
        batch_items: List[RequestItem] = []
        
        # 1. Fetch first item
        try:
            item = request_queue.get()
            batch_items.append(item)
        except Exception as e:
            print(f"Worker error: {e}")
            continue

        # 2. Opportunistic Fetch
        start_wait = time.time()
        while len(batch_items) < BATCH_SIZE:
            remaining_time = BATCH_TIMEOUT - (time.time() - start_wait)
            if remaining_time <= 0:
                break
            try:
                item = request_queue.get(timeout=remaining_time)
                batch_items.append(item)
            except queue.Empty:
                break
        
        # 3. Process
        if batch_items:
            process_batch(batch_items)

def process_batch(batch: List[RequestItem]):
    try:
        print(f"[Server] Processing Batch of Size: {len(batch)}")
        batch_messages = [item.messages for item in batch]
        
        # Prepare inputs using apply_chat_template with padding
        with torch.inference_mode():
            inputs = processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(model.device)

            # Generate
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
            )
            
            # Decode: Trim input tokens and decode
            # output_ids contains [input_ids + generated_ids]
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            
            decoded_texts = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Distribute results
            for i, text in enumerate(decoded_texts):
                item = batch[i]
                item.result_text = text.strip()
                item.result_event.set()

    except Exception as e:
        print(f"Batch Inference Failed: {e}")
        import traceback
        traceback.print_exc()
        for item in batch:
            item.error = str(e)
            item.result_event.set()

# Start Worker
worker_thread = threading.Thread(target=batch_worker, daemon=True)
worker_thread.start()


# ================= Helper Functions =================
def decode_image_base64(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def enqueue_and_wait(messages):
    # Just wrap messages and enqueue, pre-processing happens in worker
    item = RequestItem(messages)
    
    request_queue.put(item)
    
    # Wait
    item.result_event.wait(timeout=60)
    
    if item.error:
        raise RuntimeError(f"Model Inference Error: {item.error}")
    if item.result_text is None:
        raise RuntimeError("Inference Timed Out")
        
    return item.result_text

# ================= Routes =================

@app.route("/qwen-text", methods=["POST"])
def qwen_text():
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            return jsonify({"error": "Missing 'prompt'"}), 400

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = enqueue_and_wait(messages)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        
        text = enqueue_and_wait(messages)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/qwen-vl-multi", methods=["POST"])
def qwen_vl_multi():
    try:
        data = request.get_json(force=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        images_b64 = data.get("images", [])
        system = str(data.get("system", "")).strip()

        if not prompt or not images_b64:
            return jsonify({"error": "Missing 'prompt' or 'images' (list of base64)"}), 400

        content = []
        for img_b64 in images_b64:
            image_pil = decode_image_base64(img_b64)
            content.append({"type": "image", "image": image_pil})
        content.append({"type": "text", "text": prompt})

        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        text = enqueue_and_wait(messages)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182, help="Server port")
    parser.add_argument("--model", type=str, default=None, help="Override MODEL_PATH")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path (merged at load time)")
    args = parser.parse_args()

    # Reload model if CLI args override defaults
    if args.model or args.adapter:
        model_path = args.model or MODEL_PATH
        model, processor = load_model(model_path, args.adapter)

    print(f"Starting Qwen3-VL Batched Server on port {args.port} (BatchSize={BATCH_SIZE})...")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
