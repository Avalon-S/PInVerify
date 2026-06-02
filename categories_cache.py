# ====== 1) 配置 ======
DESC_DB_JSON = "./data/pv_dataset/val/object_descriptions_with_category.json"
CACHE_JSON   = "./data/pv_dataset/category_cache_7B.json"      # 生成/增量写入的缓存文件
QWEN_TEXT_URL = "http://127.0.0.1:12182/qwen-text"  # 你的 Qwen 文本接口地址

# ====== 2) 固定粗类别与工具函数 ======
import os, json, re, time, requests
from collections import Counter

COARSE_CATS = [
    "backpack","bag","ball","book","camera","cellphone",
    "eyeglasses","hat","headphones","keys","laptop","mug",
    "shoes","teddy bear","toy","visor","wallet","watch",
]

def norm(s: str) -> str:
    """统一规整：去引号/空白、小写、把下划线转空格、压缩多空格"""
    s = (s or "").strip().strip('"').strip("'").lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def to_coarse_or_empty(label: str) -> str:
    """若能映射到 18 类之一返回该类，否则返回空串。"""
    s = norm(label)
    return s if s in COARSE_CATS else ""

# ====== 3) I/O ======
def load_desc_db(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # 期望结构：{object_id: {"object_category": "...", "descriptions":[...]}, ...}
    db = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                db[str(k)] = v
    return db

def load_cache(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def save_cache(cache: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ====== 4) Qwen 文本分类提示词 & HTTP ======
def build_category_choice_prompt(desc1: str, desc2: str, desc3: str) -> str:
    classes_str = ", ".join([f'"{c}"' for c in COARSE_CATS])
    return f"""
You will classify the OBJECT CATEGORY described by three sentences into EXACTLY ONE of the following classes:
[{classes_str}]

Descriptions:
1) {desc1}
2) {desc2}
3) {desc3}

RULES:
- Output only the chosen class name, exactly as listed (case-insensitive allowed).
- Do NOT output any extra words, explanations, punctuation, or quotes.
- If multiple possible, choose the most specific one among the list.

Answer with the single class name only.
""".strip()

CONNECT_TIMEOUT, READ_TIMEOUT = 5, 45
def call_qwen_text(url: str, prompt: str):
    try:
        r = requests.post(url, json={"prompt": prompt}, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if r.status_code != 200:
            return {"text": "", "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        return r.json()
    except Exception as e:
        return {"text": "", "error": str(e)}

# ====== 5) 生成/更新缓存（断点续跑） ======
desc_db = load_desc_db(DESC_DB_JSON)
cache   = load_cache(CACHE_JSON)

all_ids = list(desc_db.keys())
todo_ids = [oid for oid in all_ids if oid not in cache]

print(f"[CACHE] objects total: {len(all_ids)}, cached: {len(cache)}, to-run: {len(todo_ids)}")

for i, oid in enumerate(todo_ids, 1):
    item = desc_db[oid]
    descs = [d for d in (item.get("descriptions") or []) if isinstance(d, str) and d.strip()]
    while len(descs) < 3:
        descs.append("<unspecified>")
    prompt = build_category_choice_prompt(descs[0], descs[1], descs[2])
    resp = call_qwen_text(QWEN_TEXT_URL, prompt)
    pred_raw = (resp.get("text") or "").strip()
    pred = norm(pred_raw)  # 统一规整
    # 安全起见：只接受 18 类之一；否则置空
    pred = pred if pred in COARSE_CATS else ""
    cache[oid] = {
        "pred_coarse": pred,
        "raw": pred_raw,
    }

    # 每 N 条持久化一次，避免中断丢失
    if i % 20 == 0 or i == len(todo_ids):
        save_cache(cache, CACHE_JSON)
        print(f"  cached {len(cache)}/{len(all_ids)} ...")

print(f"[DONE] cache saved to: {CACHE_JSON}")

# ====== 6) 评估准确率（与 GT 同源文件对比） ======
used = 0
correct = 0
per_class_total = Counter()
per_class_correct = Counter()

for oid, item in desc_db.items():
    gt_raw = item.get("object_category", "") or item.get("category", "")
    gt = to_coarse_or_empty(gt_raw)
    if not gt:
        # 该对象 GT 不在 18 粗类内，跳过（极少见；可按需提示）
        continue

    pred = cache.get(oid, {}).get("pred_coarse", "")
    if not pred:
        # 没有缓存到预测；可提示继续跑缓存
        continue

    used += 1
    per_class_total[gt] += 1
    if pred == gt:
        correct += 1
        per_class_correct[gt] += 1

if used == 0:
    print("No valid pairs to evaluate. Check cache or GT mapping.")
else:
    acc = correct / used
    print(f"\n=== Coarse Category Accuracy ===")
    print(f"Used objects: {used} / {len(desc_db)}")
    print(f"Overall accuracy: {acc:.4f}  ({correct}/{used})\n")

    # 每类
    print("Per-class accuracy:")
    rows = [(c, per_class_total[c], (per_class_correct[c]/per_class_total[c]) if per_class_total[c] else 0.0)
            for c in COARSE_CATS if per_class_total[c] > 0]
    rows.sort(key=lambda x: (-x[1], x[0]))
    for c, n, a in rows:
        print(f"  {c:12s}  n={n:4d}  acc={a:.3f}")

# ====== 7) 运行时查表（示例） ======
def load_coarse_cache(path: str = CACHE_JSON):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def lookup_category(object_id: str, cache_dict: dict):
    """运行时根据 object_id 取粗类；不存在则返回空串"""
    return cache_dict.get(str(object_id), {}).get("pred_coarse", "")

# 示例：
runtime_cache = load_coarse_cache()
print("example:", lookup_category("f1ec3599a75941b1ba52d8bee4f32ab0", runtime_cache))
