#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
from typing import Any, Dict, List

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def summarize_classification(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    results: [{"pred":0/1, "label":0/1, "pair_type":"positive|neg_same|neg_diff", ...}, ...]
    """
    n = len(results)
    tp = sum(1 for r in results if r["pred"]==1 and r["label"]==1)
    tn = sum(1 for r in results if r["pred"]==0 and r["label"]==0)
    fp = sum(1 for r in results if r["pred"]==1 and r["label"]==0)
    fn = sum(1 for r in results if r["pred"]==0 and r["label"]==1)
    acc = (tp+tn)/n if n else 0.0

    by_type = {}
    for r in results:
        t = r.get("pair_type","unknown")
        by_type.setdefault(t, {"n":0,"acc":0.0,"tp":0,"tn":0,"fp":0,"fn":0})
        by_type[t]["n"]  += 1
        if r["pred"]==r["label"]:
            by_type[t]["acc"] += 1
        if r["pred"]==1 and r["label"]==1: by_type[t]["tp"]+=1
        if r["pred"]==0 and r["label"]==0: by_type[t]["tn"]+=1
        if r["pred"]==1 and r["label"]==0: by_type[t]["fp"]+=1
        if r["pred"]==0 and r["label"]==1: by_type[t]["fn"]+=1
    for t,v in by_type.items():
        v["acc"] = v["acc"]/v["n"] if v["n"] else 0.0

    return {
        "num_pairs": n,
        "accuracy": acc,
        "confusion": {"tp":tp,"tn":tn,"fp":fp,"fn":fn},
        "by_type": by_type
    }

def print_cls_summary(summary: Dict[str, Any]):
    print("\n================= Classification Summary =================")
    print(f"Pairs evaluated       : {summary['num_pairs']}")
    print(f"Accuracy              : {summary['accuracy']:.4f}")
    c = summary["confusion"]
    print(f"Confusion (tp/tn/fp/fn): {c['tp']}/{c['tn']}/{c['fp']}/{c['fn']}")
    print("By pair_type:")
    for k,v in summary["by_type"].items():
        print(f"  - {k:10s}: n={v['n']:4d} | acc={v['acc']:.4f} | tp={v['tp']} tn={v['tn']} fp={v['fp']} fn={v['fn']}")
    print("==========================================================\n")
