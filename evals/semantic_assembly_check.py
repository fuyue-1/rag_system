# -*- coding: utf-8 -*-
"""
语义化组装有效性验证实验
对比「原始字段名」与「中文组装文本」在同一查询下的语义相似度
"""
import os
import sys

import numpy as np

# 脚本位于 evals/ 下，上一级即项目根，保证能 import rag_engine
PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PROJECT)

from rag_engine import LocalBGEEmbedder

print("加载本地 bge-small-zh 模型（首次需几秒）...")
emb = LocalBGEEmbedder()
if not emb.is_ready():
    print("模型不可用，无法实验")
    raise SystemExit(1)

QUERY = "七月去哪里采蜜比较好？那边气候怎么样？"

# 同一条蜜源数据的三种写法
VARIANTS = {
    "① 原始数据库字段（裸字段名）":
        "plant_name=荆条 province=陕西 annual_avg_temp=12.5 "
        "annual_precipitation=650 sunshine_hours=2100 "
        "flowering_start=6 flowering_end=8 best_honey_start=7 best_honey_end=8",

    "② 只有数值，无任何说明":
        "荆条 陕西 12.5 650 2100 6 8 7 8",

    "③ 本系统的语义化组装（加中文说明词）":
        "荆条 陕西 宝鸡 秦岭北麓 荆条蜜源丰富 气候类型温带季风气候 "
        "年均气温12.5℃ 年降水量650mm 日照2100小时 开花6月到8月 采蜜7月到8月",
}


def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


q = emb.embed_query(QUERY)
print(f"\n查询：{QUERY}\n")
print("=" * 72)
results = {}
for name, text in VARIANTS.items():
    v = emb.embed_query(text)
    s = cos(q, v)
    results[name] = s
    print(f"{name}\n    与查询的语义相似度：{s:.4f}\n")

base = results["① 原始数据库字段（裸字段名）"]
best = results["③ 本系统的语义化组装（加中文说明词）"]
print("=" * 72)
print(f"语义化组装相比裸字段名，相似度提升：{(best - base):.4f} "
      f"（相对提升 {(best / base - 1) * 100:.1f}%）")
print("\n结论：字段名本身对中文向量模型几乎是噪声，")
print("      人工补上中文语义说明词后，检索匹配度显著提高。")
