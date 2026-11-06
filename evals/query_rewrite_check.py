# -*- coding: utf-8 -*-
"""查询重写功能验证：多轮指代消解是否生效"""
import sys, time, os

# 脚本位于 evals/ 下，上一级即项目根
PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PROJECT)

# 复现 app.py 的 .env 加载逻辑（脚本独立运行时不会自动加载）
env_path = os.path.join(PROJECT, '.env')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

from rag_engine import RAGEngine

print("初始化 RAG 引擎（加载向量库，约几秒）...")
t0 = time.time()
engine = RAGEngine()
engine.initialize()
print(f"初始化完成，耗时 {time.time() - t0:.1f}s\n")

# 模拟真实多轮对话：第一轮问陕西，第二轮用"那里"指代
HISTORY = [
    {"role": "user", "content": "陕西哪里采蜜比较好？"},
    {"role": "assistant", "content": "陕西主要蜜源有汉中的油菜（3-4月）、延安的刺槐（5月）、秦岭的荆条（6-8月）。"},
]

TESTS = [
    ("多轮指代追问", "那里产蜜量高吗？", HISTORY),
    ("多轮省略主语", "花期是什么时候？", HISTORY),
    ("首轮完整提问", "广东三月有什么蜜源？", None),
]

print("=" * 74)
for name, q, hist in TESTS:
    sources, used = engine.retrieve(q, top_k=3, history=hist)
    flag = "已重写" if used != q else "未重写"
    print(f"\n【{name}】{flag}")
    print(f"  原始问题：{q}")
    print(f"  实际检索：{used}")
    top = sources[0] if sources else None
    if top:
        label = top.get('plant_name') or top.get('title')
        loc = f"{top.get('province', '')}{top.get('city', '')}"
        print(f"  Top1 命中：{label}　{loc}　score={top.get('score', 0):.4f}")
    else:
        print("  Top1 命中：无")
print("\n" + "=" * 74)
