#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智蜂专家 RAG 系统 —— 多轮对话（查询重写）LangSmith SDK 评测脚本
===============================================================
本脚本专门验证「查询重写 / 指代消解」功能：
    给定一段对话历史 + 一条依赖上下文的追问（"那里产蜜量高吗？"），
    系统是否能在召回层把指代补全为可独立检索的查询，并给出正确回答。

与单轮脚本的区别：
    * 单轮脚本传 history=[]，查询重写不触发，测"基础检索+生成"；
    * 本脚本传真实 history，强制触发 rewrite_query，测"指代消解是否正确"。

为什么用 SDK 而不是 Playground：同单轮——你的 /api/chat 是独立 Flask 服务，
Playground 只能调试 LangChain chain，挂不上。

用法（三条命令）：
    :: 0. 先在环境变量或 rag_system/.env 中配置 LANGSMITH_API_KEY（脚本不内置明文 key）
    :: 1. 先本地自检（不联网 LangSmith，验证接口+评委都通）
    python evaluate_rag_multiturn_langsmith.py --dry-run
    :: 2. 可选：导出 CSV 数据集（方便在 LangSmith 网页手动上传查看）
    python evaluate_rag_multiturn_langsmith.py --export-csv langsmith_multiturn_dataset.csv
    :: 3. 正式跑评测（自动把数据集推到 LangSmith 并跑 experiment）
    python evaluate_rag_multiturn_langsmith.py
"""

import os
import re
import json
import argparse
import requests

# ----------------------------------------------------------------
# 0. 可调配置（与单轮脚本保持一致）
# ----------------------------------------------------------------
BASE_URL = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:5000")
API_URL = os.environ.get("RAG_API_URL", f"{BASE_URL}/api/chat")
RAG_USERNAME = os.environ.get("RAG_USERNAME", "admin")
RAG_PASSWORD = os.environ.get("RAG_PASSWORD", "123456")
TOP_K = 5
DATASET_NAME = os.environ.get("LANGSMITH_DATASET", "langsmith_multiturn_dataset")

_HERE = os.path.dirname(os.path.abspath(__file__))
# 仓库根目录：脚本位于 evals/ 下，其上一级即项目根；可用环境变量 RAG_HOME 覆盖
_RAG_HOME = os.environ.get("RAG_HOME") or os.path.abspath(os.path.join(_HERE, ".."))
_ENV_CANDIDATES = [
    os.environ.get("RAG_ENV_FILE", ""),
    os.path.join(_RAG_HOME, ".env"),
    os.path.join(_HERE, ".env"),
]


# ----------------------------------------------------------------
# 1. 读取 rag_system/.env，复用 LLM 配置当评委
# ----------------------------------------------------------------
def _read_env_file():
    for p in _ENV_CANDIDATES:
        if os.path.isfile(p):
            cfg = {}
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    cfg[k] = v
                    # 同时注入 os.environ，使 LANGSMITH_API_KEY / LLM_API_KEY 等
                    # 均可被本脚本直接读取（注意：.env 永不提交到公开仓库）
                    os.environ.setdefault(k, v)
            return cfg
    return {}


_CFG = _read_env_file()


def _llm_key():
    return os.environ.get("LLM_API_KEY") or _CFG.get("LLM_API_KEY")


def _llm_model():
    return os.environ.get("LLM_MODEL") or _CFG.get("LLM_MODEL") or "deepseek-chat"


def _llm_base_url():
    raw = os.environ.get("LLM_BASE_URL") or _CFG.get("LLM_API_URL") or "https://api.deepseek.com/v1"
    raw = raw.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    return raw.rstrip("/")


# ----------------------------------------------------------------
# 2. 多轮评测真源数据（同时用于 --export-csv 与 LangSmith 建数据集）
#    history: 前一轮对话（[{role, content}]），question: 依赖上下文的追问
#    expected: 指代应消解到的实体 + 关键事实要求（评委按此核对"是否消解正确"）
# ----------------------------------------------------------------
MULTITURN_EXAMPLES = [
    {
        "category": "指代消解-那里",
        "history": [
            {"role": "user", "content": "荆条蜜源主要分布在哪些省份？"},
            {"role": "assistant", "content": "荆条是北方分布极广的优良蜜源，主要分布于河北、山西、陕西、河南、山东、辽宁等省份，花期6-8月。"},
        ],
        "question": "那里产蜜量高吗？",
        "expected": "必须识别'那里'指代【荆条】，并围绕荆条的产蜜量作答（说明其单群产蜜量水平或蜜质），不能跑题到其他植物或泛泛而谈。",
    },
    {
        "category": "指代消解-它",
        "history": [
            {"role": "user", "content": "荔枝蜜源的开花期是什么时候？"},
            {"role": "assistant", "content": "荔枝花期一般在2-4月，因地域而异，广东从化等地1-3月即可开花，是南方春季重要蜜源。"},
        ],
        "question": "它的采蜜期是几月？",
        "expected": "必须识别'它'指代【荔枝】，回答荔枝的采蜜期（一般3-4月），而不是泛指或答成别的植物。",
    },
    {
        "category": "话题切换-那X呢",
        "history": [
            {"role": "user", "content": "广东3月有哪些蜜源植物？"},
            {"role": "assistant", "content": "广东3月主要有荔枝、油菜、紫云英等蜜源植物，气候温暖花期早。"},
        ],
        "question": "那陕西呢？",
        "expected": "必须切换到【陕西省】，列出陕西3月左右的蜜源（如刺槐、油菜、枣树等），不应重复广东的内容。",
    },
    {
        "category": "指代消解-这个",
        "history": [
            {"role": "user", "content": "刺槐的花蜜含糖量是多少？"},
            {"role": "assistant", "content": "刺槐花蜜含糖量较高，属优质蜜源，蜜色清浅气味芳香。"},
        ],
        "question": "这个蜜源适合养什么蜂？",
        "expected": "必须识别'这个'指代【刺槐】，回答其适合的蜂种（中蜂/意蜂），不应偏离到别的蜜源。",
    },
    {
        "category": "复数指代-它们",
        "history": [
            {"role": "user", "content": "冬季有哪些辅助蜜源？"},
            {"role": "assistant", "content": "冬季辅助蜜源主要有柃木（野桂花）、枇杷等，可填补冬季蜜源空缺，对南方养蜂尤为重要。"},
        ],
        "question": "它们的分布省份是哪些？",
        "expected": "必须识别'它们'指代【冬季辅助蜜源（柃木/枇杷等）】，回答其分布省份，不能跑题。",
    },
    {
        "category": "对比延续-它们",
        "history": [
            {"role": "user", "content": "荆条和枣树，哪个产蜜量更高？"},
            {"role": "assistant", "content": "荆条与枣树均为优质蜜源，荆条单产通常更高、流蜜量大，枣树蜜质浓稠色泽深。"},
        ],
        "question": "那它们的花期重叠吗？",
        "expected": "必须识别'它们'指代【荆条和枣树】，比较两者花期是否重叠（荆条6-8月、枣树5-6月，部分重叠），需基于两实体作答。",
    },
    {
        "category": "指代消解-上面说的",
        "history": [
            {"role": "user", "content": "怎么防治蜂螨？"},
            {"role": "assistant", "content": "防治蜂螨可在断子期使用甲酸、草酸等药剂喷施，注意浓度与安全的间隔期。"},
        ],
        "question": "上面说的那个药怎么用？",
        "expected": "必须识别'上面说的药'指代【蜂螨防治药剂（甲酸/草酸等）】，回答其使用方法，不应偏离到别的病害。",
    },
    {
        "category": "多跳指代-它",
        "history": [
            {"role": "user", "content": "陕西汉中适合种什么蜜源？"},
            {"role": "assistant", "content": "陕西汉中气候湿润、山地多，适宜栽培刺槐、油菜等蜜源植物。"},
        ],
        "question": "那它的最佳采蜜月份是？",
        "expected": "必须识别'它'指代【陕西汉中的主要蜜源（刺槐）】，回答刺槐的最佳采蜜月份（一般4-5月）。",
    },
]


# ----------------------------------------------------------------
# 3. Target：登录拿 cookie -> 带 history 调 /api/chat（触发查询重写）
# ----------------------------------------------------------------
_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    s = requests.Session()
    try:
        r = s.post(f"{BASE_URL}/api/auth/login",
                   json={"username": RAG_USERNAME, "password": RAG_PASSWORD},
                   timeout=15)
        if r.status_code == 200 and r.json().get("success"):
            print(f"[登录] {RAG_USERNAME} 成功")
        else:
            print(f"[登录] 失败（返回 {r.status_code}）：{r.text[:160]}")
    except Exception as e:
        print(f"[登录] 异常：{e}")
    _SESSION = s
    return s


def _normalize_history(history):
    """history 可能是 list（程序化建数据集）或 JSON 字符串（CSV 上传）。"""
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except Exception:
            history = []
    if not isinstance(history, list):
        history = []
    # 只保留合法结构（与 app.py 解析一致）
    return [h for h in history
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content")]


def _write_token_usage(rag_usage, judge_usage):
    """在 target 执行上下文内调用（get_current_run_tree 指向当前 experiment run）：
    把【RAG 生成 + 评委打分】的 token 合并写入该 run 的 usage_metadata。

    必须在 target 内部调用——框架收尾 run 时会自动带上 extra 落库（RAG token 已验证此路可行）。
    绝不要在 evaluator 里用 Client().update_run 补写：run 已被框架 finalize，会报 409。
    """
    try:
        from langsmith import get_current_run_tree
        rt = get_current_run_tree()
        if rt is None:
            return
        z = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for u in (rag_usage or {}, judge_usage or {}):
            z["input_tokens"] += int(u.get("input_tokens", 0) or 0)
            z["output_tokens"] += int(u.get("output_tokens", 0) or 0)
            z["total_tokens"] += int(u.get("total_tokens", 0) or 0)
        rt.extra.setdefault("metadata", {})["usage_metadata"] = z
    except Exception:
        pass


def _load_gold():
    """从内嵌 MULTITURN_EXAMPLES 构建 {question: expected}，供 target 内评委对齐标准答案。"""
    gold = {}
    for ex in MULTITURN_EXAMPLES:
        q = (ex.get("question") or "").strip()
        if q:
            gold[q] = ex.get("expected", "")
    return gold


_GOLD = _load_gold()


def target(inputs):
    question = (inputs.get("question") or inputs.get("input") or "").strip()
    history = _normalize_history(inputs.get("history") or inputs.get("history_json"))
    if not question:
        return {"answer": "[数据异常] 缺少 question", "resolved_entity": None,
                "judge_score": 0.0, "judge_reason": ""}
    try:
        resp = _get_session().post(
            API_URL,
            json={"question": question, "top_k": TOP_K, "history": history},
            timeout=120,
        )
        if resp.status_code == 401:
            return {"answer": f"[鉴权失败] {resp.text[:120]}", "resolved_entity": None,
                    "judge_score": 0.0, "judge_reason": ""}
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("success"):
            data = payload["data"]
            rag_usage = data.get("token_usage") or {}
            answer = data.get("answer", "")
            hist_summary = " | ".join(f"{h['role']}:{h['content']}" for h in history)
            expected = _GOLD.get(question, "")
            # 在 target 内完成评委打分（指代消解），并把 RAG+评委 token 合并写回
            score, reason, judge_usage = multiturn_judge(question, hist_summary, answer, expected)
            _write_token_usage(rag_usage, judge_usage)
            return {
                "answer": answer,
                "method": data.get("method"),
                "total_sources": data.get("total_sources"),
                "token_usage": rag_usage,
                "resolved_entity": None,  # 非流式接口未回传 rewritten_query，由评委从答案推断
                "judge_score": score,
                "judge_reason": reason,
            }
        return {"answer": f"[接口错误] {payload.get('error')}", "resolved_entity": None,
                "judge_score": 0.0, "judge_reason": ""}
    except Exception as e:
        return {"answer": f"[调用异常] {e}", "resolved_entity": None,
                "judge_score": 0.0, "judge_reason": ""}


# ----------------------------------------------------------------
# 4. 评委：多轮专用的"指代消解正确性"裁判
# ----------------------------------------------------------------
def multiturn_judge(follow_up, history_summary, prediction, expected):
    """检查：系统是否把追问里的指代正确消解到正确实体，并给出相关回答。
    返回 (score, reason, token_usage)。"""
    _zero = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    key = _llm_key()
    if not key:
        # 退化 baseline：看答案是否明显偏离（极简）
        return 0.5, "[无LLM评委·启发式] 仅确认有回答，无法严格判定指代消解", _zero

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=_llm_base_url())
        prompt = (
            "你是养蜂问答系统的多轮评测裁判。用户的【追问】依赖对话上下文（含指代'那里/它/它们/这个/那X呢'等），"
            "系统需要先做查询重写把指代补全，再检索作答。\n"
            "请判断【模型回答】是否正确消解了指代（即回答是否指向了正确的实体/话题），并给出相关事实。\n"
            "评分：1=指代完全消解正确且事实准确；0.7=实体对、细节略缺；0.4=实体部分正确或含糊；"
            "0=实体错（跑题/答非所问/把指代弄错）。\n"
            '只输出 JSON：{"score": x, "resolved_entity": "回答实际指向的实体", "reason": "..."}\n\n'
            f"【前文摘要】{history_summary}\n"
            f"【追问】{follow_up}\n"
            f"【应消解到的实体与要求】{expected}\n"
            f"【模型回答】{prediction}\n"
        )
        r = client.chat.completions.create(
            model=_llm_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        usage = r.usage
        usage_dict = {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        text = (r.choices[0].message.content or "").strip()
        s, e = text.find("{"), text.rfind("}") + 1
        obj = json.loads(text[s:e]) if s != -1 and e > s else {}
        ent = obj.get("resolved_entity", "")
        return float(obj.get("score", 0)), f"[{ent}] {obj.get('reason', text)[:260]}", usage_dict
    except Exception as e:
        return 0.0, f"[评委异常] {e}", _zero


def make_evaluators():
    def query_rewrite_correctness(run, inputs, outputs, reference_outputs):
        # 分数已在 target 内由评委(DeepSeek)打出，这里直接回传，避免重复调用与 409。
        return {"key": "rewrite_correctness",
                "score": float((outputs or {}).get("judge_score", 0) or 0),
                "comment": (outputs or {}).get("judge_reason", "")}

    def has_sources(inputs, outputs, reference_outputs):
        n = (outputs or {}).get("total_sources") or 0
        return {"key": "has_sources", "score": 1.0 if n and n > 0 else 0.0,
                "comment": f"检索来源 {n} 条"}

    return [query_rewrite_correctness, has_sources]


# ----------------------------------------------------------------
# 5. 导出 CSV（供 LangSmith 网页手动上传）
# ----------------------------------------------------------------
def export_csv(path):
    import csv
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "history_json", "expected_answer", "category"])
        for ex in MULTITURN_EXAMPLES:
            w.writerow([
                ex["question"],
                json.dumps(ex["history"], ensure_ascii=False),
                ex["expected"],
                ex["category"],
            ])
    print(f"已导出 {len(MULTITURN_EXAMPLES)} 条到 {path}")
    print("LangSmith 网页上传时列映射：")
    print("  question      -> Input")
    print("  history_json  -> Input（附加字段，无需改动）")
    print("  expected_answer -> Expected Output（Reference）")
    print("  category      -> 可忽略或作附加 Input")


# ----------------------------------------------------------------
# 6. 本地自检
# ----------------------------------------------------------------
def dry_run():
    print("=" * 60)
    print("多轮（查询重写）本地自检模式")
    print("=" * 60)
    print(f"服务地址 : {API_URL}")
    print(f"评委模型 : {_llm_model()} @ {_llm_base_url()}")
    print(f"评委Key  : {'已读取' if _llm_key() else '未找到（退化为0.5）'}")
    print(f"用例数   : {len(MULTITURN_EXAMPLES)}")
    print("-" * 60)

    for i, ex in enumerate(MULTITURN_EXAMPLES, 1):
        print(f"\n[{i}/{len(MULTITURN_EXAMPLES)}] 类别: {ex['category']}")
        print(f"  前文: {ex['history'][0]['content']}")
        print(f"  追问: {ex['question']}")
        pred = target({"question": ex["question"], "history": ex["history"]})
        ans = pred.get("answer", "")
        print(f"  method={pred.get('method')} sources={pred.get('total_sources')}")
        print(f"  回答前120字: {ans[:120].replace(chr(10), ' ')}")
        if ans.startswith("["):
            print("  ❌ 接口调用失败，先修服务/鉴权再继续")
            continue
        score, reason = multiturn_judge(
            ex["question"],
            " | ".join(f"{h['role']}:{h['content']}" for h in ex["history"]),
            ans, ex["expected"])
        print(f"  指代消解评分: {score} | {str(reason)[:140]}")
    print("\n" + "=" * 60)
    print("自检完成。若回答能正确指向前文实体，即可正式跑评测。")


# ----------------------------------------------------------------
# 7. 正式评测（自动把数据集推到 LangSmith）
# ----------------------------------------------------------------
def build_langsmith_examples(client, dataset_name):
    """在 LangSmith 创建/复用数据集，并写入多轮用例。返回 dataset id。"""
    try:
        ds = client.create_dataset(
            dataset_name=dataset_name,
            description="智蜂RAG多轮对话（查询重写/指代消解）评测集",
        )
    except Exception:
        ds = client.read_dataset(dataset_name=dataset_name)
        # 清空旧用例，避免重复
        for ex in client.list_examples(dataset_id=ds.id):
            try:
                client.delete_example(ex.id)
            except Exception:
                pass
    for ex in MULTITURN_EXAMPLES:
        client.create_example(
            inputs={"question": ex["question"], "history": ex["history"]},
            outputs={"expected_answer": ex["expected"], "category": ex["category"]},
            dataset_id=ds.id,
        )
    print(f"[数据集] 已写入 {len(MULTITURN_EXAMPLES)} 条到 '{dataset_name}' (id={ds.id})")
    return ds.id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export-csv", metavar="PATH", help="导出多轮 CSV 数据集")
    p.add_argument("--dataset", default=DATASET_NAME)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--prefix", default="zhi-feng-rag-multiturn")
    args = p.parse_args()

    if args.export_csv:
        export_csv(args.export_csv)
        return
    if args.dry_run:
        dry_run()
        return

    # ---- LangSmith 凭证：从环境变量或 .env 读取，脚本内绝不内置明文 key ----
    # .env 中的 LANGSMITH_API_KEY 已由上方 _read_env_file 注入 os.environ。
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit(
            "缺少 LANGSMITH_API_KEY：请在本机环境变量中设置，或写入 rag_system/.env"
            "（.env 已自动加载，但绝不提交到仓库）。\n"
            "获取地址：LangSmith 左下角 Settings -> API Keys"
        )
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "智蜂RAG评测-多轮")

    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()
    ds_id = build_langsmith_examples(client, args.dataset)

    examples = None
    if args.limit:
        examples = list(client.list_examples(dataset_id=ds_id, limit=args.limit))
        print(f"[试跑] 只取前 {len(examples)} 条")

    results = evaluate(
        target,
        data=examples if examples is not None else ds_id,
        evaluators=make_evaluators(),
        experiment_prefix=args.prefix,
        num_repetitions=1,
        max_concurrency=2,
    )

    try:
        scores = [r["evaluation_results"]["results"] for r in results]
        flat = {}
        for evs in scores:
            for ev in evs:
                flat.setdefault(ev.key, []).append(ev.score)
        print("\n=== 汇总 ===")
        for k, v in flat.items():
            vals = [x for x in v if x is not None]
            if vals:
                print(f"  {k}: 平均 {sum(vals)/len(vals):.3f}（{len(vals)} 条）")
    except Exception:
        pass

    print("\n✅ 完成。到 LangSmith -> Datasets & Experiments -> 你的数据集 -> Experiments 查看逐条对比。")


if __name__ == "__main__":
    main()
