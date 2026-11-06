import os
import re
import sys
import json
import argparse
import requests

# ----------------------------------------------------------------
# 0. 可调配置
# ----------------------------------------------------------------
DATASET_NAME = os.environ.get("LANGSMITH_DATASET", "langsmith_dataset")  # 你 LangSmith 里的数据集名
BASE_URL = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:5000")
API_URL = os.environ.get("RAG_API_URL", f"{BASE_URL}/api/chat")
RAG_USERNAME = os.environ.get("RAG_USERNAME", "admin")
RAG_PASSWORD = os.environ.get("RAG_PASSWORD", "123456")
TOP_K = 5  # 与前端聊天默认一致

_HERE = os.path.dirname(os.path.abspath(__file__))
# 仓库根目录：脚本位于 evals/ 下，其上一级即项目根；可用环境变量 RAG_HOME 覆盖
_RAG_HOME = os.environ.get("RAG_HOME") or os.path.abspath(os.path.join(_HERE, ".."))
_ENV_CANDIDATES = [
    os.environ.get("RAG_ENV_FILE", ""),
    os.path.join(_RAG_HOME, ".env"),                    # 项目根目录下的 .env
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
    """DeepSeek 的 LLM_API_URL 是 .../v1/chat/completions，OpenAI SDK 只要 base 部分。"""
    raw = os.environ.get("LLM_BASE_URL") or _CFG.get("LLM_API_URL") or "https://api.deepseek.com/v1"
    raw = raw.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    return raw.rstrip("/")


# ----------------------------------------------------------------
# 2. Target：登录拿 cookie -> 调本地 Flask /api/chat
# ----------------------------------------------------------------
_SESSION = None


def _get_session():
    """登录一次，之后复用 cookie（Flask session 是 cookie 鉴权，不是 Bearer token）。"""
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
            print("[登录] 若服务已用 AUTH_ENABLED=false 启动，可忽略本提示")
    except Exception as e:
        print(f"[登录] 异常：{e}")
    _SESSION = s
    return s


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
    """从 langsmith_dataset.csv 读入 {question: expected_answer}，供 target 内评委对齐标准答案。"""
    gold = {}
    csv_path = os.path.join(_HERE, "langsmith_dataset.csv")
    if os.path.isfile(csv_path):
        import csv as _csv
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                q = (row.get("question") or "").strip()
                if q:
                    gold[q] = row.get("expected_answer", "")
    return gold


_GOLD = _load_gold()


# 注：评委 token 现于 target 内与 RAG token 合并写入（见 _write_token_usage），
# 不再在 evaluator 里单独 update_run —— 框架已 finalize 该 run，会触发 409 Duplicate run update。


def target(inputs):
    """inputs 来自 LangSmith 数据集（列 question -> inputs）。返回 prediction 供评委比对。
    评委打分也在本函数内完成，便于把 RAG 生成 + 评委 token 合并写入同一条 experiment run。"""
    question = (inputs.get("question") or inputs.get("input") or "").strip()
    if not question:
        return {"answer": "[数据异常] 该条数据集没有 question 字段", "method": None,
                "judge_score": 0.0, "judge_reason": ""}
    try:
        resp = _get_session().post(
            API_URL,
            json={"question": question, "top_k": TOP_K, "history": []},
            timeout=120,
        )
        if resp.status_code == 401:
            return {"answer": f"[鉴权失败] {resp.text[:120]}", "method": None,
                    "judge_score": 0.0, "judge_reason": ""}
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("success"):
            data = payload["data"]
            rag_usage = data.get("token_usage") or {}
            answer = data.get("answer", "")
            # 在 target 内完成评委打分（用本地 gold 对齐标准答案），并把 RAG+评委 token 合并写回
            reference = _GOLD.get(question, "")
            score, reason, judge_usage = llm_judge(question, answer, reference)
            _write_token_usage(rag_usage, judge_usage)
            return {
                "answer": answer,
                "method": data.get("method"),
                "total_sources": data.get("total_sources"),
                "token_usage": rag_usage,
                "judge_score": score,
                "judge_reason": reason,
            }
        return {"answer": f"[接口错误] {payload.get('error')}", "method": None,
                "judge_score": 0.0, "judge_reason": ""}
    except Exception as e:
        return {"answer": f"[调用异常] {e}", "method": None,
                "judge_score": 0.0, "judge_reason": ""}


# ----------------------------------------------------------------
# 3. 评委：LLM-as-judge（DeepSeek 当裁判，比对 预测 vs 标准答案）
# ----------------------------------------------------------------
def llm_judge(question, prediction, reference):
    """返回 (score 0~1, reasoning, token_usage)。无 key 时退化为关键词命中，仅作 baseline。"""
    _zero = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    key = _llm_key()
    if not key:
        terms = [t for t in re.split(r"[，,、;；\s]+", reference or "") if len(t) >= 2]
        hit = sum(1 for t in terms if t in (prediction or ""))
        score = round(min(1.0, hit / max(1, len(terms))), 2)
        return score, f"[无LLM评委·启发式] 命中 {hit}/{len(terms)} 个关键片段", _zero

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=_llm_base_url())
        prompt = (
            "你是养蜂领域问答评测裁判。根据【问题】和【标准答案】，判断【模型回答】的事实正确性，"
            "给 0~1 分并给简短理由。\n"
            "评分：1=完全正确且完整；0.7=核心事实正确但有遗漏；0.4=部分正确；0=错误或答非所问。\n"
            '只输出 JSON：{"score": x, "reason": "..."}\n\n'
            f"【问题】{question}\n"
            f"【标准答案】{reference}\n"
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
        return float(obj.get("score", 0)), obj.get("reason", text)[:300], usage_dict
    except Exception as e:
        return 0.0, f"[评委异常] {e}", _zero


def _q(inputs):
    if isinstance(inputs, dict):
        return inputs.get("question") or inputs.get("input") or ""
    return ""


def _ref(outputs):
    if isinstance(outputs, dict):
        for k in ("expected_answer", "output", "expected", "answer"):
            if outputs.get(k):
                return outputs[k]
    return ""


def make_evaluators():
    """构造 LangSmith 评委列表。

    注意：langsmith >= 0.2 的 `langsmith.evaluation.evaluator` 是模块而非装饰器，
    这里直接用裸函数即可；官方支持的参数名是
    (run, example, inputs, outputs, reference_outputs)。
    """

    def correctness(run, inputs, outputs, reference_outputs):
        # 分数已在 target 内由评委(DeepSeek)打出，这里直接回传，避免重复调用与 409。
        return {"key": "correctness",
                "score": float((outputs or {}).get("judge_score", 0) or 0),
                "comment": (outputs or {}).get("judge_reason", "")}

    def has_sources(inputs, outputs, reference_outputs):
        """辅助指标：是否确实检索到来源（防'接口炸了但答案看起来正常'）。"""
        n = (outputs or {}).get("total_sources") or 0
        return {"key": "has_sources", "score": 1.0 if n and n > 0 else 0.0,
                "comment": f"检索来源 {n} 条"}

    return [correctness, has_sources]


# ----------------------------------------------------------------
# 4. 本地自检（不连 LangSmith）：验证登录 + 接口 + 评委
# ----------------------------------------------------------------
def dry_run():
    print("=" * 60)
    print("本地自检模式（不联网 LangSmith）")
    print("=" * 60)
    print(f"服务地址 : {API_URL}")
    print(f"登录账号 : {RAG_USERNAME}")
    print(f"评委模型 : {_llm_model()} @ {_llm_base_url()}")
    print(f"评委Key  : {'已读取' if _llm_key() else '未找到（会退化为关键词命中）'}")
    print("-" * 60)

    import csv
    csv_path = os.path.join(_HERE, "langsmith_dataset.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"找不到 {csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    test_rows = rows[:2]
    for i, row in enumerate(test_rows, 1):
        q = row.get("question", "")
        ref = row.get("expected_answer", "")
        print(f"\n[{i}/{len(test_rows)}] Q: {q}")
        pred = target({"question": q})
        ans = pred.get("answer", "")
        print(f"  method={pred.get('method')} sources={pred.get('total_sources')}")
        print(f"  回答前100字: {ans[:100].replace(chr(10), ' ')}")
        if ans.startswith("["):
            print("  ❌ 接口调用失败，先修服务/鉴权再继续")
            continue
        score = pred.get("judge_score", 0)
        reason = pred.get("judge_reason", "")
        print(f"  评分: {score} | 理由: {str(reason)[:120]}")
    print("\n" + "=" * 60)
    print("自检完成。若上面能拿到回答且评分正常，就可以正式跑评测了：")
    print("  python evaluate_rag_langsmith.py")


# ----------------------------------------------------------------
# 5. 正式评测
# ----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="本地自检，不连 LangSmith")
    p.add_argument("--dataset", default=DATASET_NAME, help="LangSmith 数据集名")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 条（省钱/试跑）")
    p.add_argument("--prefix", default="zhi-feng-rag", help="experiment 名前缀（用英文更稳）")
    args = p.parse_args()

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
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")   # 旧版 langsmith 用这个
    os.environ.setdefault("LANGSMITH_TRACING", "true")      # 新版用这个
    os.environ.setdefault("LANGSMITH_PROJECT", "智蜂RAG评测")

    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()
    try:
        ds = client.read_dataset(dataset_name=args.dataset)
        print(f"[数据集] {args.dataset}（id={ds.id}）")
    except Exception as e:
        raise SystemExit(
            f"找不到数据集 '{args.dataset}'。请确认 LangSmith 里的数据集名，"
            f"或用 --dataset 指定。\n原始错误：{e}"
        )

    examples = None
    if args.limit:
        examples = list(client.list_examples(dataset_id=ds.id, limit=args.limit))
        print(f"[试跑] 只取前 {len(examples)} 条")

    results = evaluate(
        target,
        data=examples if examples is not None else args.dataset,
        evaluators=make_evaluators(),
        experiment_prefix=args.prefix,
        num_repetitions=1,
        max_concurrency=2,   # 别开太大，本地 Flask + LLM 容易被限流
    )

    # 汇总打印平均分
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
