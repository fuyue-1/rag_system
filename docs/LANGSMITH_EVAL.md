# 智蜂专家 RAG 系统 —— LangSmith 评测完整流程

> 适用现状：数据集 `langsmith_dataset` 已上传（10 条单轮问答），LangSmith 里已配了一个
> `Correctness`（LLM-as-a-judge）Evaluator。下面是从零到拿到分数的全流程。

---

## 0. 结论先行：用 SDK，不要用 Playground

你的截图里 `+ Experiment` 有三个选项：

| 选项 | 能不能用 | 原因 |
|---|---|---|
| Run in Playground | ❌ | 只能调试 LangChain chain / LangSmith 里的 prompt，挂不上你的独立 Flask 服务 |
| Run in LangSmith Studio | ❌ | 同上，要求代码在 LangSmith 托管环境里 |
| **Run in SDK** | ✅ | 写脚本逐条调你的 `/api/chat`，结果回传记成 experiment |

点 "Run in SDK" 弹出来的代码片段本质就是我们要做的事，但它**不会**帮你处理登录鉴权和评委模型，
所以直接用下面已经封装好的脚本更省事。

---

## 1. 前置检查（3 件）

### ① 主系统在跑

```bash
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/
# 期望 200
```

没跑就先双击 `E:\rag_system\启动系统.bat`。

### ② 登录账号可用

系统默认 `AUTH_ENABLED=true`，`/api/chat` 需要登录（Flask session cookie，不是 Bearer token）。
现成可用的账号是 **admin / 123456**。

> 脚本会自动调 `/api/auth/login` 拿 cookie，**所以不需要重启服务、不需要关鉴权**。
> 如果换了密码，用环境变量传：`set RAG_USERNAME=xxx` 和 `set RAG_PASSWORD=xxx`。

### ③ 装依赖（已装好，路径如下）

```bash
python
```

已装版本（实测可用）：`langsmith 0.14.12` / `openai 3.36.1` / `requests 2.34.2`。
如果重装，**必须用阿里云镜像**（清华源会 403 拦 pip 的 UA）：

```bash
python -m pip install langsmith openai requests -i https://mirrors.aliyun.com/pypi/simple
```

---

## 2. 本地自检（强烈建议先跑这一步）

```bash
python E:\rag_system\evaluate_rag_langsmith.py --dry-run
```

这一步**不联网 LangSmith**，只做三件事：
1. 登录你的 Flask 服务；
2. 拿 CSV 里前 2 个问题真实调一次 `/api/chat`；
3. 用 DeepSeek 当评委打一次分。

看到类似输出就说明链路全通（下面是本机实测的真实输出节选）：

```
服务地址 : http://127.0.0.1:5000/api/chat
登录账号 : admin
评委模型 : deepseek-chat @ https://api.deepseek.com/v1
评委Key  : 已读取

[1/2] Q: 广东3月有什么蜜源植物？
[登录] admin 成功
  method=llm sources=4
  回答前100字: 根据检索资料，广东3月主要有以下蜜源植物：**1. 荔枝（首选推荐）** …
  评分: 1.0 | 理由: 模型回答准确指出广东3月主要蜜源为荔枝，并涵盖从化、高州等核心产区…

[2/2] Q: 荆条蜜源主要分布在哪些省份？
  method=llm sources=5
  评分: 1.0 | 理由: 完整列出标准答案中的五个省份…
```

**如果这一步就报错，先别继续**，按第 5 节 FAQ 排查。

---

## 3. 正式跑评测

### ① 设环境变量（PowerShell 版）

```powershell
$env:LANGSMITH_API_KEY="lsv2_pt_你的key"     # LangSmith 左下角 Settings -> API Keys
$env:LANGSMITH_PROJECT="智蜂RAG评测"
$env:LANGSMITH_DATASET="langsmith_dataset"    # 你截图里的数据集名，默认已是这个
```

cmd 版用 `set LANGSMITH_API_KEY=...`。

### ② 跑

```bash
python E:\rag_system\evaluate_rag_langsmith.py
```

### ③ 想先小成本试跑（只跑前 3 条）

```bash
...\python.exe evaluate_rag_langsmith.py --limit 3
```

跑完终端会打印汇总：

```
=== 汇总 ===
  correctness: 平均 0.820（10 条）
  has_sources: 平均 1.000（10 条）
```

---

## 4. 到哪看结果

1. LangSmith → **Datasets & Experiments** → 点 `langsmith_dataset`
2. 切到 **Experiments** 标签 → 看到 `zhi-feng-rag-xxxx` 这次实验
3. 点进去：每行一个问题，列有 `correctness`（0~1）、`has_sources`（有没有真的检索到来源）
4. 点单行可以对比「模型回答 vs 标准答案 vs 评委理由」

---

## 5. 常见问题 FAQ

| 现象 | 原因 | 解决 |
|---|---|---|
| `[鉴权失败]` / `AUTH_REQUIRED` | 账号密码不对 | `set RAG_PASSWORD=你的密码`，或确认 admin/123456 能登 |
| `[调用异常] Connection refused` | Flask 没起 | 双击 `启动系统.bat`，确认 5000 端口 |
| `找不到数据集 'langsmith_dataset'` | 数据集名不一致 | 打开 LangSmith 看实际名字，用 `--dataset 实际名字` |
| 分数全是 0 | 评委 key 没读到 | 确认 `E:\rag_system\.env` 里有 `LLM_API_KEY`（脚本自动读） |
| 报 429 限流 | DeepSeek 并发太高 | 已默认 `max_concurrency=2`，还报就改成 1 |
| pip 报 `403 Forbidden` | 清华源拦 pip 的 UA | 换 `-i https://mirrors.aliyun.com/pypi/simple` |
| `TypeError: 'module' object is not callable` | 老教程写 `@evaluator` 装饰器，新版里 `langsmith.evaluation.evaluator` 是模块 | 直接用裸函数 `def f(inputs, outputs, reference_outputs)`，不要加装饰器 |
| LangSmith 网页上 `Correctness` 一直没分 | 那个 UI 级 Evaluator 只支持 OpenAI/Anthropic/Gemini/Bedrock，**不支持 DeepSeek** | 见下方说明 |

### 关于你截图里那个 UI 级 `Correctness` Evaluator

它是 **LangSmith 平台侧**的 LLM 评委，走 LangSmith 自己的模型额度。
你截图里 "Tracks LLM-as-a-judge evaluator spend for models from providers: Anthropic, Bedrock, Google Gemini, OpenAI" ——
**DeepSeek 不在支持列表里**，所以它要么需要你在 Settings 里配 OpenAI key 并充值，要么就一直不出分。

**两种处理方式，任选：**

- **推荐**：把那个 UI Evaluator 关掉（Enabled 开关关掉），只用脚本里的评委（DeepSeek，免费）。两者 feedback key 都叫 `correctness`，同时开会打架。
- 或者：在 LangSmith Settings 里配 OpenAI key，让它跑，那就要花钱。

---

## 6. 答辩怎么说（关键）

**说法模板：**

> "我们用 LangSmith 做自动化评测。数据集是自己构造的 10 条单轮问答 gold set，覆盖蜜源检索、花期查询、推荐、养蜂知识四类；
> 评测方式是用 SDK 脚本逐条调用系统的 `/api/chat` 接口，再用 DeepSeek 作为 LLM-as-judge 裁判，
> 按 1/0.7/0.4/0 四档比对回答与标准答案的事实正确性，最终 correctness 平均 X 分。"

**红线提醒：**

1. **0.765 这个数字没有出处，绝对不要报**。用这次实测的分数替代它，评委追问"怎么测的"你能正面接住。
2. 别说是 Playground 测的 —— 你用的是 SDK + 自建数据集。
3. 评委问"10 条是不是太少了"，正面承认：**"这是演示用 gold set，不是学术评测集；工程上已把评测链路跑通，扩到百条只需继续往 CSV 里加数据。"**
4. `has_sources` 这个辅助指标可以主动提——它证明"检索真的命中了"，不是模型凭空生成。

---

## 7. 文件清单

| 文件 | 作用 |
|---|---|
| `E:\rag_system\langsmith_dataset.csv` | 10 条单轮 gold 数据集（question / expected_answer / category） |
| `E:\rag_system\evaluate_rag_langsmith.py` | 单轮 SDK 评测脚本（自动登录 + 调接口 + DeepSeek 评委） |
| `E:\rag_system\langsmith_multiturn_dataset.csv` | 8 条多轮 gold 数据集（question / history_json / expected_answer / category） |
| `E:\rag_system\evaluate_rag_multiturn_langsmith.py` | 多轮（查询重写/指代消解）SDK 评测脚本 |
| 已装 `langsmith` / `openai` 的 Python 环境 | 哪个解释器装了就填哪个（`pip install langsmith openai requests`） |

---

## 8. 多轮对话（查询重写）评测 —— 额外一份

单轮脚本传 `history=[]`，查询重写**不触发**（代码 `rewrite_query` 第758行：`if not history: return question`）。
要验证"指代消解"功能，必须传真实历史，用这份多轮脚本。

### 它能证明什么
回答你之前的核心疑问——**"一句话本来就能独立检索的，还会不会触发重写？"**
答案是：只有**带历史**的追问才会进重写分支；首轮/独立问句零开销、直接原样检索。
多轮脚本传了前一轮对话，强制触发 `rewrite_query`，测试系统能否把
"那里 / 它 / 它们 / 这个 / 那X呢 / 上面说的药" 等指代补全成可独立检索的查询。

### 命令（同样三条）
```bat
:: 1. 设 key（正式跑才需要）
set LANGSMITH_API_KEY=lsv2_pt_xxxxxxxx
set LANGSMITH_PROJECT=智蜂RAG评测-多轮
:: 2. 本地自检（不联网，验证链路）
...\envs\default\Scripts\python.exe evaluate_rag_multiturn_langsmith.py --dry-run
:: 3. 导出 CSV（可选，去网页手动上传查看）
...\envs\default\Scripts\python.exe evaluate_rag_multiturn_langsmith.py --export-csv langsmith_multiturn_dataset.csv
:: 4. 正式跑（自动建数据集+跑 experiment）
...\envs\default\Scripts\python.exe evaluate_rag_multiturn_langsmith.py
```

### 本地自检实测结果（已验证）
8 条追问的指代全部正确消解，DeepSeek 评委打分：

| # | 类别 | 追问 | 应消解到 | 评分 |
|---|---|---|---|---|
| 1 | 那里 | 那里产蜜量高吗？ | 荆条 | 1.0 |
| 2 | 它 | 它的采蜜期是几月？ | 荔枝 | 1.0 |
| 3 | 那X呢 | 那陕西呢？ | 陕西省 | 1.0 |
| 4 | 这个 | 这个蜜源适合养什么蜂？ | 刺槐 | 0.7（实体对，细节略超范围） |
| 5 | 它们 | 它们的分布省份是哪些？ | 冬季辅助蜜源 | 0.7（部分未给省份） |
| 6 | 它们 | 那它们的花期重叠吗？ | 荆条+枣树 | 1.0 |
| 7 | 上面说的 | 上面说的那个药怎么用？ | 蜂螨药剂 | 1.0 |
| 8 | 它(多跳) | 那它的最佳采蜜月份是？ | 陕西汉中刺槐 | 1.0 |

**rewrite_correctness 平均 ≈ 0.925**。这条链路直接证明查询重写功能在真实接口上生效，
答辩时可以说："我们用 8 条多轮指代用例做端到端验证，指代消解正确性（LLM-as-judge）平均 0.925，
覆盖了'那里/它/它们/这个/话题切换'等典型指代类型。"

### 答辩话术（查询重写）
> "多轮对话在**两层**实现：生成层把 `history[-8:]` 拼进 prompt 维持上下文；**召回层**在向量化前
> 先调 `rewrite_query` 把指代补全（`history[-6:]`），避免'那里产蜜量高吗'这类追问被当成独立句而跑题。
> 重写仅在有历史且 LLM 开启时触发，首轮零开销；任何异常都回退原问题，不影响主流程。
> 我们用 LangSmith 8 条多轮用例实测指代消解正确率 0.925。"

