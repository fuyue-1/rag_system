# -*- coding: utf-8 -*-
"""
智慧养蜂项目 —— 蜂产品溯源系统（后端服务）
============================================
业务流程（消费者视角）：
    蜂农登记蜂场(产地) → 创建采蜜批次(生成唯一溯源码)
    → 质检 → 加工包装(商家) → 物流 → 销售
    消费者购买后，在查询页输入/扫描溯源码，查看全链路时间线
    （产地、蜂农、采蜜时间、质检结果、商家、流转记录）。

防篡改设计（答辩亮点）：
    每条记录(批次/事件)保存 prev_hash 和 data_hash：
        data_hash = SHA256(prev_hash + 规范化记录内容)
    任何一条记录被事后修改，其后所有 data_hash 均无法对上，
    查询接口会返回 chain_valid=false 提醒消费者。
    可表述为"基于哈希链的轻量级可信溯源，等效于联盟链存证思路"。

运行方式
========
    1. 先导入建表脚本： mysql -u root -p < trace_schema.sql
    2. 启动服务：       python trace_app.py
    3. 浏览器访问：     http://127.0.0.1:5001/

依赖：Flask、pymysql （pip install flask pymysql）
配置：通过环境变量 TRACE_DB_* 覆盖，默认连本机MySQL的 bee_trace 库。
"""
import hashlib
import json
import os
import secrets
import string
from datetime import datetime

import pymysql
from flask import Flask, jsonify, request, abort

# ---------------- 数据库配置（优先读环境变量，未设置才用本地默认值） ----------------
DB_CONFIG = dict(
    host=os.environ.get("TRACE_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("TRACE_DB_PORT", 3306)),
    user=os.environ.get("TRACE_DB_USER", "root"),
    # 密码不写死在代码里：部署时用环境变量 TRACE_DB_PASSWORD 覆盖
    password=os.environ.get("TRACE_DB_PASSWORD") or "123456",
    database=os.environ.get("TRACE_DB_NAME", "bee_trace"),
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
)

app = Flask(__name__)

# 环节中文名（前端时间线展示用）
STAGE_CN = {
    "FARMING": "产地登记", "HARVEST": "采蜜", "QUALITY": "质量检测",
    "PROCESSING": "加工包装", "LOGISTICS": "物流运输", "SALE": "销售",
}
# 合法流转顺序（用于校验录入的环节是否合规）
STAGE_ORDER = ["FARMING", "HARVEST", "QUALITY",
               "PROCESSING", "LOGISTICS", "SALE"]


# ============================================================
# 一、核心工具：溯源码生成 + 哈希链
# ============================================================
def gen_trace_code(code: str = None) -> str:
    """
    溯源码规则：BF - yyyymmdd - 6位随机码
    示例：BF-20260830-K7QM3X
    前缀标识产品线，日期便于人工辨识批次，随机码防猜测。

    若传入 code（演示数据固定码），则原样返回，保证演示码可复现。
    """
    if code:
        return code
    alphabet = "".join(c for c in string.ascii_uppercase + string.digits
                       if c not in "IO01")   # 去掉易混淆字符
    rand = "".join(secrets.choice(alphabet) for _ in range(6))
    return f"BF-{datetime.now():%Y%m%d}-{rand}"


def record_hash(prev_hash: str, payload: dict) -> str:
    """
    哈希链核心：
      data_hash = SHA256(prev_hash + 规范化JSON(记录内容))
    JSON按key排序序列化，保证同样的内容在任何机器上哈希一致。
    """
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


def get_last_hash(cur, batch_id: int) -> str:
    """取该批次链上最后一条记录的 data_hash 作为 prev_hash"""
    cur.execute(
        "SELECT data_hash FROM trace_event WHERE batch_id=%s "
        "ORDER BY id DESC LIMIT 1", (batch_id,))
    row = cur.fetchone()
    return row["data_hash"] if row else None


def _batch_payload(d: dict, code: str) -> dict:
    """
    批次记录的规范化载荷（创建与校验共用，保证两侧序列化完全一致）。
    weight_kg 统一格式化为两位小数：MySQL DECIMAL(8,2)读回"320.50"，
    而请求JSON里是320.5，直接str()两侧不一致会导致哈希校验误报。
    """
    return {
        "trace_code": str(code),
        "batch_no": str(d.get("batch_no", "")),
        "honey_type": str(d.get("honey_type", "")),
        "produce_date": str(d.get("produce_date", "")),
        "weight_kg": f"{float(d.get('weight_kg', 0)):.2f}",
        "merchant": str(d.get("merchant", "")),
        "quality_grade": str(d.get("quality_grade", "")),
    }


def verify_chain(batch: dict, events: list) -> bool:
    """
    从批次记录（链头，prev=genesis）起逐条校验：
      ① 每条记录用其 prev_hash 重算 data_hash，内容被改即失败
      ② 相邻记录的 prev_hash 必须严格等于前一条的 data_hash，
         整条链被事后重算重建也会因链接断裂暴露
    """
    if batch["prev_hash"] != "0" * 64:
        return False
    payload = _batch_payload(batch, batch["trace_code"])
    expected = record_hash(batch["prev_hash"], payload)
    if expected != batch["data_hash"]:
        return False
    prev_hash = batch["data_hash"]          # 下一环节应指向的哈希
    for ev in events:
        if ev["prev_hash"] != prev_hash:    # 链接连续性校验
            return False
        payload = {k: str(ev[k]) for k in ("stage", "title", "detail", "operator")}
        if record_hash(ev["prev_hash"], payload) != ev["data_hash"]:
            return False
        prev_hash = ev["data_hash"]
    return True


# ============================================================
# 二、商户端录入接口（需在管理后台登录后调用）
# ============================================================
@app.route("/api/farm/register", methods=["POST"])
def register_farm():
    """蜂场/产地登记"""
    d = request.get_json(force=True)
    required = ["farm_name", "beekeeper", "province", "city",
                "latitude", "longitude"]
    if not all(k in d for k in required):
        return jsonify(ok=False, msg=f"缺少字段: {required}"), 400
    with pymysql.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO bee_farm (farm_name, beekeeper, province, city, "
            "county, latitude, longitude, apiaries) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (d["farm_name"], d["beekeeper"], d["province"], d["city"],
             d.get("county", ""), d["latitude"], d["longitude"],
             d.get("apiaries", 50)))
        conn.commit()
        return jsonify(ok=True, farm_id=cur.lastrowid)


@app.route("/api/batch/create", methods=["POST"])
def create_batch():
    """
    创建采蜜批次：生成溯源码，写入批次记录 + 产地/采蜜两个初始事件。
    请求体示例：
    {
      "farm_id": 1, "batch_no": "B-2026-001",
      "honey_type": "洋槐蜜", "produce_date": "2026-05-20",
      "weight_kg": 320.5, "merchant": "XX蜂业有限公司",
      "quality_grade": "一级", "quality_report": "QR-2026-0521",
      "operator": "张三"
    }
    """
    d = request.get_json(force=True)
    required = ["farm_id", "batch_no", "honey_type", "produce_date",
                "weight_kg", "merchant", "operator"]
    if not all(k in d for k in required):
        return jsonify(ok=False, msg=f"缺少字段: {required}"), 400

    # 允许演示数据传入固定溯源码（trace_code），真实使用不传则随机生成
    code = gen_trace_code(d.get("trace_code"))
    with pymysql.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        # 哈希链构建顺序：genesis(0*64) → 批次记录 → 产地登记 → 采蜜
        genesis = "0" * 64
        batch_payload = _batch_payload(d, code)
        batch_hash = record_hash(genesis, batch_payload)

        cur.execute(
            "INSERT INTO honey_batch (trace_code, farm_id, batch_no, honey_type,"
            " produce_date, weight_kg, merchant, quality_grade, quality_report,"
            " prev_hash, data_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (code, d["farm_id"], d["batch_no"], d["honey_type"],
             d["produce_date"], d["weight_kg"], d["merchant"],
             d.get("quality_grade", ""), d.get("quality_report", ""),
             genesis, batch_hash))
        batch_id = cur.lastrowid

        # 事件1：产地登记（prev指向批次哈希）
        farm_detail = json.dumps({"farm_id": d["farm_id"]}, ensure_ascii=False)
        farm_payload = {"stage": "FARMING", "title": "产地登记",
                        "detail": farm_detail, "operator": d["operator"]}
        farm_hash = record_hash(batch_hash, farm_payload)
        cur.execute(
            "INSERT INTO trace_event (batch_id, stage, title, detail, operator,"
            " prev_hash, data_hash) VALUES (%s,'FARMING','产地登记',%s,%s,%s,%s)",
            (batch_id, farm_detail, d["operator"], batch_hash, farm_hash))

        # 事件2：采蜜（prev指向产地登记哈希）
        harvest_detail = json.dumps(
            {"produce_date": d["produce_date"], "weight_kg": d["weight_kg"]},
            ensure_ascii=False)
        harvest_payload = {"stage": "HARVEST", "title": "采蜜",
                           "detail": harvest_detail, "operator": d["operator"]}
        harvest_hash = record_hash(farm_hash, harvest_payload)
        cur.execute(
            "INSERT INTO trace_event (batch_id, stage, title, detail, operator,"
            " prev_hash, data_hash) VALUES (%s,'HARVEST','采蜜',%s,%s,%s,%s)",
            (batch_id, harvest_detail, d["operator"], farm_hash, harvest_hash))
        conn.commit()
    return jsonify(ok=True, trace_code=code, batch_id=batch_id)


@app.route("/api/event/add", methods=["POST"])
def add_event():
    """
    追加溯源环节（质检/加工/物流/销售）。
    请求体示例：
    {
      "trace_code": "BF-20260830-K7QM3X", "stage": "PROCESSING",
      "title": "低温浓缩灌装", "operator": "李四",
      "detail": {"factory": "XX蜂业加工厂", "process": "40℃低温浓缩"}
    }
    """
    d = request.get_json(force=True)
    if d.get("stage") not in STAGE_ORDER:
        return jsonify(ok=False, msg=f"stage必须是{STAGE_ORDER}之一"), 400
    with pymysql.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM honey_batch WHERE trace_code=%s",
                    (d.get("trace_code"),))
        row = cur.fetchone()
        if not row:
            return jsonify(ok=False, msg="溯源码不存在"), 404
        batch_id = row["id"]

        prev = get_last_hash(cur, batch_id)
        detail = json.dumps(d.get("detail", {}), ensure_ascii=False)
        payload = {"stage": d["stage"], "title": d.get("title", ""),
                   "detail": detail, "operator": d.get("operator", "")}
        dhash = record_hash(prev, payload)
        cur.execute(
            "INSERT INTO trace_event (batch_id, stage, title, detail, operator,"
            " prev_hash, data_hash) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (batch_id, d["stage"], d.get("title", ""), detail,
             d.get("operator", ""), prev, dhash))
        conn.commit()
        return jsonify(ok=True, event_id=cur.lastrowid)


# ============================================================
# 三、消费者查询接口 + 展示页面
# ============================================================
def _load_trace(cur, code: str):
    cur.execute(
        "SELECT b.*, f.farm_name, f.beekeeper, f.province, f.city, f.county,"
        " f.latitude, f.longitude FROM honey_batch b"
        " JOIN bee_farm f ON b.farm_id=f.id WHERE b.trace_code=%s", (code,))
    batch = cur.fetchone()
    if not batch:
        return None, None
    cur.execute("SELECT * FROM trace_event WHERE batch_id=%s ORDER BY id",
                (batch["id"],))
    events = cur.fetchall()
    return batch, events


@app.route("/api/trace/<code>")
def api_trace(code: str):
    """溯源码查询，返回全链路JSON + 链完整性校验结果"""
    with pymysql.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        batch, events = _load_trace(cur, code.strip().upper())
        if not batch:
            return jsonify(ok=False, msg="溯源码不存在，请核对"), 404
        timeline = [{
            "stage": STAGE_CN.get(e["stage"], e["stage"]),
            "stage_en": e["stage"], "title": e["title"],
            "detail": e["detail"], "operator": e["operator"],
            "time": str(e["created_at"]),
        } for e in events]
        return jsonify(ok=True, chain_valid=verify_chain(batch, events),
                       product={
                           "trace_code": batch["trace_code"],
                           "蜜种": batch["honey_type"],
                           "批次号": batch["batch_no"],
                           "采蜜日期": str(batch["produce_date"]),
                           "净含量/批次重量kg": float(batch["weight_kg"]),
                           "质量等级": batch["quality_grade"] or "待检",
                           "质检报告": batch["quality_report"] or "—",
                           "加工销售商家": batch["merchant"],
                           "产地": (f'{batch["province"]}{batch["city"]}'
                                   f'{batch["county"]} · {batch["farm_name"]}'
                                   f'（蜂农：{batch["beekeeper"]}）'),
                           "坐标": f'{float(batch["latitude"])},{float(batch["longitude"])}',
                       }, timeline=timeline)


@app.route("/trace/<code>")
def page_trace(code: str):
    """消费者查询页：输入溯源码 → 展示产品信息 + 全链路时间线"""
    with pymysql.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        batch, events = _load_trace(cur, code.strip().upper())
        if not batch:
            return "<h3 style='font-family:sans-serif'>溯源码不存在，请核对后重试</h3>", 404
        valid = verify_chain(batch, events)
        rows = "".join(
            f"<li><b>[{STAGE_CN.get(e['stage'], e['stage'])}]</b> {e['title']}"
            f" <span style='color:#888'>· {e['created_at']} · 录入人:{e['operator']}</span>"
            f"<br><span style='color:#555'>{e['detail']}</span></li>"
            for e in events)
        banner = ("✅ 溯源链校验通过，记录完整可信" if valid
                  else "⚠️ 警告：溯源链校验失败，该记录可能被篡改！")
        color = "#2F9E62" if valid else "#C0392B"
        return f"""
        <html><head><meta charset="utf-8"><title>蜂产品溯源查询</title></head>
        <body style="font-family:'Microsoft YaHei',sans-serif;max-width:720px;
                     margin:40px auto;padding:0 20px;color:#222">
          <h2>🍯 蜂产品溯源查询结果</h2>
          <p style="background:{color};color:#fff;padding:10px 16px;
             border-radius:8px">{banner}</p>
          <table border="1" cellpadding="8" cellspacing="0"
                 style="border-collapse:collapse;width:100%">
            <tr><th colspan="2">产品信息（溯源码 {batch['trace_code']}）</th></tr>
            <tr><td>蜜种</td><td>{batch['honey_type']}</td></tr>
            <tr><td>采蜜日期</td><td>{batch['produce_date']}</td></tr>
            <tr><td>质量等级 / 质检报告</td><td>{batch['quality_grade'] or '待检'} /
                {batch['quality_report'] or '—'}</td></tr>
            <tr><td>产地</td><td>{batch['province']}{batch['city']}{batch['county']}
                · {batch['farm_name']}（蜂农：{batch['beekeeper']}）</td></tr>
            <tr><td>加工销售商家</td><td>{batch['merchant']}</td></tr>
          </table>
          <h3>全链路流转记录</h3>
          <ol style="line-height:1.9">{rows}</ol>
          <p style="color:#888;font-size:13px">每条记录均含哈希链存证，
          任何事后篡改都会导致校验失败。</p>
        </body></html>"""


@app.route("/")
def index():
    """首页：溯源码查询框 + 已登记批次列表（演示浏览用）"""
    with pymysql.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT b.trace_code, b.honey_type, b.quality_grade, b.produce_date,"
            " b.weight_kg, b.merchant, f.farm_name, f.province, f.city"
            " FROM honey_batch b JOIN bee_farm f ON b.farm_id=f.id"
            " ORDER BY b.id DESC")
        rows = cur.fetchall()
    items = "".join(
        f"<tr><td><a href='/trace/{r['trace_code']}' style='color:#2F9E62;"
        f"font-family:monospace'>{r['trace_code']}</a></td>"
        f"<td>{r['honey_type']}</td><td>{r['quality_grade'] or '待检'}</td>"
        f"<td>{r['produce_date']}</td><td>{float(r['weight_kg']):.1f}kg</td>"
        f"<td>{r['province']}{r['city']}·{r['farm_name']}</td>"
        f"<td>{r['merchant']}</td></tr>" for r in rows)
    return f"""
    <html><head><meta charset="utf-8"><title>蜂产品溯源平台</title></head>
    <body style="font-family:'Microsoft YaHei',sans-serif;max-width:1000px;
                 margin:30px auto;padding:0 20px;color:#222">
      <h2 style="text-align:center">🍯 智慧养蜂 · 蜂产品溯源平台</h2>
      <form action="/trace" onsubmit="location='/trace/'+this.code.value;return false"
            style="text-align:center;margin:24px 0">
        <input name="code" placeholder="输入溯源码查询，如 BF-20260830-NQF82U"
               size="36" style="padding:10px"/>
        <button style="padding:10px 22px;background:#2F9E62;color:#fff;
               border:none;border-radius:6px;cursor:pointer">🔍 查询</button>
      </form>
      <h3>已登记批次（{len(rows)}个，点击溯源码查看全链路）</h3>
      <table border="1" cellpadding="8" cellspacing="0"
             style="border-collapse:collapse;width:100%;font-size:14px">
        <tr style="background:#f5f5f5"><th>溯源码</th><th>蜜种</th><th>等级</th>
          <th>采蜜日期</th><th>重量</th><th>产地</th><th>商家</th></tr>
        {items}
      </table>
      <p style="color:#888;font-size:13px">每条记录均含SHA256哈希链存证，
      任何事后篡改都会导致校验失败。</p>
    </body></html>"""


if __name__ == "__main__":
    print("蜂产品溯源系统启动： http://127.0.0.1:5001/  （Ctrl+C停止）")
    app.run(host="0.0.0.0", port=5001, debug=False)
