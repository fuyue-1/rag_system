# -*- coding: utf-8 -*-
"""
BP神经网络 + 改进D-S证据理论 融合评估服务
独立运行于 5002 端口（与主系统5000、溯源系统5001互不冲突）

启动：python fusion_app.py  （或双击 启动融合评估系统.bat）

设计说明：页面采用「表单提交 + 服务端渲染」，不依赖任何前端 JS，
因此在预览面板、iframe、老旧浏览器中均可正常点击使用。
"""
from flask import Flask, request, render_template_string

from bp_ds_fusion import (
    FOCAL, FOCAL_NAMES, dempster, combine_improved, pignistic, decide,
    _vec, FusionSystem,
)

app = Flask(__name__)

print("=" * 60)
print("[初始化] 正在训练3个BP神经网络证据生成器（约10秒）...")
fusion = FusionSystem(epochs=600)
print("[初始化] 训练完成，服务就绪 → http://127.0.0.1:5002")
print("=" * 60)

# ---------------- 表单字段定义 ----------------
FIELDS = [
    ("temp", "气温", "℃", -5, 40, 26),
    ("humidity", "相对湿度", "%", 20, 100, 70),
    ("sunshine", "日照时长", "h", 0, 12, 7),
    ("rain", "降水量", "mm", 0, 60, 2),
    ("coverage", "蜜源覆盖度", "%", 0, 100, 75),
    ("altitude", "海拔", "m", 0, 3500, 800),
    ("water_dist", "水源距离", "km", 0, 10, 1.2),
    ("month", "月份", "月", 1, 12, 7),
    ("bloom_index", "花期指数", "0~1", 0, 1, 0.9),
]
WEATHER_KEYS = ["temp", "humidity", "sunshine", "rain"]
GEO_KEYS = ["coverage", "altitude", "water_dist"]
PHENO_KEYS = ["month", "bloom_index"]

BAR_CLS = ["b-y", "b-m", "b-n", "b-u"]


def pct(v):
    return f"{v * 100:.1f}%"


def bar_cell(name, m):
    """一行：证据源名称 + 4个焦元的百分比与比例条"""
    tds = ""
    for i, k in enumerate(FOCAL_NAMES):
        v = m.get(k, 0.0)
        tds += (f'<td>{pct(v)}<div class="bar-wrap">'
                f'<div class="bar {BAR_CLS[i]}" style="width:'
                f'{min(v * 100, 100):.1f}%"></div></div></td>')
    return f"<tr><td><b>{name}</b></td>{tds}</tr>"


def result_block(r: dict) -> str:
    cls = {"适宜": "d-good", "一般": "d-mid"}.get(r["decision"], "d-bad")
    rows = ('<tr><th>证据源</th><th>适宜</th><th>一般</th>'
            '<th>不适宜</th><th>Θ(不确定)</th></tr>')
    rows += bar_cell("气象BP证据", r["bpa_weather"])
    rows += bar_cell("地理BP证据", r["bpa_geography"])
    rows += bar_cell("物候BP证据", r["bpa_phenology"])
    rows += bar_cell("融合后 mass", r["fused_mass"])
    probs = r["pignistic_prob"]
    k = r["conflict_K"]
    return f"""
    <div class="card">
      <h2>融合评估结果</h2>
      <div style="margin-bottom:14px">决策结论：
        <span class="decision {cls}">{r['decision']}</span></div>
      <table>{rows}</table>
      <div class="note">
        Pignistic 决策概率：适宜 {pct(probs['适宜'])} · 一般 {pct(probs['一般'])}
        · 不适宜 {pct(probs['不适宜'])}<br>
        证据间冲突系数 K：气象-地理 {k['weather-geo']} · 气象-物候
        {k['weather-pheno']} · 地理-物候 {k['geo-pheno']}
      </div>
    </div>"""


def compare_block() -> str:
    m1 = {FOCAL[0]: 0.90, FOCAL[3]: 0.10}
    m2 = {FOCAL[0]: 0.80, FOCAL[3]: 0.20}
    m3 = {FOCAL[2]: 0.99, FOCAL[3]: 0.01}
    m12, _ = dempster(m1, m2)
    classic, K = dempster(m12, m3)
    improved = combine_improved(m1, m2, m3)

    def fmt(m):
        return {FOCAL_NAMES[i]: round(float(_vec(m)[i]), 4) for i in range(4)}

    cm, im = fmt(classic), fmt(improved)
    cd, idc = decide(pignistic(classic)), decide(pignistic(improved))

    def cells(m):
        return "".join(f"<td>{pct(m[k])}</td>" for k in FOCAL_NAMES)

    ev_rows = ""
    for name, desc, m in (("证据1", "气象数据源", m1),
                          ("证据2", "地理数据源", m2),
                          ("证据3", "物候数据源（模拟失真）", m3)):
        ev_rows += (f'<tr><td>{name}<br><small style="color:#6b7f6b">'
                    f'{desc}</small></td>{cells(fmt(m))}</tr>')

    return f"""
    <div class="card">
      <h2>对比实验：异常证据下的经典D-S vs 改进D-S</h2>
      <table>
        <tr><th>证据</th><th>适宜</th><th>一般</th><th>不适宜</th><th>Θ</th></tr>
        {ev_rows}
      </table>
      <table style="margin-top:10px">
        <tr><th>方法</th><th>适宜</th><th>一般</th><th>不适宜</th><th>Θ</th><th>决策</th></tr>
        <tr><td>经典Dempster（K={K:.4f}）</td>{cells(cm)}
            <td class="bad">{cd} ✗</td></tr>
        <tr><td>改进D-S（本文）</td>{cells(im)}
            <td class="good">{idc} ✓</td></tr>
      </table>
      <div class="note">模拟某数据源（如某气象站）数据失真，与另外两源结论强烈冲突：
        经典Dempster规则被单一异常证据"翻盘"，误判为不适宜；改进方法通过Jousselme证据距离
        计算权重，自动给异常证据降权，保持多数一致证据的主导地位——这是"改进"二字的核心证据。</div>
    </div>"""


# ---------------- 页面 ----------------
PAGE = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>智慧养蜂 · 蜜源环境融合评估</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Microsoft YaHei', sans-serif; background: #f4f7f4;
         color: #2c3e2c; padding: 24px; }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 22px; color: #1e5631; margin-bottom: 4px; }
  .sub { color: #6b7f6b; font-size: 13px; margin-bottom: 20px; }
  .card { background: #fff; border-radius: 12px; padding: 20px 22px;
          margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,60,20,.08); }
  h2 { font-size: 15px; color: #1e5631; margin-bottom: 12px;
       border-left: 4px solid #2F9E62; padding-left: 8px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
          gap: 14px; }
  .field label { display: block; font-size: 12px; color: #556; margin-bottom: 4px; }
  .field input { width: 100%; padding: 8px 10px; border: 1px solid #cfd8cf;
                 border-radius: 8px; font-size: 14px; }
  .field input:focus { outline: none; border-color: #2F9E62; }
  .btn { background: #2F9E62; color: #fff; border: none; border-radius: 8px;
         padding: 10px 28px; font-size: 15px; cursor: pointer; margin-top: 14px; }
  .btn:hover { background: #278654; }
  .decision { display: inline-block; font-size: 20px; font-weight: bold;
              padding: 8px 26px; border-radius: 10px; color: #fff; }
  .d-good { background: #2F9E62; } .d-mid { background: #d9930d; }
  .d-bad { background: #c0392b; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { border: 1px solid #dde5dd; padding: 6px 10px; text-align: center; }
  th { background: #eaf4ec; color: #1e5631; }
  .bar-wrap { background: #eef2ee; border-radius: 6px; height: 14px;
              overflow: hidden; margin-top: 4px; }
  .bar { height: 100%; border-radius: 6px; }
  .b-y { background: #2F9E62; } .b-m { background: #d9930d; }
  .b-n { background: #c0392b; } .b-u { background: #8aa08a; }
  .note { font-size: 12px; color: #6b7f6b; margin-top: 8px; line-height: 1.7; }
  .good { color: #1e7a45; font-weight: bold; }
  .bad  { color: #c0392b; font-weight: bold; }
  .err  { background: #fdecea; border-left: 4px solid #c0392b; color: #8a2b22;
          padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 16px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>智慧养蜂 · 蜜源环境融合评估</h1>
  <div class="sub">BP神经网络证据生成 × 改进D-S证据理论融合</div>

  {{ err|safe }}
  <div class="card">
    <h2>输入蜜源地多源数据</h2>
    <form method="post" action="/">
    <div class="grid">
      {{ fields|safe }}
    </div>
    <button class="btn" type="submit">开始融合评估</button>
    </form>
  </div>

  {{ result|safe }}

  {{ compare|safe }}
</div>
</body>
</html>
"""


def _read_values(src) -> dict:
    vals = {}
    for key, _label, _unit, lo, hi, default in FIELDS:
        raw = src.get(key, "")
        try:
            vals[key] = float(raw)
        except (TypeError, ValueError):
            vals[key] = float(default)
    return vals


def _render(values: dict, error: str = "", result: str = ""):
    fields_html = ""
    for key, label, unit, lo, hi, default in FIELDS:
        v = values.get(key, default)
        fields_html += (
            f'<div class="field"><label>{label}（{lo} ~ {hi} {unit}）</label>'
            f'<input name="{key}" value="{v}"></div>')
    err_html = f'<div class="err">{error}</div>' if error else ""
    return render_template_string(
        PAGE, fields=fields_html, result=result,
        compare=compare_block(), err=err_html)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return _render({key: default for key, *_r, default in FIELDS})
    try:
        v = _read_values(request.form)
        r = fusion.evaluate_site(
            weather={k: v[k] for k in WEATHER_KEYS},
            geography={k: v[k] for k in GEO_KEYS},
            phenology={k: v[k] for k in PHENO_KEYS},
        )
        return _render(v, result=result_block(r))
    except Exception as e:                       # 输入异常时保留已填内容
        return _render(_read_values(request.form), error=f"评估失败：{e}")


# ---------------- API（供其他模块/前端调用） ----------------
@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    d = request.get_json(force=True)
    try:
        result = fusion.evaluate_site(
            weather={k: float(d.get(k, v)) for k, v in
                     {"temp": 25, "humidity": 65, "sunshine": 6, "rain": 3}.items()},
            geography={k: float(d.get(k, v)) for k, v in
                       {"coverage": 60, "altitude": 800, "water_dist": 2.0}.items()},
            phenology={k: float(d.get(k, v)) for k, v in
                       {"month": 6, "bloom_index": 0.7}.items()},
        )
        return jsonify_ok(result)
    except Exception as e:
        from flask import jsonify
        return jsonify(ok=False, error=str(e)), 400


def jsonify_ok(result):
    from flask import jsonify
    return jsonify(ok=True, **result)


@app.route("/api/compare", methods=["GET"])
def api_compare():
    from flask import jsonify
    m1 = {FOCAL[0]: 0.90, FOCAL[3]: 0.10}
    m2 = {FOCAL[0]: 0.80, FOCAL[3]: 0.20}
    m3 = {FOCAL[2]: 0.99, FOCAL[3]: 0.01}
    m12, _ = dempster(m1, m2)
    classic, K = dempster(m12, m3)
    improved = combine_improved(m1, m2, m3)
    fmt = lambda m: {FOCAL_NAMES[i]: round(float(_vec(m)[i]), 4) for i in range(4)}
    return jsonify(
        evidence=[{"name": "证据1", "desc": "气象数据源", "bpa": fmt(m1)},
                  {"name": "证据2", "desc": "地理数据源", "bpa": fmt(m2)},
                  {"name": "证据3", "desc": "物候数据源（模拟失真）", "bpa": fmt(m3)}],
        classic={"mass": fmt(classic), "decision": decide(pignistic(classic)),
                 "K": round(float(K), 4)},
        improved={"mass": fmt(improved), "decision": decide(pignistic(improved))},
    )


if __name__ == "__main__":
    # 与主系统 5000、溯源系统 5001 保持一致，监听 0.0.0.0：
    # 这样从局域网其他设备（如手机、答辩用笔记本）访问主系统时，
    # 页面上的"环境评估"按钮也能连到本机 5002 端口。
    app.run(host="0.0.0.0", port=5002, debug=False)
