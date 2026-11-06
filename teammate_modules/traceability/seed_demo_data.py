# -*- coding: utf-8 -*-
"""批量插入溯源演示数据：6个蜂场、8个批次、完整流转链"""
import json
import os

import pymysql
import requests

B = "http://127.0.0.1:5001"

# 固定溯源码（与启动脚本 启动溯源系统.bat 中演示码一致），保证演示可复现
TRACE_CODES = [
    "BF-20260830-39G6U8",  # B-2026-001 洋槐蜜（演示码）
    "BF-20260830-22FV2W",  # B-2026-002 洋槐蜜
    "BF-20260830-L3ZHZT",  # B-2026-101 椴树蜜
    "BF-20260830-EPA8KK",  # B-2026-201 野坝子蜜
    "BF-20260830-P3XRK3",  # B-2026-301 油菜花蜜
    "BF-20260830-SCEGVQ",  # B-2026-401 中华土蜂蜜
    "BF-20260830-DNT2A6",  # B-2026-501 荔枝蜜
    "BF-20260830-A3Y9XJ",  # B-2026-502 龙眼蜜
]

DB = dict(
    host=os.environ.get("TRACE_DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("TRACE_DB_PORT", 3306)),
    user=os.environ.get("TRACE_DB_USER", "root"),
    password=os.environ.get("TRACE_DB_PASSWORD") or "123456",
    database=os.environ.get("TRACE_DB_NAME", "bee_trace"),
)


def clear_existing():
    """清空旧演示数据，保证可重复执行（不影响真实业务表结构）"""
    try:
        c = pymysql.connect(**DB)
        cur = c.cursor()
        cur.execute("DELETE FROM trace_event")
        cur.execute("DELETE FROM honey_batch")
        cur.execute("DELETE FROM bee_farm")
        cur.execute("ALTER TABLE trace_event AUTO_INCREMENT=1")
        cur.execute("ALTER TABLE honey_batch AUTO_INCREMENT=1")
        cur.execute("ALTER TABLE bee_farm AUTO_INCREMENT=1")
        c.commit()
        c.close()
        print("== 已清空旧演示数据 ==")
    except Exception as e:
        print("清空失败(可忽略，首次执行):", e)

FARMS = [
    # (名称, 蜂农, 省, 市, 县, 纬度, 经度, 蜂箱数)
    ("秦岭洋槐蜂场", "张三", "陕西省", "宝鸡市", "太白县", 34.0583, 107.3197, 120),
    ("长白山椴树蜂场", "李长山", "吉林省", "延边朝鲜族自治州", "安图县", 42.5836, 128.2970, 200),
    ("云贵高原野坝子蜂场", "王秀英", "云南省", "大理白族自治州", "鹤庆县", 26.4580, 100.1770, 150),
    ("祁连山油菜花蜂场", "马建国", "青海省", "海北藏族自治州", "门源回族自治县", 37.3760, 101.6210, 260),
    ("秦岭土蜂蜜生态园", "陈守业", "陕西省", "汉中市", "留坝县", 33.6150, 106.9200, 80),
    ("岭南荔枝蜂场", "黄振华", "广东省", "从化区", "广州市", 23.5452, 113.5860, 180),
]

BATCHES = [
    # (farm_idx, 批次号, 蜜种, 采蜜日期, 重量, 商家, 等级, 质检报告, 操作员)
    (0, "B-2026-001", "洋槐蜜", "2026-05-20", 320.5, "秦岭蜂业有限公司", "一级", "QR-2026-0521", "张三"),
    (0, "B-2026-002", "洋槐蜜", "2026-06-02", 215.0, "秦岭蜂业有限公司", "二级", "QR-2026-0605", "张三"),
    (1, "B-2026-101", "椴树蜜", "2026-07-08", 480.0, "长白山绿色食品公司", "特级", "QR-2026-0710", "李长山"),
    (2, "B-2026-201", "野坝子蜜", "2026-10-15", 260.0, "云贵山货直供公司", "一级", "QR-2026-1018", "王秀英"),
    (3, "B-2026-301", "油菜花蜜", "2026-07-25", 690.0, "祁连山畜牧产品公司", "二级", "QR-2026-0728", "马建国"),
    (4, "B-2026-401", "中华土蜂蜜", "2026-11-05", 85.0, "留坝山珍合作社", "特级", "QR-2026-1108", "陈守业"),
    (5, "B-2026-501", "荔枝蜜", "2026-04-12", 350.0, "岭南蜂业科技公司", "一级", "QR-2026-0415", "黄振华"),
    (5, "B-2026-502", "龙眼蜜", "2026-04-28", 295.5, "岭南蜂业科技公司", "二级", "QR-2026-0501", "黄振华"),
]

# 不同蜜种的流转差异：荔枝蜜走顺丰空运、土蜂蜜走顺丰冷链、大宗蜜走德邦陆运
EVENTS = [
    # stage, title, detail, operator（通用质检+加工）
    ("QUALITY", "质量检测",
     {"波美度": "42.5", "水分": "18%", "抗生素残留": "未检出", "淀粉酶值": "13.2"},
     "质检员·周敏"),
    ("PROCESSING", "低温浓缩灌装",
     {"factory": "无菌灌装车间", "process": "40℃低温浓缩、玻璃瓶灌装"},
     "车间主任·刘强"),
    ("SALE", "门店上架",
     {"store": "XX超市（国贸店）", "shelf_life": "24个月"},
     "销售·钱七"),
]
LOGISTICS = [
    ("顺丰空运", "广州→北京（次日达）", "恒温航空舱"),
    ("顺丰冷链", "汉中→北京（48小时）", "0-15℃冷藏车"),
    ("德邦陆运", "宝鸡→西安→北京", "常温避光"),
    ("京东物流", "延吉→北京（隔日达）", "恒温车厢"),
]


def add_logistics_events(code, carrier, route, condition, op="物流·赵六"):
    requests.post(B + "/api/event/add", json={
        "trace_code": code, "stage": "LOGISTICS", "title": f"{carrier}运输",
        "operator": op, "detail": {"carrier": carrier, "route": route, "condition": condition}})


def main():
    clear_existing()
    farm_ids = []
    print("== 登记蜂场 ==")
    for name, keeper, prov, city, county, lat, lon, apiaries in FARMS:
        r = requests.post(B + "/api/farm/register", json={
            "farm_name": name, "beekeeper": keeper, "province": prov,
            "city": city, "county": county, "latitude": lat,
            "longitude": lon, "apiaries": apiaries}).json()
        farm_ids.append(r["farm_id"])
        print(f"  [{r['farm_id']}] {name}（{prov}{city}{county}）")

    print("\n== 创建批次 + 流转链 ==")
    codes = []
    for i, (fi, batch_no, htype, pdate, weight, merchant, grade, report, op) in enumerate(BATCHES):
        r = requests.post(B + "/api/batch/create", json={
            "farm_id": farm_ids[fi], "batch_no": batch_no, "honey_type": htype,
            "produce_date": pdate, "weight_kg": weight, "merchant": merchant,
            "quality_grade": grade, "quality_report": report, "operator": op,
            "trace_code": TRACE_CODES[i]}).json()
        code = r["trace_code"]
        codes.append((code, htype, grade))
        # 通用质检+加工+物流+销售
        for stage, title, detail, operator in EVENTS[:2]:
            requests.post(B + "/api/event/add", json={
                "trace_code": code, "stage": stage, "title": title,
                "operator": operator,
                "detail": detail if stage != "PROCESSING" else
                          {**detail, "factory": f"{merchant}·{detail['factory']}"}})
        carrier, route, cond = LOGISTICS[i % len(LOGISTICS)]
        add_logistics_events(code, carrier, route, cond)
        for stage, title, detail, operator in EVENTS[2:]:
            requests.post(B + "/api/event/add", json={
                "trace_code": code, "stage": stage, "title": title,
                "operator": operator, "detail": detail})
        print(f"  {code}  {htype}（{grade}）")

    print("\n== 校验全部批次 ==")
    all_ok = True
    for code, htype, grade in codes:
        r = requests.get(B + f"/api/trace/{code}").json()
        ok = r.get("ok") and r.get("chain_valid")
        all_ok = all_ok and ok
        print(f"  {'✅' if ok else '❌'} {code}  {r['product']['蜜种']}  "
              f"{len(r['timeline'])}个环节")
    print(f"\n{'全部校验通过' if all_ok else '存在校验失败！'}")


if __name__ == "__main__":
    main()
