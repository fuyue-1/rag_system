# -*- coding: utf-8 -*-
"""
智慧养蜂项目 —— 多源数据融合评估模块
（对应成员2：数据融合与评估算法负责人；成员1参与算法设计）

整体架构
========
    多源数据(气象/蜜源地理/物候时间)
        │
        ▼  各自特征向量
    ┌─────────────────────────────┐
    │  BP神经网络 × 3（特征提取）   │   每个证据源训练一个BP网络，
    │  特征 → 基本概率赋值 BPA     │   输出对该证据源的信任分配
    └─────────────────────────────┘
        │  m1(气象证据)  m2(地理证据)  m3(物候证据)
        ▼
    ┌─────────────────────────────┐
    │ 改进的D-S证据理论            │   ① Jousselme证据距离 → 相似度矩阵
    │  ① 证据距离计算证据可信度    │   ② 可信度归一化为权重
    │  ② 加权平均得到平均证据      │   ③ 平均证据自组合 n-1 次
    │  ③ Dempster组合规则融合      │   （解决高冲突下的Zadeh悖论）
    └─────────────────────────────┘
        │
        ▼
    Pignistic概率(BetP) → 蜜源环境适宜性决策（适宜/一般/不适宜）

运行方式
========
    python bp_ds_fusion.py
依赖：numpy  （pip install numpy）

给队友的对接说明
================
    from bp_ds_fusion import FusionSystem
    fs = FusionSystem()          # 构造时自动训练3个BP网络
    result = fs.evaluate_site(
        weather={"temp": 26, "humidity": 70, "sunshine": 7, "rain": 2},
        geography={"coverage": 75, "altitude": 800, "water_dist": 1.2},
        phenology={"month": 7, "bloom_index": 0.9},
    )
    # result 含各证据BPA、融合后mass、BetP概率、决策标签
"""
import json
import numpy as np

np.random.seed(42)

# ============================================================
# 一、识别框架与焦元定义
#    Θ = {Y(适宜), M(一般), N(不适宜)}
#    焦元：三个单例 + 全集Θ(表示"不确定")
#    BP输出层4个softmax值依次对应这4个焦元的mass值
# ============================================================
LABEL_Y, LABEL_M, LABEL_N = "适宜", "一般", "不适宜"
FOCAL = [frozenset({LABEL_Y}), frozenset({LABEL_M}),
         frozenset({LABEL_N}), frozenset({LABEL_Y, LABEL_M, LABEL_N})]
FOCAL_NAMES = ["适宜", "一般", "不适宜", "Θ(不确定)"]


# ============================================================
# 二、BP神经网络（纯numpy手写，便于答辩讲解网络结构）
#    结构：输入层 → 隐层(tanh) → 输出层(softmax)
#    损失：交叉熵；优化：梯度下降（误差反向传播）
# ============================================================
class BPNeuralNetwork:
    def __init__(self, n_in, n_hidden=12, n_out=4, lr=0.08):
        self.n_in, self.n_hidden, self.n_out, self.lr = n_in, n_hidden, n_out, lr
        limit = np.sqrt(6.0 / (n_in + n_hidden))
        self.W1 = np.random.uniform(-limit, limit, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        limit2 = np.sqrt(6.0 / (n_hidden + n_out))
        self.W2 = np.random.uniform(-limit2, limit2, (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        X = np.atleast_2d(X)
        self.h = np.tanh(X @ self.W1 + self.b1)
        self.out = self._softmax(self.h @ self.W2 + self.b2)
        return self.out

    def backward(self, X, Y):
        """Y为BPA标签(softmax目标)，交叉熵梯度"""
        X = np.atleast_2d(X)
        m = X.shape[0]
        dZ2 = (self.out - Y) / m                    # softmax+交叉熵联合梯度
        dW2 = self.h.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dH = dZ2 @ self.W2.T
        dZ1 = dH * (1 - self.h ** 2)                # tanh导数
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        for p, g in ((self.W1, dW1), (self.b1, db1),
                     (self.W2, dW2), (self.b2, db2)):
            p -= self.lr * g

    def train(self, X, Y, epochs=600, batch=32, verbose_name=""):
        n = X.shape[0]
        for ep in range(1, epochs + 1):
            idx = np.random.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                self.forward(X[b])
                self.backward(X[b], Y[b])
            if ep % 150 == 0 or ep == 1:
                pred = self.forward(X)
                loss = -np.mean(np.sum(Y * np.log(pred + 1e-12), axis=1))
                acc = np.mean(pred.argmax(1) == Y.argmax(1))
                print(f"    [{verbose_name}] epoch={ep:4d}  loss={loss:.4f}  acc={acc:.3f}")

    def bpa(self, x):
        """输入单个特征向量，返回该证据源的BPA(dict: 焦元->mass)"""
        p = self.forward(np.asarray(x, dtype=float).reshape(1, -1))[0]
        return {foc: float(v) for foc, v in zip(FOCAL, p)}


# ============================================================
# 三、改进的D-S证据理论
#    经典Dempster规则在高冲突(K→1)时会产生违反直觉的结果
#    (Zadeh悖论：两个高置信但矛盾的证据融合出荒谬结论)。
#    本实现采用"基于Jousselme证据距离的加权平均"改进(Deng, 2006)：
#      ① 计算两两证据距离 d(i,j)，相似度 sim = 1 - d
#      ② 证据i的支持度 sup(i) = Σ_{j≠i} sim(i,j)
#      ③ 权重 w(i) = sup(i)/Σsup，加权平均得平均证据 m̄
#      ④ 用Dempster规则将 m̄ 自身组合 n-1 次
#    效果：可信证据权重高，异常/冲突证据被自动降权。
# ============================================================
_JACCARD = np.array([[len(a & b) / len(a | b) if (a | b) else 1.0
                      for b in FOCAL] for a in FOCAL], dtype=float)


def _vec(m: dict) -> np.ndarray:
    return np.array([m.get(f, 0.0) for f in FOCAL])


def _to_mass(v: np.ndarray) -> dict:
    return {FOCAL[i]: float(max(v[i], 0.0)) for i in range(len(FOCAL))}


def evidence_distance(m1: dict, m2: dict) -> float:
    """Jousselme距离: d = sqrt(0.5 * (v1-v2)^T D (v1-v2))"""
    d = _vec(m1) - _vec(m2)
    return float(np.sqrt(max(0.0, 0.5 * d @ _JACCARD @ d)))


def dempster(m1: dict, m2: dict):
    """经典Dempster正交组合规则；返回(组合后mass, 冲突系数K)"""
    combined, conflict = {}, 0.0
    for A, a in m1.items():
        if a <= 0:
            continue
        for B, b in m2.items():
            if b <= 0:
                continue
            inter = A & B
            if inter:
                combined[inter] = combined.get(inter, 0.0) + a * b
            else:
                conflict += a * b
    if conflict >= 1.0:            # 完全冲突，无法组合
        return None, conflict
    for k in combined:
        combined[k] /= (1.0 - conflict)
    return combined, conflict


def weighted_average(masses):
    """基于证据距离的加权平均 → 平均证据 m̄"""
    n = len(masses)
    if n == 1:
        return dict(masses[0])
    # ① 相似度矩阵  ② 支持度  ③ 权重
    sim = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i, j] = 1.0 - evidence_distance(masses[i], masses[j])
    sup = sim.sum(axis=1)
    w = sup / sup.sum()
    mbar = np.zeros(len(FOCAL))
    for wi, m in zip(w, masses):
        mbar += wi * _vec(m)
    return _to_mass(mbar)


def combine_improved(*masses):
    """改进D-S融合入口：加权平均 + 平均证据自组合n-1次"""
    mbar = weighted_average(list(masses))
    result = mbar
    for _ in range(len(masses) - 1):
        result, _ = dempster(result, mbar)
    return result


def pignistic(m: dict) -> dict:
    """Pignistic概率 BetP(A) = Σ_{B∋A} m(B)/|B|，用于最终决策"""
    bet = {LABEL_Y: 0.0, LABEL_M: 0.0, LABEL_N: 0.0}
    for B, mb in m.items():
        if mb <= 0:
            continue
        for a in B:
            bet[a] += mb / len(B)
    return bet


def decide(bet: dict) -> str:
    return max(bet, key=bet.get)


# ============================================================
# 四、训练数据生成：用领域知识规则 + 噪声构造监督样本
#    （实际部署时替换为MySQL中清洗后的真实多源数据）
# ============================================================
def _label_from_score(s: float, rng) -> np.ndarray:
    """适宜性得分s∈[0,1] → 带噪声的BPA标签"""
    if s > 0.65:
        v = np.array([0.68, 0.13, 0.04, 0.15])
    elif s > 0.35:
        v = np.array([0.15, 0.62, 0.10, 0.13])
    else:
        v = np.array([0.05, 0.16, 0.64, 0.15])
    v = v + rng.normal(0, 0.04, 4)
    v = np.clip(v, 0.01, None)
    return v / v.sum()


def make_weather_data(n=900, rng=None):
    """证据源1：气象特征 [气温℃, 相对湿度%, 日照h, 降水量mm]"""
    rng = rng or np.random.default_rng(1)
    X = np.column_stack([
        rng.uniform(-5, 40, n), rng.uniform(20, 100, n),
        rng.uniform(0, 12, n), rng.uniform(0, 60, n)])
    Y = np.zeros((n, 4))
    for i, (t, h, s, r) in enumerate(X):
        score = 0.0
        if 15 <= t <= 32: score += 0.38                    # 蜜蜂采集适温
        elif 10 <= t < 15 or 32 < t <= 35: score += 0.20
        if 55 <= h <= 80: score += 0.26                     # 适宜巢内湿度
        elif 40 <= h < 55 or 80 < h <= 90: score += 0.12
        if s >= 5: score += 0.22                            # 充足日照利于飞行采集
        elif s >= 3: score += 0.10
        if r < 5: score += 0.14                             # 降水少利于出勤
        elif r < 15: score += 0.06
        Y[i] = _label_from_score(min(score, 1.0), rng)
    return X, Y


def make_geo_data(n=900, rng=None):
    """证据源2：蜜源地理特征 [蜜源覆盖度%, 海拔m, 水源距离km]"""
    rng = rng or np.random.default_rng(2)
    X = np.column_stack([
        rng.uniform(0, 100, n), rng.uniform(0, 3500, n), rng.uniform(0, 10, n)])
    Y = np.zeros((n, 4))
    for i, (cov, alt, wd) in enumerate(X):
        score = 0.0
        if cov >= 60: score += 0.45                        # 蜜源植物丰富度
        elif cov >= 30: score += 0.25
        elif cov >= 10: score += 0.10
        if 200 <= alt <= 1500: score += 0.30                # 中低海拔蜜源带
        elif 1500 < alt <= 2200: score += 0.15
        if wd <= 2: score += 0.25                           # 近水源
        elif wd <= 4: score += 0.12
        Y[i] = _label_from_score(min(score, 1.0), rng)
    return X, Y


def make_phenology_data(n=900, rng=None):
    """证据源3：物候时间特征 [月份, 花期指数0-1]"""
    rng = rng or np.random.default_rng(3)
    X = np.column_stack([
        rng.integers(1, 13, n).astype(float), rng.uniform(0, 1, n)])
    Y = np.zeros((n, 4))
    for i, (mo, bloom) in enumerate(X):
        score = 0.0
        if 3 <= mo <= 9: score += 0.35                     # 主要流蜜期
        elif mo in (2, 10): score += 0.12
        score += 0.6 * bloom                                # 当地花期吻合度
        Y[i] = _label_from_score(min(score, 1.0), rng)
    return X, Y


# ============================================================
# 五、对外统一接口：FusionSystem
# ============================================================
class FusionSystem:
    """训练三个BP证据生成器 + 改进D-S融合，输出蜜源环境适宜性评估"""

    def __init__(self, epochs=600):
        print("[训练] 正在训练3个BP神经网络证据生成器 ...")
        self.bp_weather = BPNeuralNetwork(n_in=4)
        self.bp_geo = BPNeuralNetwork(n_in=3)
        self.bp_pheno = BPNeuralNetwork(n_in=2)
        rng = np.random.default_rng(0)
        Xw, Yw = make_weather_data(rng=rng)
        Xg, Yg = make_geo_data(rng=rng)
        Xp, Yp = make_phenology_data(rng=rng)
        # 归一化到[0,1]加速收敛
        self._stats = {
            "weather": (Xw.mean(0), Xw.std(0) + 1e-8),
            "geo": (Xg.mean(0), Xg.std(0) + 1e-8),
            "pheno": (Xp.mean(0), Xp.std(0) + 1e-8),
        }
        Xw = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-8)
        Xg = (Xg - Xg.mean(0)) / (Xg.std(0) + 1e-8)
        Xp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-8)
        self.bp_weather.train(Xw, Yw, epochs=epochs, verbose_name="气象BP")
        self.bp_geo.train(Xg, Yg, epochs=epochs, verbose_name="地理BP")
        self.bp_pheno.train(Xp, Yp, epochs=epochs, verbose_name="物候BP")
        print("[训练] 完成。\n")

    def _norm(self, key, x):
        mu, sd = self._stats[key]
        return (np.asarray(x, dtype=float) - mu) / sd

    def evaluate_site(self, weather: dict, geography: dict, phenology: dict) -> dict:
        """
        输入（示例）：
            weather   = {"temp":26, "humidity":70, "sunshine":7, "rain":2}
            geography = {"coverage":75, "altitude":800, "water_dist":1.2}
            phenology = {"month":7, "bloom_index":0.9}
        输出：各证据BPA、冲突系数、融合mass、BetP概率、决策结论
        """
        m1 = self.bp_weather.bpa(self._norm("weather", [
            weather["temp"], weather["humidity"],
            weather["sunshine"], weather["rain"]]))
        m2 = self.bp_geo.bpa(self._norm("geo", [
            geography["coverage"], geography["altitude"],
            geography["water_dist"]]))
        m3 = self.bp_pheno.bpa(self._norm("pheno", [
            phenology["month"], phenology["bloom_index"]]))

        # 参考：经典Dempster两两组合的冲突系数
        _, k12 = dempster(m1, m2)
        _, k13 = dempster(m1, m3)
        _, k23 = dempster(m2, m3)

        fused = combine_improved(m1, m2, m3)
        bet = pignistic(fused)

        def fmt(m):
            return {FOCAL_NAMES[i]: round(_vec(m)[i], 4) for i in range(4)}

        return {
            "bpa_weather": fmt(m1), "bpa_geography": fmt(m2), "bpa_phenology": fmt(m3),
            "conflict_K": {"weather-geo": round(k12, 4),
                           "weather-pheno": round(k13, 4),
                           "geo-pheno": round(k23, 4)},
            "fused_mass": fmt(fused),
            "pignistic_prob": {k: round(v, 4) for k, v in bet.items()},
            "decision": decide(bet),
        }


# ============================================================
# 六、演示：① 完整融合评估  ② 高冲突对比实验(Zadeh悖论)
# ============================================================
def demo_conflict_robustness():
    """对比实验：异常/失真证据下，经典D-S链式组合 vs 改进D-S（答辩亮点）

    实际系统中某个数据源(如某气象站数据失真)可能给出与其他证据
    强烈矛盾的结论。经典Dempster规则逐级组合时，会被该异常证据
    "翻盘"；改进方法通过证据距离计算权重，自动给异常证据降权。
    """
    print("=" * 64)
    print("对比实验：异常证据下的经典D-S vs 改进D-S")
    print("=" * 64)
    # 证据1、2高置信"适宜"；证据3为失真证据，高置信"不适宜"
    m1 = {FOCAL[0]: 0.90, FOCAL[3]: 0.10}
    m2 = {FOCAL[0]: 0.80, FOCAL[3]: 0.20}
    m3 = {FOCAL[2]: 0.99, FOCAL[3]: 0.01}
    print("证据1: {适宜:0.90, Θ:0.10}")
    print("证据2: {适宜:0.80, Θ:0.20}")
    print("证据3: {不适宜:0.99, Θ:0.01}  ← 模拟某数据源失真\n")

    # 经典做法：逐级两两Dempster组合
    m12, _ = dempster(m1, m2)
    classic, K = dempster(m12, m3)
    classic_fmt = {FOCAL_NAMES[i]: round(_vec(classic)[i], 4) for i in range(4)}
    print(f"经典D-S：先组合证据1、2，再与证据3组合，冲突系数 K={K:.4f}")
    print(f"  结果: {classic_fmt}")
    print(f"  决策: {decide(pignistic(classic))}")
    print("  → 结论被单一异常证据'翻盘'，与多数证据严重矛盾！\n")

    # 改进做法：证据距离加权 + 平均证据自组合
    improved = combine_improved(m1, m2, m3)
    improved_fmt = {FOCAL_NAMES[i]: round(_vec(improved)[i], 4) for i in range(4)}
    print(f"改进D-S：Jousselme证据距离 → 相似度权重（异常证据自动降权）")
    print(f"  结果: {improved_fmt}")
    print(f"  决策: {decide(pignistic(improved))}")
    print("  → 多数一致证据占主导，系统对失真数据源具备鲁棒性。\n")


def main():
    print("=" * 64)
    print("智慧养蜂 - BP神经网络 + 改进D-S证据理论 多源融合评估")
    print("=" * 64)
    fs = FusionSystem()

    print("=" * 64)
    print("场景1：某蜜源地 7月 气温26℃ 湿度70% 日照7h 降水2mm，")
    print("       蜜源覆盖度75% 海拔800m 水源1.2km，花期指数0.9")
    print("=" * 64)
    r1 = fs.evaluate_site(
        weather={"temp": 26, "humidity": 70, "sunshine": 7, "rain": 2},
        geography={"coverage": 75, "altitude": 800, "water_dist": 1.2},
        phenology={"month": 7, "bloom_index": 0.9})
    print(json.dumps(r1, ensure_ascii=False, indent=2))
    print(f"\n>>> 决策结论：该蜜源环境【{r1['decision']}】\n")

    print("=" * 64)
    print("场景2：某地 1月 气温-2℃ 湿度88% 日照1h 降水28mm，")
    print("       蜜源覆盖度12% 海拔2800m 水源7km，花期指数0.05")
    print("=" * 64)
    r2 = fs.evaluate_site(
        weather={"temp": -2, "humidity": 88, "sunshine": 1, "rain": 28},
        geography={"coverage": 12, "altitude": 2800, "water_dist": 7.0},
        phenology={"month": 1, "bloom_index": 0.05})
    print(f">>> 决策结论：该蜜源环境【{r2['decision']}】")
    print(f"    BetP概率: {r2['pignistic_prob']}\n")

    demo_conflict_robustness()

    print("=" * 64)
    print("集成到Web后端示例（Flask路由内调用）：")
    print("=" * 64)
    print("""
    from bp_ds_fusion import FusionSystem
    fusion = FusionSystem()          # 服务启动时训练一次，常驻内存

    @app.route('/api/evaluate', methods=['POST'])
    def evaluate():
        d = request.get_json()
        return jsonify(fusion.evaluate_site(
            weather=d['weather'], geography=d['geography'],
            phenology=d['phenology']))
    """)


if __name__ == "__main__":
    main()
