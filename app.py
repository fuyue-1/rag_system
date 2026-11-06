# -*- coding: utf-8 -*-
"""
智慧养蜂 · RAG蜜源信息推荐系统 - Flask后端
API: 智能问答 / 蜜源推荐 / 地图数据 / 知识库 / 统计 / 用户鉴权
"""
import os
import json
import time
import secrets
import pymysql
from functools import wraps
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, session
from werkzeug.security import generate_password_hash, check_password_hash
from rag_engine import RAGEngine, MySQLNectarStore, OUTBOUND_PROXIES

# ===== 加载.env配置 =====
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

app = Flask(__name__, static_folder='static', static_url_path='/static')

# ===== 用户鉴权配置 =====
AUTH_ENABLED = os.environ.get('AUTH_ENABLED', 'true').lower() == 'true'
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 7  # 登录态保留7天

# ===== 初始化RAG引擎 =====
rag_engine = RAGEngine()
mysql_store = MySQLNectarStore(
    host=os.environ.get('MYSQL_HOST', '127.0.0.1'),
    port=int(os.environ.get('MYSQL_PORT', 3306)),
    user=os.environ.get('MYSQL_USER', 'root'),
    password=os.environ.get('MYSQL_PASSWORD', '123456'),
    database=os.environ.get('MYSQL_DATABASE', 'bee_rag')
)

print("=" * 50)
print("  智慧养蜂 · RAG蜜源信息推荐系统")
print("=" * 50)
rag_engine.initialize()
print("=" * 50)
print(f"  服务启动: http://127.0.0.1:5000")
print(f"  登录鉴权: {'已启用（问答/知识库需登录）' if AUTH_ENABLED else '已关闭（免登录）'}")
print("=" * 50)


# ==================== 用户鉴权 ====================
def _init_users_table():
    """确保 users 表存在（首次启动自动建表）"""
    conn = mysql_store._get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                nickname VARCHAR(50) DEFAULT '',
                role VARCHAR(20) DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        cursor.close()
    finally:
        conn.close()


_init_users_table()


def _init_chat_token_log_table():
    """确保 chat_token_log 表存在（记录每次对话的 token 消耗，用于用量统计/计费）"""
    conn = mysql_store._get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_token_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT DEFAULT NULL,
                username VARCHAR(50) DEFAULT '',
                question TEXT,
                answer TEXT,
                model VARCHAR(50) DEFAULT '',
                method VARCHAR(20) DEFAULT '',
                input_tokens INT DEFAULT 0,
                output_tokens INT DEFAULT 0,
                total_tokens INT DEFAULT 0,
                total_sources INT DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_created_at (created_at),
                INDEX idx_user (user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        cursor.close()
    finally:
        conn.close()


_init_chat_token_log_table()


def login_required(f):
    """登录校验装饰器：AUTH_ENABLED=false 时直接放行"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)
        if 'user_id' not in session:
            return jsonify({"success": False, "error": "未登录，请先登录", "code": "AUTH_REQUIRED"}), 401
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    """管理员校验装饰器：未开启鉴权时放行，但会话用户非admin时拒绝"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)
        if 'user_id' not in session:
            return jsonify({"success": False, "error": "未登录，请先登录", "code": "AUTH_REQUIRED"}), 401
        if session.get('role') != 'admin':
            return jsonify({"success": False, "error": "需要管理员权限"}), 403
        return f(*args, **kwargs)
    return wrapper


def _is_username_valid(username: str) -> bool:
    """用户名：2-20位，仅限中文/字母/数字/下划线"""
    if not username or len(username) < 2 or len(username) > 20:
        return False
    for ch in username:
        if not (ch.isalnum() or ch == '_' or '\u4e00' <= ch <= '\u9fff'):
            return False
    return True


@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    """用户注册（首个注册用户自动成为管理员）"""
    try:
        data = request.get_json() or {}
        username = (data.get('username') or '').strip()
        password = data.get('password') or ''
        nickname = (data.get('nickname') or '').strip()[:20]

        if not _is_username_valid(username):
            return jsonify({"success": False, "error": "用户名需为2-20位中文/字母/数字/下划线"}), 400
        if len(password) < 6:
            return jsonify({"success": False, "error": "密码至少6位"}), 400

        conn = mysql_store._get_conn()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute("SELECT COUNT(*) AS cnt FROM users")
            first_user = cursor.fetchone()['cnt'] == 0
            cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
            if cursor.fetchone():
                return jsonify({"success": False, "error": "用户名已被占用"}), 409

            cursor.execute(
                "INSERT INTO users (username, password_hash, nickname, role) VALUES (%s,%s,%s,%s)",
                (username, generate_password_hash(password), nickname or username,
                 'admin' if first_user else 'user')
            )
            conn.commit()
            new_id = cursor.lastrowid
            cursor.close()
        finally:
            conn.close()

        # 注册后自动登录
        session.clear()
        session['user_id'] = new_id
        session['username'] = username
        session['nickname'] = nickname or username
        session['role'] = 'admin' if first_user else 'user'
        session.permanent = True

        return jsonify({
            "success": True,
            "data": {
                "id": new_id,
                "username": username,
                "nickname": nickname or username,
                "role": 'admin' if first_user else 'user',
                "message": "注册成功" + ("（系统首位用户，已授予管理员权限）" if first_user else "")
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    """用户登录"""
    try:
        data = request.get_json() or {}
        username = (data.get('username') or '').strip()
        password = data.get('password') or ''
        if not username or not password:
            return jsonify({"success": False, "error": "请输入用户名和密码"}), 400

        conn = mysql_store._get_conn()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(
                "SELECT id, username, password_hash, nickname, role FROM users WHERE username=%s",
                (username,)
            )
            user = cursor.fetchone()
            cursor.close()
        finally:
            conn.close()

        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({"success": False, "error": "用户名或密码错误"}), 401

        session.clear()
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['nickname'] = user['nickname'] or user['username']
        session['role'] = user['role']
        session.permanent = True

        return jsonify({
            "success": True,
            "data": {
                "id": user['id'],
                "username": user['username'],
                "nickname": user['nickname'] or user['username'],
                "role": user['role'],
                "message": "登录成功"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    """退出登录"""
    session.clear()
    return jsonify({"success": True, "data": {"message": "已退出登录"}})


@app.route('/api/auth/me', methods=['GET'])
def auth_me():
    """当前登录用户信息（未登录返回401）"""
    if not AUTH_ENABLED:
        return jsonify({"success": True, "data": {"auth_enabled": False, "user": None}})
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "未登录", "code": "AUTH_REQUIRED"}), 401
    return jsonify({
        "success": True,
        "data": {
            "auth_enabled": True,
            "user": {
                "id": session['user_id'],
                "username": session.get('username'),
                "nickname": session.get('nickname'),
                "role": session.get('role', 'user')
            }
        }
    })


# ==================== 页面路由 ====================
@app.route('/')
def index():
    return render_template('index.html')


# ==================== API: 智能问答 ====================
def _format_sources(raw_sources):
    """将检索结果格式化为前端需要的来源结构"""
    sources = []
    for s in raw_sources:
        if s.get('type') == 'nectar_source':
            sources.append({
                "type": "nectar_source",
                "id": s.get('id'),
                "plant_name": s.get('plant_name', ''),
                "province": s.get('province', ''),
                "city": s.get('city', ''),
                "location_name": s.get('location_name', ''),
                "lat": s.get('latitude'),
                "lng": s.get('longitude'),
                "flowering_start": s.get('flowering_start'),
                "flowering_end": s.get('flowering_end'),
                "best_honey_start": s.get('best_honey_start'),
                "best_honey_end": s.get('best_honey_end'),
                "honey_yield": s.get('honey_yield'),
                "nectar_sugar_content": s.get('nectar_sugar_content'),
                "score": s.get('score', 0),
                "description": s.get('description', '')[:200]
            })
        else:
            sources.append({
                "type": "knowledge",
                "id": s.get('id'),
                "title": s.get('title', ''),
                "category": s.get('category', ''),
                "content": (s.get('content') or s.get('document', ''))[:300],
                "score": s.get('score', 0)
            })
    return sources


def _parse_chat_request():
    """解析问答请求的公共参数"""
    data = request.get_json() or {}
    question = (data.get('question') or '').strip()
    top_k = int(data.get('top_k', 5))
    province = data.get('province') or None
    month = data.get('month')
    if month is not None and month != '':
        month = int(month)
    history = data.get('history') or []
    # 只保留合法的历史结构
    history = [h for h in history
               if isinstance(h, dict) and h.get('role') in ('user', 'assistant')
               and h.get('content')]
    return question, top_k, province, month, history


def _log_chat_tokens(question, answer, method, token_usage, model, total_sources):
    """把每次对话的 token 消耗落库（chat_token_log 表）。失败不影响主对话流程。"""
    try:
        token_usage = token_usage or {}
        conn = mysql_store._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO chat_token_log
               (user_id, username, question, answer, model, method,
                input_tokens, output_tokens, total_tokens, total_sources, created_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())""",
            (session.get('user_id'), session.get('username'),
             (question or '')[:1000], (answer or '')[:2000], model or '', method or '',
             int(token_usage.get('input_tokens') or 0),
             int(token_usage.get('output_tokens') or 0),
             int(token_usage.get('total_tokens') or 0),
             int(total_sources or 0))
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[token日志] 写入失败（不影响对话）: {e}")


@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """智能问答API - 输入问题，返回RAG检索+LLM生成的回答（支持多轮历史）"""
    try:
        question, top_k, province, month, history = _parse_chat_request()
        if not question:
            return jsonify({"success": False, "error": "请输入问题"}), 400

        result = rag_engine.query(
            question, top_k=top_k, province=province, month=month, history=history
        )

        # 记录本次对话的 token 消耗（含 RAG 生成答案的用量）
        _log_chat_tokens(
            question, result['answer'], result['method'],
            result.get('token_usage'), rag_engine.generator.model,
            result['total_sources'])

        return jsonify({
            "success": True,
            "data": {
                "answer": result['answer'],
                "method": result['method'],
                "sources": _format_sources(result['sources']),
                "total_sources": result['total_sources'],
                "token_usage": result.get('token_usage', {
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
                "query": {
                    "question": question,
                    "province": province,
                    "month": month,
                    "top_k": top_k
                }
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: 流式问答（SSE打字机效果） ====================
@app.route('/api/chat/stream', methods=['POST'])
@login_required
def chat_stream():
    """流式问答API - 先推送检索来源，再逐块推送生成的回答"""
    question, top_k, province, month, history = _parse_chat_request()
    if not question:
        return jsonify({"success": False, "error": "请输入问题"}), 400

    def sse_event(event_type, payload):
        return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"

    def generate():
        try:
            # 1) 检索阶段：查询重写（多轮指代消解）→ 向量检索 → 推送来源
            t0 = time.time()
            sources, search_query = rag_engine.retrieve(
                question, top_k=top_k, province=province,
                month=month, history=history)
            yield sse_event('sources', {
                "sources": _format_sources(sources),
                "total": len(sources),
                "retrieval_ms": int((time.time() - t0) * 1000),
                "rewritten_query": search_query if search_query != question else None
            })

            # 2) 生成阶段：逐块推送回答，并捕获末尾的 usage 标记
            method = 'template'
            usage = None
            answer_parts = []
            for chunk in rag_engine.generator.generate_stream(
                    question, sources, 'general', history):
                if isinstance(chunk, tuple):
                    if chunk[0] == '__method__':
                        method = chunk[1]
                    elif chunk[0] == '__usage__':
                        usage = chunk[1]
                    continue
                answer_parts.append(chunk)
                yield sse_event('delta', {"content": chunk})

            answer_text = ''.join(answer_parts)
            # 记录本次对话的 token 消耗（流式接口此前未统计，这里补上）
            _log_chat_tokens(
                question, answer_text, method, usage,
                rag_engine.generator.model, len(sources))

            yield sse_event('done', {
                "method": method,
                "token_usage": usage or {
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            })
        except Exception as e:
            yield sse_event('error', {"message": str(e)})

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


# ==================== API: token 用量统计 ====================
@app.route('/api/token/log', methods=['GET'])
@login_required
def token_log():
    """查看对话 token 消耗统计：总量 + 今日 + 最近若干条明细"""
    try:
        conn = mysql_store._get_conn()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        # 总量 + 今日量
        cursor.execute("""
            SELECT
                COUNT(*)                       AS total_calls,
                COALESCE(SUM(total_tokens),0)  AS total_tokens,
                COALESCE(SUM(input_tokens),0)  AS total_input,
                COALESCE(SUM(output_tokens),0) AS total_output,
                COALESCE(SUM(total_sources),0) AS total_sources
            FROM chat_token_log
        """)
        summary = cursor.fetchone() or {}
        cursor.execute("""
            SELECT
                COUNT(*)                      AS today_calls,
                COALESCE(SUM(total_tokens),0) AS today_tokens
            FROM chat_token_log
            WHERE DATE(created_at) = CURDATE()
        """)
        today = cursor.fetchone() or {}
        # 最近 50 条明细
        cursor.execute("""
            SELECT id, username, LEFT(question,80) AS question, model, method,
                   input_tokens, output_tokens, total_tokens, total_sources, created_at
            FROM chat_token_log
            ORDER BY id DESC LIMIT 50
        """)
        recent = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify({
            "success": True,
            "data": {
                "summary": summary,
                "today": today,
                "recent": recent,
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: 蜜源推荐 ====================
@app.route('/api/recommend', methods=['POST'])
def recommend():
    """蜜源推荐API - 按省份/月份/植物名筛选，返回推荐列表"""
    try:
        data = request.get_json() or {}
        province = data.get('province') or None
        month = data.get('month')
        if month is not None:
            month = int(month)
        plant_name = data.get('plant_name') or None
        top_k = int(data.get('top_k', 10))

        result = rag_engine.recommend_nectar(
            province=province, month=month,
            plant_name=plant_name, top_k=top_k
        )

        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: 地图数据 ====================
@app.route('/api/map', methods=['GET'])
def map_data():
    """地图数据API - 返回全部蜜源点坐标，供前端地图渲染"""
    try:
        map_points = rag_engine.get_map_data()
        return jsonify({
            "success": True,
            "data": map_points,
            "total": len(map_points)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: 统计信息 ====================
@app.route('/api/stats', methods=['GET'])
def stats():
    """统计信息API"""
    try:
        s = rag_engine.get_stats()
        return jsonify({
            "success": True,
            "data": {
                "nectar_count": s['nectar_count'],
                "plant_count": s['plant_count'],
                "province_count": s['province_count'],
                "knowledge_count": s['knowledge_count'],
                "plant_distribution": s['plant_distribution'],
                "province_distribution": s['province_distribution'],
                "llm_enabled": bool(os.environ.get('LLM_API_KEY', ''))
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: 系统配置 ====================
@app.route('/api/config', methods=['GET'])
def config():
    """系统配置状态"""
    return jsonify({
        "success": True,
        "data": {
            "llm_enabled": bool(os.environ.get('LLM_API_KEY', '')),
            "llm_model": os.environ.get('LLM_MODEL', 'deepseek-chat'),
            "auth_enabled": AUTH_ENABLED,
            "database": "bee_rag (MySQL 8.0)",
            "vector_db": "Chroma (持久化向量数据库)",
            "vector_method": "bge-small-zh本地离线语义向量 (fastembed, 512维) → 存入Chroma",
            "retrieval_method": "混合检索: SQL结构化筛选 + Chroma余弦相似度"
        }
    })


# ==================== API: 今日天气 ====================
# 天气数据源：Open-Meteo（免费、无需key）；地点反查：BigDataCloud（免费、无需key）
# 浏览器无法定位时，使用 .env 配置的默认位置（WEATHER_LAT/WEATHER_LON/WEATHER_LOCATION）
WEATHER_LAT = float(os.environ.get('WEATHER_LAT', '39.9042'))
WEATHER_LON = float(os.environ.get('WEATHER_LON', '116.4074'))
WEATHER_LOCATION = os.environ.get('WEATHER_LOCATION', '')
_weather_cache = {}          # { (lat, lon): (timestamp, data) }
_WEATHER_CACHE_TTL = 300     # 缓存5分钟，避免频繁请求外部接口


def _fetch_weather(lat, lon):
    """调用 Open-Meteo 获取当前温度与湿度"""
    import requests
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m",
        "timezone": "auto"
    }
    resp = requests.get(url, params=params, timeout=6, proxies=OUTBOUND_PROXIES)
    resp.raise_for_status()
    cur = (resp.json() or {}).get('current', {}) or {}
    return {
        "temperature": cur.get('temperature_2m'),
        "humidity": cur.get('relative_humidity_2m'),
        "time": cur.get('time')
    }


def _fetch_location_name(lat, lon):
    """反查中文地点名（尽力而为，失败返回空字符串）"""
    try:
        import requests
        url = "https://api.bigdatacloud.net/data/reverse-geocode-client"
        resp = requests.get(
            url, params={"latitude": lat, "longitude": lon, "localityLanguage": "zh"},
            timeout=4, proxies=OUTBOUND_PROXIES)
        resp.raise_for_status()
        d = resp.json() or {}
        return (d.get('city') or d.get('locality') or d.get('principalSubdivision') or '').strip()
    except Exception:
        return ''


@app.route('/api/weather', methods=['GET'])
def weather():
    """今日天气API - 返回温度、湿度与地点（未传经纬度时用默认位置）"""
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            lat, lon, located = WEATHER_LAT, WEATHER_LON, False
        else:
            located = True

        key = (round(lat, 4), round(lon, 4))
        now = time.time()
        cached = _weather_cache.get(key)
        if cached and now - cached[0] < _WEATHER_CACHE_TTL:
            data = dict(cached[1])
            data['from_cache'] = True
            return jsonify({"success": True, "data": data})

        weather_info = _fetch_weather(lat, lon)
        location = _fetch_location_name(lat, lon) if located else WEATHER_LOCATION
        data = {
            "temperature": weather_info['temperature'],
            "humidity": weather_info['humidity'],
            "time": weather_info['time'],
            "location": location or None,
            "located": located,
            "from_cache": False
        }
        _weather_cache[key] = (now, data)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        # 外部天气服务异常时不阻塞页面：返回 200 + data=null，前端显示 "--"
        return jsonify({"success": False, "error": str(e), "data": None})


# ==================== API: 蜜源详情 ====================
@app.route('/api/nectar/<int:nectar_id>', methods=['GET'])
def nectar_detail(nectar_id):
    """获取单个蜜源详情"""
    try:
        ns = mysql_store.get_by_id(nectar_id)
        if not ns:
            return jsonify({"success": False, "error": "记录不存在"}), 404
        return jsonify({"success": True, "data": ns})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: 知识库 ====================
@app.route('/api/knowledge', methods=['GET'])
@login_required
def knowledge_list():
    """获取知识库列表"""
    try:
        category = request.args.get('category') or None
        keyword = request.args.get('keyword') or None
        entries = mysql_store.search_knowledge(keyword=keyword, category=category)
        summary = [{
            "id": e['id'],
            "title": e['title'],
            "category": e['category'],
            "content_preview": e['content'][:150] + "...",
            "tags": e.get('tags', '')
        } for e in entries]
        return jsonify({"success": True, "data": summary, "total": len(summary)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/knowledge/<int:kid>', methods=['GET'])
@login_required
def knowledge_detail(kid):
    """获取知识库详情"""
    try:
        entries = mysql_store.search_knowledge()
        for e in entries:
            if e['id'] == kid:
                return jsonify({"success": True, "data": e})
        return jsonify({"success": False, "error": "不存在"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: 向量数据库管理 ====================
@app.route('/api/vector/sync', methods=['POST'])
@admin_required
def vector_sync():
    """增量同步向量库：MySQL新增/删除的数据同步到Chroma，无需重启服务器

    使用场景：往MySQL里INSERT了新蜜源/知识后，调用此接口立即生效
    """
    try:
        result = rag_engine.sync_vectors()
        return jsonify({
            "success": True,
            "data": {
                "added": result['added'],
                "removed": result['removed'],
                "total_in_vector_db": result['total'],
                "message": f"同步完成：新增{result['added']}条，删除{result['removed']}条，向量库共{result['total']}条"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/vector/rebuild', methods=['POST'])
@admin_required
def vector_rebuild():
    """全量重建向量库：清空Chroma后从MySQL重新向量化全部数据"""
    try:
        count = rag_engine.rebuild_vectors()
        return jsonify({
            "success": True,
            "data": {"rebuilt": count, "message": f"全量重建完成，共{count}条向量"}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/vector/stats', methods=['GET'])
def vector_stats():
    """向量数据库状态：持久化路径、向量数量、词表大小"""
    try:
        return jsonify({"success": True, "data": rag_engine.vector_stats()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: 新增知识（自动同步向量库） ====================
@app.route('/api/knowledge', methods=['POST'])
@admin_required
def add_knowledge():
    """新增知识库条目并自动同步到向量库（无需重启、无需手动sync）"""
    try:
        data = request.get_json() or {}
        title = (data.get('title') or '').strip()
        category = (data.get('category') or '通用').strip()
        content = (data.get('content') or '').strip()
        tags = (data.get('tags') or '').strip()
        if not title or not content:
            return jsonify({"success": False, "error": "title和content不能为空"}), 400

        conn = mysql_store._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO bee_knowledge (title, category, content, tags) VALUES (%s,%s,%s,%s)",
                (title, category, content, tags)
            )
            conn.commit()
            new_id = cursor.lastrowid
            cursor.close()
        finally:
            conn.close()

        # 自动增量同步向量库
        sync_result = rag_engine.sync_vectors()
        return jsonify({
            "success": True,
            "data": {
                "id": new_id,
                "vector_synced": True,
                "vector_added": sync_result['added'],
                "message": f"知识已入库(id={new_id})，向量库已自动同步"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/nectar', methods=['POST'])
@admin_required
def add_nectar():
    """新增蜜源记录并自动同步到向量库"""
    try:
        data = request.get_json() or {}
        required = ['plant_name', 'province', 'city', 'latitude', 'longitude']
        for field in required:
            if data.get(field) in (None, ''):
                return jsonify({"success": False, "error": f"缺少必填字段: {field}"}), 400

        conn = mysql_store._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO nectar_sources
                (plant_name, plant_category, is_primary, province, city, location_name,
                 latitude, longitude, planting_area, annual_avg_temp, annual_precipitation,
                 sunshine_hours, climate_type, flowering_start, flowering_end,
                 best_honey_start, best_honey_end, honey_yield, nectar_sugar_content,
                 pollen_quality, description)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    data['plant_name'], data.get('plant_category', '草本'),
                    int(data.get('is_primary', 1)), data['province'], data['city'],
                    data.get('location_name', ''), float(data['latitude']),
                    float(data['longitude']), int(data.get('planting_area', 0)),
                    float(data.get('annual_avg_temp', 0)), int(data.get('annual_precipitation', 0)),
                    int(data.get('sunshine_hours', 0)), data.get('climate_type', ''),
                    int(data.get('flowering_start', 1)), int(data.get('flowering_end', 1)),
                    int(data.get('best_honey_start', 1)), int(data.get('best_honey_end', 1)),
                    float(data.get('honey_yield', 0)), float(data.get('nectar_sugar_content', 0)),
                    data.get('pollen_quality', '良'), data.get('description', '')
                )
            )
            conn.commit()
            new_id = cursor.lastrowid
            cursor.close()
        finally:
            conn.close()

        sync_result = rag_engine.sync_vectors()
        return jsonify({
            "success": True,
            "data": {
                "id": new_id,
                "vector_synced": True,
                "vector_added": sync_result['added'],
                "message": f"蜜源已入库(id={new_id})，向量库已自动同步"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: 省份/植物列表 ====================
@app.route('/api/filters', methods=['GET'])
def filters():
    """获取筛选选项：省份列表、植物列表、月份列表"""
    try:
        conn = mysql_store._get_conn()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT DISTINCT province FROM nectar_sources ORDER BY province")
        provinces = [r['province'] for r in cursor.fetchall()]
        cursor.execute("SELECT DISTINCT plant_name FROM nectar_sources ORDER BY plant_name")
        plants = [r['plant_name'] for r in cursor.fetchall()]
        cursor.close()
        conn.close()
        months = list(range(1, 13))
        return jsonify({
            "success": True,
            "data": {
                "provinces": provinces,
                "plants": plants,
                "months": months
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 启动 ====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
