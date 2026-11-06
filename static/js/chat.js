/* ===== 智蜂专家 · RAG蜜源信息推荐系统 · 前端逻辑 ===== */

// ===== 全局状态 =====
let map = null;              // 高德地图实例 (AMap.Map)
let mapMarkers = [];         // 蜜源点覆盖物 (AMap.CircleMarker)
let highlightOverlays = [];  // 高亮圈覆盖物 (AMap.Circle)
let allMarkers = [];         // [{id, data}] 供推荐结果联动定位
let infoWindow = null;       // 气泡窗 (AMap.InfoWindow)
let chatHistory = [];        // 多轮对话历史 [{role, content}]
let isStreaming = false;     // 是否正在流式接收回答
const HISTORY_KEY = 'beeChatHistory';
const HISTORY_MAX = 40;      // 本地最多保存的消息条数
let authState = { enabled: false, user: null };   // 鉴权状态

// ===== 统一请求封装（401时自动弹登录框） =====
async function apiFetch(url, options = {}) {
    const resp = await window.fetch(url, options);
    if (resp.status === 401 && authState.enabled) {
        openAuthModal('登录已过期，请重新登录');
        const err = new Error('AUTH_REQUIRED');
        err.isAuth = true;
        throw err;
    }
    return resp;
}

// ===== Toast 提示 =====
function showToast(message, type = 'info', duration = 2600) {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'toast toast-' + type;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
        el.classList.add('toast-out');
        setTimeout(() => el.remove(), 300);
    }, duration);
}

// ===== 初始化 =====
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initResizableSplitter();
    loadStats();
    loadFilters();
    loadMapData();
    bindEvents();
    initVoice();
    initWeather();   // 今日天气（温度 + 湿度）
    initAuth();      // 鉴权初始化（内部异步，登录后恢复历史）
});

// ===== WGS-84 → GCJ-02 坐标转换 =====
// 数据库存的是 WGS-84 坐标，高德底图使用 GCJ-02（火星坐标），
// 不转换会整体向东南偏移约 300~600 米，因此打点前必须转换。
const GCJ_A = 6378245.0;
const GCJ_EE = 0.00669342162296594323;

function _outOfChina(lat, lng) {
    return (lng < 72.004 || lng > 137.8347 || lat < 0.8293 || lat > 55.8271);
}

function _transformLat(x, y) {
    let ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * Math.sqrt(Math.abs(x));
    ret += (20.0 * Math.sin(6.0 * x * Math.PI) + 20.0 * Math.sin(2.0 * x * Math.PI)) * 2.0 / 3.0;
    ret += (20.0 * Math.sin(y * Math.PI) + 40.0 * Math.sin(y / 3.0 * Math.PI)) * 2.0 / 3.0;
    ret += (160.0 * Math.sin(y / 12.0 * Math.PI) + 320 * Math.sin(y * Math.PI / 30.0)) * 2.0 / 3.0;
    return ret;
}

function _transformLng(x, y) {
    let ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * Math.sqrt(Math.abs(x));
    ret += (20.0 * Math.sin(6.0 * x * Math.PI) + 20.0 * Math.sin(2.0 * x * Math.PI)) * 2.0 / 3.0;
    ret += (20.0 * Math.sin(x * Math.PI) + 40.0 * Math.sin(x / 3.0 * Math.PI)) * 2.0 / 3.0;
    ret += (150.0 * Math.sin(x / 12.0 * Math.PI) + 300.0 * Math.sin(x / 30.0 * Math.PI)) * 2.0 / 3.0;
    return ret;
}

function wgs84ToGcj02(lat, lng) {
    if (_outOfChina(lat, lng)) return { lat, lng };
    let dLat = _transformLat(lng - 105.0, lat - 35.0);
    let dLng = _transformLng(lng - 105.0, lat - 35.0);
    const radLat = lat / 180.0 * Math.PI;
    let magic = Math.sin(radLat);
    magic = 1 - GCJ_EE * magic * magic;
    const sqrtMagic = Math.sqrt(magic);
    dLat = (dLat * 180.0) / ((GCJ_A * (1 - GCJ_EE)) / (magic * sqrtMagic) * Math.PI);
    dLng = (dLng * 180.0) / (GCJ_A / sqrtMagic * Math.cos(radLat) * Math.PI);
    return { lat: lat + dLat, lng: lng + dLng };
}

// ===== 地图初始化（高德地图 JS API 2.0，底图展示免费不限次） =====
// 注意：高德坐标顺序为 [经度, 纬度]，与 Leaflet 的 [纬度, 经度] 相反
const MAP_CENTER_GCJ = [107.0, 35.5];   // 全国视角（GCJ-02）
const MAP_ZOOM = 4;

function initMap() {
    const container = document.getElementById('map');
    if (typeof AMap === 'undefined') {
        container.innerHTML =
            '<div style="padding:24px;color:#8a6d3b;font-size:13px;line-height:1.8">' +
            '地图组件未能加载。请检查：① 网络是否可访问 webapi.amap.com；' +
            '② 是否已在高德开放平台申请 Key 与安全密钥，并替换 index.html 中的占位符。' +
            '<br>（智能问答与蜜源推荐不依赖地图，可正常使用）</div>';
        return;
    }

    map = new AMap.Map('map', {
        center: MAP_CENTER_GCJ,
        zoom: MAP_ZOOM,
        zooms: [3, 14],
        viewMode: '2D'
    });

    infoWindow = new AMap.InfoWindow({
        offset: new AMap.Pixel(0, -12),
        isCustom: true,
        autoMove: true
    });

    // 点击空白处关闭气泡
    map.on('click', () => infoWindow.close());
}

// ===== 清除地图覆盖物 =====
function clearMarkers() {
    if (!map || mapMarkers.length === 0) return;
    map.remove(mapMarkers);
    mapMarkers = [];
    allMarkers = [];
}

function clearHighlights() {
    if (!map || highlightOverlays.length === 0) return;
    map.remove(highlightOverlays);
    highlightOverlays = [];
}

// ===== 在地图上弹出某个蜜源点的详情气泡 =====
function showMarkerInfo(p) {
    if (!map || !infoWindow) return;
    const gcj = wgs84ToGcj02(p.lat, p.lng);
    infoWindow.setContent(
        '<div style="background:#fff;border-radius:8px;padding:10px 12px;' +
        'box-shadow:0 2px 10px rgba(0,0,0,.18);font-size:13px;line-height:1.7;min-width:180px">' +
        '<b>🐝 ' + p.plant_name + '</b><br>' +
        p.province + ' · ' + p.city + '<br>' +
        '<span style="color:#888;font-size:12px">' + (p.location_name || '') + '</span>' +
        '<hr style="margin:6px 0;border:none;border-top:1px solid #eee">' +
        '🌸 花期：' + p.flowering_start + '月 - ' + p.flowering_end + '月<br>' +
        '🍯 采蜜期：' + p.best_honey_start + '月 - ' + p.best_honey_end + '月<br>' +
        '📊 预计产蜜：' + (p.honey_yield || 'N/A') + ' kg/群<br>' +
        (p.is_primary === 1
            ? '<span style="color:#f5a623;font-weight:600">主要蜜源</span>'
            : '<span style="color:#52c41a">辅助蜜源</span>') +
        '</div>');
    infoWindow.open(map, [gcj.lng, gcj.lat]);
}

// ===== 可拖动分隔条 =====
function initResizableSplitter() {
    const splitter = document.getElementById('splitter');
    const rightPanel = document.querySelector('.right-panel');
    const main = document.querySelector('.main-content');
    if (!splitter || !rightPanel || !main) return;

    let isDragging = false;
    let rafId = null;

    const onMove = (e) => {
        if (!isDragging) return;
        const mainRect = main.getBoundingClientRect();
        const splitterHalf = splitter.offsetWidth / 2;
        // 计算右侧面板宽度 = main右边界 - 鼠标x - 分隔条一半
        let newWidth = mainRect.right - e.clientX - splitterHalf;
        // 限制范围：最小320，最大55vw且留至少360给左侧面板
        const maxWidth = Math.min(mainRect.width * 0.55, mainRect.width - 360);
        newWidth = Math.max(320, Math.min(newWidth, maxWidth));
        rightPanel.style.width = newWidth + 'px';

        // 拖动过程中触发地图重绘（节流）
        if (rafId) cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(() => {
            if (map && map.resize) map.resize();
        });
    };

    const onUp = () => {
        if (!isDragging) return;
        isDragging = false;
        splitter.classList.remove('dragging');
        document.body.style.userSelect = '';
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        if (map && map.resize) map.resize();
    };

    splitter.addEventListener('mousedown', (e) => {
        isDragging = true;
        splitter.classList.add('dragging');
        document.body.style.userSelect = 'none';
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        e.preventDefault();
    });
}

// ===== 加载统计信息 =====
async function loadStats() {
    try {
        const resp = await apiFetch('/api/stats');
        const data = await resp.json();
        if (data.success) {
            const d = data.data;
            document.getElementById('statNectar').textContent = `蜜源: ${d.nectar_count}`;
            document.getElementById('statPlant').textContent = `植物: ${d.plant_count}`;
            document.getElementById('statProvince').textContent = `省份: ${d.province_count}`;
        }
        // 加载LLM状态
        const cfgResp = await apiFetch('/api/config');
        const cfgData = await cfgResp.json();
        if (cfgData.success) {
            const llm = cfgData.data.llm_enabled;
            const badge = document.getElementById('statLLM');
            if (llm) {
                badge.textContent = 'LLM: 已启用';
                badge.style.background = 'rgba(82,196,26,0.4)';
            } else {
                badge.textContent = 'LLM: 模板模式';
                badge.style.background = 'rgba(255,255,255,0.25)';
            }
        }
    } catch (e) {
        console.error('加载统计失败:', e);
    }
}

// ===== 加载筛选选项 =====
async function loadFilters() {
    try {
        const resp = await apiFetch('/api/filters');
        const data = await resp.json();
        if (data.success) {
            const d = data.data;
            const provSelect = document.getElementById('filterProvince');
            d.provinces.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p;
                opt.textContent = p;
                provSelect.appendChild(opt);
            });
            const plantSelect = document.getElementById('filterPlant');
            d.plants.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p;
                opt.textContent = p;
                plantSelect.appendChild(opt);
            });
        }
    } catch (e) {
        console.error('加载筛选失败:', e);
    }
}

// ===== 加载地图数据并渲染标记 =====
async function loadMapData() {
    try {
        const resp = await apiFetch('/api/map');
        const data = await resp.json();
        if (data.success) {
            renderMarkers(data.data);
        }
    } catch (e) {
        console.error('加载地图数据失败:', e);
    }
}

// ===== 渲染地图标记 =====
function renderMarkers(points) {
    if (!map) return;
    clearMarkers();

    points.forEach(p => {
        const isPrimary = p.is_primary === 1;
        const gcj = wgs84ToGcj02(p.lat, p.lng);
        p.gcjLat = gcj.lat;
        p.gcjLng = gcj.lng;

        const marker = new AMap.CircleMarker({
            center: [gcj.lng, gcj.lat],
            radius: 7,
            fillColor: isPrimary ? '#f5a623' : '#52c41a',
            fillOpacity: 1,
            strokeColor: '#ffffff',
            strokeWeight: 2,
            cursor: 'pointer'
        });
        marker.on('click', () => showMarkerInfo(p));
        mapMarkers.push(marker);
        allMarkers.push({ id: p.id, data: p });
    });

    map.add(mapMarkers);
}

// ===== 蜜源推荐 =====
async function doRecommend() {
    const province = document.getElementById('filterProvince').value || null;
    const plantName = document.getElementById('filterPlant').value || null;
    const monthVal = document.getElementById('filterMonth').value;
    const month = monthVal ? parseInt(monthVal) : null;

    const btn = document.getElementById('btnRecommend');
    btn.textContent = '🔍 检索中...';
    btn.disabled = true;

    try {
        const resp = await apiFetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ province, month, plant_name: plantName, top_k: 15 })
        });
        const data = await resp.json();

        if (data.success) {
            renderRecommendations(data.data);
            // 在地图上高亮推荐点
            highlightRecommendations(data.data.recommendations);
        } else {
            alert('推荐失败: ' + data.error);
        }
    } catch (e) {
        alert('请求失败: ' + e.message);
    } finally {
        btn.textContent = '🔍 推荐蜜源';
        btn.disabled = false;
    }
}

// ===== 渲染推荐结果 =====
function renderRecommendations(data) {
    const panel = document.getElementById('recommendPanel');
    const list = document.getElementById('recommendList');
    const count = document.getElementById('recommendCount');

    panel.style.display = 'block';
    count.textContent = `（共 ${data.total} 条，显示 ${data.recommendations.length} 条）`;

    list.innerHTML = '';

    if (data.recommendations.length === 0) {
        list.innerHTML = '<div style="padding:20px;text-align:center;color:#999;">未找到符合条件的蜜源，请调整筛选条件</div>';
        return;
    }

    data.recommendations.forEach((r, idx) => {
        const card = document.createElement('div');
        card.className = 'rec-card';
        card.innerHTML = `
            <div class="rec-card-header">
                <div>
                    <span class="rec-plant">${idx + 1}. 🐝 ${r.plant_name}</span>
                    <span class="rec-location">${r.province} · ${r.city} ${r.location_name || ''}</span>
                </div>
                <span class="rec-badge">产蜜 ${r.honey_yield || '?'}kg</span>
            </div>
            <div class="rec-info">
                <span><span class="label">花期:</span> <span class="value">${r.flowering_start}-${r.flowering_end}月</span></span>
                <span><span class="label">采蜜期:</span> <span class="value">${r.best_honey_start}-${r.best_honey_end}月</span></span>
                <span><span class="label">含糖量:</span> <span class="value">${r.nectar_sugar_content || '?'}%</span></span>
                <span><span class="label">年均温:</span> <span class="value">${r.annual_avg_temp || '?'}℃</span></span>
            </div>
        `;
        // 点击卡片在地图上定位并弹出详情
        card.addEventListener('click', () => {
            const target = allMarkers.find(m => m.data.id === r.id);
            const gcj = target ? { lat: target.data.gcjLat, lng: target.data.gcjLng }
                               : wgs84ToGcj02(r.latitude, r.longitude);
            map.setZoomAndCenter(8, [gcj.lng, gcj.lat]);
            showMarkerInfo(target ? target.data : r);
        });
        list.appendChild(card);
    });
}

// ===== 在地图上高亮推荐点 =====
function highlightRecommendations(recs) {
    if (!map) return;
    clearHighlights();

    recs.forEach(r => {
        const gcj = wgs84ToGcj02(r.latitude, r.longitude);
        highlightOverlays.push(new AMap.Circle({
            center: [gcj.lng, gcj.lat],
            radius: 20000,                    // 单位：米
            fillColor: '#ff4d4f',
            fillOpacity: 0.15,
            strokeColor: '#ff4d4f',
            strokeWeight: 2
        }));
    });

    map.add(highlightOverlays);

    // 有推荐点时，自动缩放视野覆盖全部推荐点
    if (highlightOverlays.length > 0) {
        map.setFitView(highlightOverlays, false, [60, 60, 60, 60], 8);
    }
}

// ===== 重置筛选 =====
function resetFilter() {
    document.getElementById('filterProvince').value = '';
    document.getElementById('filterPlant').value = '';
    document.getElementById('filterMonth').value = '';
    document.getElementById('recommendPanel').style.display = 'none';
    clearHighlights();
    if (infoWindow) infoWindow.close();
    if (map) map.setZoomAndCenter(MAP_ZOOM, MAP_CENTER_GCJ);
}

// ===== 智能问答（流式SSE版） =====
function escapeHtml(text) {
    return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

async function sendQuestion(question) {
    if (!question || !question.trim() || isStreaming) return;
    if (!ensureAuthed()) return;   // 未登录 → 弹登录框并终止

    const input = document.getElementById('chatInput');
    const btn = document.getElementById('btnSend');
    input.value = '';
    input.disabled = true;
    btn.disabled = true;
    isStreaming = true;

    // 添加用户消息并记录历史
    addMessage('user', escapeHtml(question));
    chatHistory.push({ role: 'user', content: question });

    // 添加加载动画
    const loadingId = addMessage('bot',
        '<div class="msg-loading"><span></span><span></span><span></span></div>' +
        '<div class="msg-status">正在检索知识库…</div>');

    // 获取筛选条件（如果有）
    const province = document.getElementById('filterProvince').value || null;
    const monthVal = document.getElementById('filterMonth').value;
    const month = monthVal ? parseInt(monthVal) : null;

    let fullAnswer = '';
    let streamContentEl = null;
    let method = 'template';

    try {
        const resp = await apiFetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                top_k: 5,
                province: province,
                month: month,
                history: chatHistory.slice(-8)   // 最近几轮作为上下文
            })
        });
        if (!resp.ok || !resp.body) throw new Error('HTTP ' + resp.status);

        const reader = resp.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // SSE事件以空行分隔
            let sep;
            while ((sep = buffer.indexOf('\n\n')) >= 0) {
                const raw = buffer.slice(0, sep).trim();
                buffer = buffer.slice(sep + 2);
                if (!raw.startsWith('data:')) continue;

                let evt;
                try { evt = JSON.parse(raw.slice(5).trim()); } catch (e) { continue; }

                if (evt.type === 'sources') {
                    // 检索完成：移除loading，渲染来源，创建打字机消息
                    removeMessage(loadingId);
                    renderSources(evt.sources);
                    highlightChatSources(evt.sources);
                    streamContentEl = createStreamMessage();
                    const statusEl = streamContentEl.closest('.msg').querySelector('.msg-status');
                    if (statusEl) {
                        statusEl.textContent = `已检索 ${evt.total} 条相关资料（${evt.retrieval_ms}ms），正在思考…`;
                    }
                } else if (evt.type === 'delta') {
                    fullAnswer += evt.content;
                    if (streamContentEl) updateStreamMessage(streamContentEl, fullAnswer);
                } else if (evt.type === 'done') {
                    method = evt.method || 'template';
                } else if (evt.type === 'error') {
                    removeMessage(loadingId);
                    addMessage('bot', '❌ 出错了: ' + escapeHtml(evt.message || '未知错误'));
                }
            }
        }

        // 收尾：确定消息、更新模式标签、保存历史
        if (streamContentEl) {
            streamContentEl.innerHTML = fullAnswer ? formatAnswer(fullAnswer) : '（未生成回答）';
            if (fullAnswer) {
                const msgEl = streamContentEl.closest('.msg');
                if (msgEl) attachSpeakButton(msgEl, fullAnswer);
            }
        }
        if (fullAnswer) {
            chatHistory.push({ role: 'assistant', content: fullAnswer });
            saveHistory();
        }
        const modeLabel = document.getElementById('chatMode');
        modeLabel.textContent = method === 'llm' ? 'DeepSeek LLM' : '本地模板';
        modeLabel.className = 'chat-mode ' + (method === 'llm' ? 'mode-llm' : 'mode-template');
    } catch (e) {
        removeMessage(loadingId);
        addMessage('bot', '❌ 请求失败: ' + escapeHtml(e.message));
    } finally {
        input.disabled = false;
        btn.disabled = false;
        isStreaming = false;
        input.focus();
    }
}

// ===== 创建流式消息（打字机容器） =====
function createStreamMessage() {
    const container = document.getElementById('chatMessages');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg bot';
    msgDiv.innerHTML = `
        <div class="msg-avatar">🤖</div>
        <div class="msg-content">
            <div class="msg-status">正在思考…</div>
            <div class="msg-body streaming"></div>
        </div>
    `;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    const body = msgDiv.querySelector('.msg-body');
    // 思考状态在收到第一个字后隐藏
    const status = msgDiv.querySelector('.msg-status');
    const observer = () => { if (status) status.style.display = 'none'; };
    body._hideStatus = observer;
    return body;
}

// ===== 更新流式消息内容 =====
function updateStreamMessage(el, text) {
    if (el._hideStatus) { el._hideStatus(); el._hideStatus = null; }
    el.innerHTML = formatAnswer(text) + '<span class="stream-cursor"></span>';
    const container = document.getElementById('chatMessages');
    container.scrollTop = container.scrollHeight;
}

// ===== 对话历史持久化 =====
function saveHistory() {
    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(chatHistory.slice(-HISTORY_MAX)));
    } catch (e) { /* 存储满时静默失败 */ }
}

function restoreHistory() {
    let saved = [];
    try { saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch (e) { return; }
    if (!Array.isArray(saved) || saved.length === 0) return;

    chatHistory = saved.slice(-HISTORY_MAX);
    const container = document.getElementById('chatMessages');
    container.innerHTML = '';
    chatHistory.forEach(m => {
        const content = m.role === 'user' ? escapeHtml(m.content) : formatAnswer(m.content);
        const id = addMessage(m.role, content);
        if (m.role === 'bot' && m.content) {
            const el = document.getElementById(id);
            if (el) attachSpeakButton(el, m.content);
        }
    });
    const hint = document.createElement('div');
    hint.className = 'history-hint';
    hint.textContent = '— 已恢复上次对话（点「新对话」可清空）—';
    container.insertBefore(hint, container.firstChild);
}

// ===== 开始新对话 =====
function newConversation() {
    if (isStreaming) return;
    stopSpeak();
    chatHistory = [];
    try { localStorage.removeItem(HISTORY_KEY); } catch (e) { /* ignore */ }
    document.getElementById('sourcesPanel').style.display = 'none';
    const container = document.getElementById('chatMessages');
    container.innerHTML = `
        <div class="msg bot">
            <div class="msg-avatar">🤖</div>
            <div class="msg-content">
                <p>您好！我是智蜂专家AI助手。我可以帮您：</p>
                <ul>
                    <li>🐝 推荐全国蜜源地（按省份、花期筛选）</li>
                    <li>📍 在地图上查看蜜源分布</li>
                    <li>📅 查询最佳采蜜时间</li>
                    <li>📚 回答养蜂技术问题</li>
                </ul>
                <p>试试点击上方快捷问题，或直接输入您的问题吧！</p>
            </div>
        </div>
    `;
}

// ===== 问答来源在地图上高亮 =====
function highlightChatSources(sources) {
    const nectarSources = (sources || []).filter(s => s.type === 'nectar_source' && s.lat && s.lng);
    if (!map) return;
    clearHighlights();
    if (nectarSources.length === 0) return;

    nectarSources.forEach(s => {
        const gcj = wgs84ToGcj02(s.lat, s.lng);
        highlightOverlays.push(new AMap.Circle({
            center: [gcj.lng, gcj.lat],
            radius: 18000,                    // 单位：米
            fillColor: '#1890ff',
            fillOpacity: 0.15,
            strokeColor: '#1890ff',
            strokeWeight: 2
        }));
    });

    map.add(highlightOverlays);

    if (nectarSources.length <= 3) {
        const gcj = wgs84ToGcj02(nectarSources[0].lat, nectarSources[0].lng);
        map.setZoomAndCenter(6, [gcj.lng, gcj.lat]);
    }
}

// ===== 格式化AI回答（简单Markdown解析） =====
function formatAnswer(text) {
    let html = text;
    // 转义HTML
    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    // 粗体
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // 代码块
    html = html.replace(/`(.+?)`/g, '<code>$1</code>');
    // 换行
    html = html.replace(/\n/g, '<br>');
    return html;
}

// ===== 添加消息 =====
function addMessage(role, content) {
    const container = document.getElementById('chatMessages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${role}`;
    msgDiv.innerHTML = `
        <div class="msg-avatar">${role === 'bot' ? '🤖' : '👤'}</div>
        <div class="msg-content">${content}</div>
    `;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    return msgDiv.id = 'msg-' + Date.now() + '-' + Math.random().toString(36).slice(2, 6);
}

// ===== 移除消息 =====
function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ===== 渲染检索来源 =====
function renderSources(sources) {
    const panel = document.getElementById('sourcesPanel');
    const list = document.getElementById('sourcesList');

    if (!sources || sources.length === 0) {
        panel.style.display = 'none';
        return;
    }

    panel.style.display = 'block';
    list.innerHTML = '';

    sources.forEach(s => {
        const item = document.createElement('div');
        item.className = 'source-item';
        if (s.type === 'nectar_source') {
            item.innerHTML = `
                <span class="src-type">蜜源</span>
                🐝 ${s.plant_name} @ ${s.province}${s.city}
                <span class="src-score">相似度: ${s.score}</span>
            `;
        } else {
            item.innerHTML = `
                <span class="src-type">知识</span>
                📚 ${s.title}
                <span class="src-score">相似度: ${s.score}</span>
            `;
        }
        list.appendChild(item);
    });
}

// ===== 事件绑定 =====
function bindEvents() {
    // 发送按钮
    document.getElementById('btnSend').addEventListener('click', () => {
        const input = document.getElementById('chatInput');
        sendQuestion(input.value);
    });

    // 回车发送
    document.getElementById('chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            sendQuestion(e.target.value);
        }
    });

    // 快捷问题
    document.querySelectorAll('.quick-q').forEach(btn => {
        btn.addEventListener('click', () => {
            sendQuestion(btn.dataset.q);
        });
    });

    // 新对话
    document.getElementById('btnNewChat').addEventListener('click', newConversation);

    // 知识库浏览
    document.getElementById('btnKnowledge').addEventListener('click', openKnowledge);
    // 蜂产品溯源查询（跳转独立溯源服务）
    document.getElementById('btnTrace').addEventListener('click', openTrace);
    document.getElementById('btnFusion').addEventListener('click', openFusion);
    document.getElementById('kbClose').addEventListener('click', closeKnowledge);
    document.getElementById('kbCategory').addEventListener('change', renderKnowledgeList);
    document.getElementById('kbSearch').addEventListener('input', debounce(renderKnowledgeList, 250));
    document.getElementById('kbModal').addEventListener('click', (e) => {
        if (e.target === document.getElementById('kbModal')) closeKnowledge();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeKnowledge();
    });

    // 推荐按钮
    document.getElementById('btnRecommend').addEventListener('click', doRecommend);

    // 重置按钮
    document.getElementById('btnResetFilter').addEventListener('click', resetFilter);

    // 登录/注册
    document.getElementById('tabLogin').addEventListener('click', () => setAuthMode('login'));
    document.getElementById('tabRegister').addEventListener('click', () => setAuthMode('register'));
    document.getElementById('authForm').addEventListener('submit', (e) => {
        e.preventDefault();
        submitAuth();
    });
    document.getElementById('btnLogout').addEventListener('click', logout);
    document.getElementById('btnGuest').addEventListener('click', () => {
        document.getElementById('authModal').style.display = 'none';
        showToast('您当前为游客模式，问答与知识库需要登录后使用');
    });

    // 语音输入
    document.getElementById('btnVoice').addEventListener('click', toggleVoiceInput);
}

// ===== 知识库浏览 =====
let kbData = [];          // 全部知识条目（一次加载，前端过滤）
let kbSelectedId = null;  // 当前选中条目id

function debounce(fn, wait) {
    let timer = null;
    return function (...args) {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), wait);
    };
}

async function openKnowledge() {
    if (!ensureAuthed()) return;   // 未登录 → 弹登录框并终止
    document.getElementById('kbModal').style.display = 'flex';
    if (kbData.length === 0) {
        await loadKnowledgeAll();
    } else {
        renderKnowledgeList();
    }
}

// ===== 蜂产品溯源查询（独立服务，端口5001）=====
// 用当前页面的主机名拼子系统地址：若通过局域网 IP / 域名打开主系统，
// 子系统也走同一台主机，而不是写死 127.0.0.1（否则在别的设备上必然打不开）。
const SUBSYS_HOST = window.location.hostname || '127.0.0.1';
const TRACE_BASE = `http://${SUBSYS_HOST}:5001`;
async function openTrace() {
    // 先探测溯源服务是否在运行（3秒超时）
    try {
        await fetch(TRACE_BASE + '/', { mode: 'no-cors', signal: AbortSignal.timeout(3000) });
    } catch (e) {
        alert(`溯源服务未启动或无法连接：${TRACE_BASE}\n\n请先双击运行项目根目录下的「start_trace.bat」，\n等窗口出现「正在启动服务」后，再点击此按钮。`);
        return;
    }
    window.open(TRACE_BASE + '/', '_blank');
}

// ===== 蜜源环境融合评估（独立服务，端口5002）=====
const FUSION_BASE = `http://${SUBSYS_HOST}:5002`;
async function openFusion() {
    // 先探测融合评估服务是否在运行（3秒超时）
    try {
        await fetch(FUSION_BASE + '/', { mode: 'no-cors', signal: AbortSignal.timeout(3000) });
    } catch (e) {
        alert(`融合评估服务未启动或无法连接：${FUSION_BASE}\n\n请先双击运行项目根目录下的「start_fusion.bat」，\n等 BP 网络训练完成（约10秒）后，再点击此按钮。`);
        return;
    }
    window.open(FUSION_BASE + '/', '_blank');
}

function closeKnowledge() {
    document.getElementById('kbModal').style.display = 'none';
}

async function loadKnowledgeAll() {
    const listEl = document.getElementById('kbList');
    listEl.innerHTML = '<div class="kb-loading">📂 加载中...</div>';
    try {
        const resp = await apiFetch('/api/knowledge');
        const data = await resp.json();
        if (!data.success) throw new Error(data.error || '加载失败');
        kbData = data.data || [];
        // 聚合分类到下拉框
        const catSelect = document.getElementById('kbCategory');
        const cats = [...new Set(kbData.map(x => x.category).filter(Boolean))].sort();
        catSelect.innerHTML = '<option value="">全部分类</option>' +
            cats.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
        renderKnowledgeList();
    } catch (e) {
        listEl.innerHTML = `<div class="kb-loading">⚠️ 加载失败：${escapeHtml(e.message)}</div>`;
    }
}

function renderKnowledgeList() {
    const kw = (document.getElementById('kbSearch').value || '').trim().toLowerCase();
    const cat = document.getElementById('kbCategory').value;
    const list = kbData.filter(x => {
        if (cat && x.category !== cat) return false;
        if (kw) {
            const hay = ((x.title || '') + ' ' + (x.content_preview || '') + ' ' + (x.tags || '')).toLowerCase();
            if (!hay.includes(kw)) return false;
        }
        return true;
    });

    const listEl = document.getElementById('kbList');
    const countEl = document.getElementById('kbCount');
    countEl.textContent = list.length > 0 ? `共 ${list.length} 篇` : '';

    if (list.length === 0) {
        listEl.innerHTML = '<div class="kb-loading">没有找到匹配的文章</div>';
        return;
    }
    listEl.innerHTML = list.map(x => `
        <div class="kb-item ${x.id === kbSelectedId ? 'active' : ''}" data-id="${x.id}">
            <div class="kb-item-title">${escapeHtml(x.title)}</div>
            <div class="kb-item-meta">
                <span class="kb-tag">${escapeHtml(x.category || '通用')}</span>
                <span class="kb-item-preview">${escapeHtml((x.content_preview || '').slice(0, 42))}</span>
            </div>
        </div>
    `).join('');
    listEl.querySelectorAll('.kb-item').forEach(item => {
        item.addEventListener('click', () => showKnowledgeDetail(parseInt(item.dataset.id)));
    });
}

async function showKnowledgeDetail(id) {
    kbSelectedId = id;
    document.querySelectorAll('.kb-item').forEach(el => {
        el.classList.toggle('active', parseInt(el.dataset.id) === id);
    });
    const detailEl = document.getElementById('kbDetail');
    detailEl.innerHTML = '<div class="kb-loading">📖 加载中...</div>';
    try {
        const resp = await apiFetch('/api/knowledge/' + id);
        const data = await resp.json();
        if (!data.success) throw new Error(data.error || '加载失败');
        const x = data.data;
        detailEl.innerHTML = `
            <div class="kb-detail-head">
                <div class="kb-detail-title">${escapeHtml(x.title)}</div>
                <div class="kb-detail-meta">
                    <span class="kb-tag">${escapeHtml(x.category || '通用')}</span>
                    ${x.tags ? `<span class="kb-tag kb-tag-aux"># ${escapeHtml(x.tags)}</span>` : ''}
                </div>
            </div>
            <div class="kb-detail-content">${escapeHtml(x.content || '').replace(/\n/g, '<br>')}</div>
        `;
    } catch (e) {
        detailEl.innerHTML = `<div class="kb-loading">⚠️ 加载失败：${escapeHtml(e.message)}</div>`;
    }
}

// ===== 用户鉴权 =====
let authMode = 'login';   // login / register
let authModalOpen = false;

async function initAuth() {
    try {
        const resp = await apiFetch('/api/config');
        const cfg = await resp.json();
        authState.enabled = !!(cfg.data && cfg.data.auth_enabled);
    } catch (e) { /* 保持默认 */ }

    if (!authState.enabled) {
        // 鉴权关闭：不显示登录，直接恢复历史
        restoreHistory();
        return;
    }

    try {
        const resp = await apiFetch('/api/auth/me');
        const data = await resp.json();
        if (data.success && data.data.user) {
            authState.user = data.data.user;
            renderUserBar();
            restoreHistory();
            return;
        }
    } catch (e) { /* 未登录 */ }

    authState.user = null;
    openAuthModal('欢迎使用智蜂专家，请先登录');
}

function setAuthMode(mode) {
    authMode = mode;
    document.getElementById('tabLogin').classList.toggle('active', mode === 'login');
    document.getElementById('tabRegister').classList.toggle('active', mode === 'register');
    document.getElementById('fieldNickname').style.display = mode === 'register' ? 'block' : 'none';
    document.getElementById('authSubmit').textContent = mode === 'login' ? '登 录' : '注 册';
    document.getElementById('authError').textContent = '';
    document.getElementById('authTip').textContent = '';
}

function openAuthModal(tip) {
    if (!authState.enabled) return;
    authModalOpen = true;
    const modal = document.getElementById('authModal');
    modal.style.display = 'flex';
    if (tip) document.getElementById('authTip').textContent = tip;
    document.getElementById('authError').textContent = '';
    setTimeout(() => {
        const uname = document.getElementById('authUsername');
        if (uname && !uname.value) uname.focus();
    }, 100);
}

function closeAuthModal() {
    authModalOpen = false;
    document.getElementById('authModal').style.display = 'none';
}

async function submitAuth() {
    const username = document.getElementById('authUsername').value.trim();
    const password = document.getElementById('authPassword').value;
    const nickname = document.getElementById('authNickname').value.trim();
    const errEl = document.getElementById('authError');
    const submitBtn = document.getElementById('authSubmit');
    errEl.textContent = '';
    if (!username || !password) {
        errEl.textContent = '请输入用户名和密码';
        return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = authMode === 'login' ? '登录中…' : '注册中…';
    try {
        const url = authMode === 'login' ? '/api/auth/login' : '/api/auth/register';
        const body = authMode === 'login'
            ? { username, password }
            : { username, password, nickname };
        const resp = await apiFetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await resp.json();
        if (!data.success) {
            errEl.textContent = data.error || '操作失败';
            return;
        }
        authState.user = data.data;
        closeAuthModal();
        renderUserBar();
        showToast(data.data.message || '登录成功');
        // 登录后恢复历史（游客/未登录时暂未加载）
        restoreHistory();
    } catch (e) {
        if (e.isAuth) return;  // apiFetch 已弹登录框
        errEl.textContent = '网络错误，请重试';
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = authMode === 'login' ? '登 录' : '注 册';
    }
}

async function logout() {
    try {
        await apiFetch('/api/auth/logout', { method: 'POST' });
    } catch (e) { /* ignore */ }
    authState.user = null;
    document.getElementById('topbarUser').style.display = 'none';
    openAuthModal('已退出登录');
    // 清空对话历史（避免跨用户串数据）
    chatHistory = [];
    saveHistory();
    document.getElementById('chatMessages').innerHTML = '';
    document.getElementById('chatMode').textContent = '--';
    document.getElementById('sourcesPanel').style.display = 'none';
}

function renderUserBar() {
    if (!authState.user) return;
    const badge = document.getElementById('userBadge');
    const roleLabel = authState.user.role === 'admin' ? '管理员' : '用户';
    badge.textContent = `👤 ${authState.user.nickname || authState.user.username}（${roleLabel}）`;
    document.getElementById('topbarUser').style.display = 'flex';
}

// 问答/知识库的登录拦截
function ensureAuthed() {
    if (authState.enabled && !authState.user) {
        openAuthModal('请先登录后使用该功能');
        return false;
    }
    return true;
}

// ===== 语音输入 =====
let voiceRecognition = null;
let voiceListening = false;

function initVoice() {
    const btn = document.getElementById('btnVoice');
    if (!btn) return;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
        btn.title = '当前浏览器不支持语音输入';
        btn.style.opacity = '0.4';
        return;
    }
    voiceRecognition = new SR();
    voiceRecognition.lang = 'zh-CN';
    voiceRecognition.continuous = false;
    voiceRecognition.interimResults = false;
    voiceRecognition.maxAlternatives = 1;

    voiceRecognition.onresult = (e) => {
        const text = e.results[0][0].transcript;
        const input = document.getElementById('chatInput');
        input.value = text;
        // 识别完成后自动发送
        if (ensureAuthed()) {
            sendQuestion(text);
        } else {
            showToast('已识别：' + text);
        }
    };
    voiceRecognition.onerror = (e) => {
        setVoiceState(false);
        let msg = '语音识别失败';
        if (e.error === 'network') {
            msg = '语音识别服务不可用（需要联网，建议使用 Edge 浏览器）';
        } else if (e.error === 'not-allowed') {
            msg = '麦克风权限被拒绝，请允许使用麦克风';
        } else if (e.error === 'no-speech') {
            msg = '没有听到声音，请再试一次';
        }
        showToast(msg, 'error');
    };
    voiceRecognition.onend = () => setVoiceState(false);
}

function toggleVoiceInput() {
    if (!voiceRecognition) {
        showToast('当前浏览器不支持语音输入，请使用 Edge / Chrome 浏览器', 'error');
        return;
    }
    if (voiceListening) {
        voiceRecognition.stop();
        setVoiceState(false);
        return;
    }
    try {
        voiceRecognition.start();
        setVoiceState(true);
        showToast('🎤 正在聆听，请说话…');
    } catch (e) {
        showToast('语音启动失败，请重试', 'error');
    }
}

function setVoiceState(listening) {
    voiceListening = listening;
    const btn = document.getElementById('btnVoice');
    if (!btn) return;
    btn.classList.toggle('listening', listening);
    btn.title = listening ? '点击停止聆听' : '语音输入提问';
}

// ===== 语音播报 =====
let speakUtterance = null;
let speakBtnRef = null;

function stopSpeak() {
    if (speakUtterance && window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
    speakUtterance = null;
    if (speakBtnRef) {
        speakBtnRef.textContent = '🔊 播报';
        speakBtnRef.classList.remove('speaking');
        speakBtnRef = null;
    }
}

function toggleSpeak(text, btn) {
    // 正在播报同一段 → 停止
    if (speakUtterance && speakBtnRef === btn) {
        stopSpeak();
        return;
    }
    // 切换段落：先停旧的
    if (speakUtterance) stopSpeak();

    if (!window.speechSynthesis) {
        showToast('当前浏览器不支持语音播报', 'error');
        return;
    }
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'zh-CN';
    u.rate = 1.0;
    u.pitch = 1.0;
    // 优先选中文语音
    const voices = window.speechSynthesis.getVoices();
    const zh = voices.find(v => v.lang && v.lang.toLowerCase().startsWith('zh'));
    if (zh) u.voice = zh;

    speakUtterance = u;
    speakBtnRef = btn;
    btn.textContent = '⏹ 停止';
    btn.classList.add('speaking');

    u.onend = () => stopSpeak();
    u.onerror = () => stopSpeak();
    window.speechSynthesis.speak(u);
}

// 在答案消息后追加播报按钮
function attachSpeakButton(msgEl, text) {
    if (!text) return;
    const btn = document.createElement('button');
    btn.className = 'btn-speak';
    btn.textContent = '🔊 播报';
    btn.addEventListener('click', () => toggleSpeak(text, btn));
    const content = msgEl.querySelector('.msg-content');
    if (content) content.appendChild(btn);
}

// ===== 今日天气（温度 + 湿度） =====
let weatherTimer = null;
const WEATHER_REFRESH_MS = 30 * 60 * 1000;   // 每30分钟自动刷新

function initWeather() {
    const el = document.getElementById('statWeather');
    if (!el) return;
    el.addEventListener('click', refreshWeather);
    // 首屏先用默认位置快速显示，避免等待定位 API；定位成功后再静默切换为当地天气
    loadWeather(null);
    getPosition().then(pos => { if (pos) loadWeather(pos); });
    weatherTimer = setInterval(() => loadWeather(null), WEATHER_REFRESH_MS);
}

// 浏览器定位（最多等待 6 秒，失败/超时时返回 null → 后端使用默认位置）
function getPosition(timeout = 5000) {
    return new Promise((resolve) => {
        const timer = setTimeout(() => resolve(null), timeout + 1000);
        if (!navigator.geolocation) {
            clearTimeout(timer);
            return resolve(null);
        }
        navigator.geolocation.getCurrentPosition(
            (pos) => { clearTimeout(timer); resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }); },
            () => { clearTimeout(timer); resolve(null); },
            { timeout, maximumAge: 10 * 60 * 1000 }
        );
    });
}

async function loadWeather(pos) {
    const el = document.getElementById('statWeather');
    if (!el) return;
    const qs = pos ? `?lat=${pos.lat.toFixed(4)}&lon=${pos.lon.toFixed(4)}` : '';
    try {
        const resp = await apiFetch('/api/weather' + qs);
        const data = await resp.json();
        if (data.success && data.data) {
            const w = data.data;
            const temp = w.temperature != null ? Math.round(w.temperature) + '°C' : '--';
            const hum = w.humidity != null ? Math.round(w.humidity) + '%' : '--';
            const loc = w.location ? '📍' + w.location : '🌤️';
            el.textContent = `${loc} ${temp} · 💧${hum}`;
            const t = w.time ? w.time.replace('T', ' ').slice(5, 16) : '';
            el.title = `今日天气 ${t ? t + '更新 · ' : ''}点击刷新`;
            el.classList.remove('weather-error');
        } else {
            setWeatherError(el);
        }
    } catch (e) {
        setWeatherError(el);
    }
}

function setWeatherError(el) {
    el.textContent = '🌤️ 天气 --';
    el.title = '天气获取失败，点击重试';
    el.classList.add('weather-error');
}

function refreshWeather() {
    const el = document.getElementById('statWeather');
    if (!el) return;
    el.textContent = '⏳ 刷新中…';
    loadWeather(null);
    getPosition().then(pos => { if (pos) loadWeather(pos); });
}
