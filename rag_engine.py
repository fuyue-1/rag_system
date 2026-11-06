# -*- coding: utf-8 -*-
"""
RAG 引擎 - MySQL结构化数据 + Chroma向量数据库 + 语义Embedding混合检索
智慧养蜂平台 · 蜜源信息推荐系统

流程：MySQL结构化检索 → Chroma向量语义检索 → 上下文组装 → LLM生成
向量化：默认本地离线 bge-small-zh（fastembed ONNX，512维固定，真增量同步）
        .env 中设置 EMBEDDING_PROVIDER=api 可切回硅基流动 bge-m3（1024维）
向量库特性：持久化到磁盘 / 增量同步 / 支持删除 / 重启无需重建
"""
import os
import json
import time
import re
import numpy as np
import requests
from typing import List, Dict, Optional, Tuple
import pymysql
import chromadb


# 出站 HTTPS 请求的代理策略：
# 若本机开启了系统代理（注册表 ProxyEnable=1），requests 会自动读取并走该代理；
# 代理对 HTTPS 做中间人解密时，Python 侧证书链校验必然失败
# （SSLError: unable to get local issuer certificate），LLM 调用就会静默回退成模板回答。
# 这里对出站 API 显式禁用环境代理（本机直连可通），保证真实调用走到大模型。
OUTBOUND_PROXIES = {"http": None, "https": None}


# ==================== 语义向量化器（本地离线 bge-small-zh） ====================
class LocalBGEEmbedder:
    """本地离线语义向量化（fastembed + BAAI/bge-small-zh-v1.5 ONNX）

    特性：
    - 完全离线：模型下载到 models/ 目录后不再依赖任何外部API/密钥
    - 512维固定 → 新增数据只算新向量，旧向量永不重算（真增量同步）
    - 首次运行自动从 HuggingFace 镜像（hf-mirror.com）下载模型，开箱即用
    - 模型不可用时 is_ready() 返回 False，上层优雅降级（SQL/LLM功能不受影响）
    """

    DIM = 512
    MODEL_NAME = 'BAAI/bge-small-zh-v1.5'

    def __init__(self, cache_dir: str = None):
        # 国内网络：统一走 HuggingFace 镜像，首次下载模型时自动生效
        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        self.cache_dir = cache_dir
        self.model = self.MODEL_NAME
        self._embedder = None
        self._error = None
        self._cache = {}

    def _ensure(self) -> bool:
        """惰性加载模型（首次调用时加载，失败记录原因不抛异常）

        模型已下载 → 强制离线加载（HF_HUB_OFFLINE=1，不联网，部署无忧）
        模型缺失   → 允许联网自动下载（走 hf-mirror.com 镜像，开箱即用）
        """
        if self._embedder is not None:
            return True
        try:
            # 国内网络：HuggingFace 镜像（仅下载时需要）
            os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
            model_dir = os.path.join(self.cache_dir, 'fast-bge-small-zh-v1.5')
            if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, 'model_optimized.onnx')):
                # 模型已就位：强制离线，避免每次启动联网检查导致超时
                os.environ['HF_HUB_OFFLINE'] = '1'
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(
                model_name=self.MODEL_NAME, cache_dir=self.cache_dir
            )
            print(f"[Embedding] 本地模型加载成功: {self.MODEL_NAME}（{self.DIM}维，完全离线）")
            return True
        except Exception as e:
            self._error = e
            print(f"[Embedding] 本地模型加载失败: {e}")
            return False

    def is_ready(self) -> bool:
        return self._ensure()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """批量文档向量化（fastembed 内部自动截断512token、缓存去重）"""
        if not self._ensure():
            raise RuntimeError(f"本地Embedding模型不可用: {self._error}")
        # fastembed 返回 numpy 向量，转为 Python list 供 Chroma 持久化
        return [v.tolist() for v in self._embedder.embed(documents)]

    def embed_query(self, text: str) -> List[float]:
        if not self._ensure():
            raise RuntimeError(f"本地Embedding模型不可用: {self._error}")
        key = hash(text)
        if key not in self._cache:
            self._cache[key] = next(iter(self._embedder.embed([text]))).tolist()
        return self._cache[key]

    def embed_query_safe(self, text: str) -> Optional[List[float]]:
        """查询向量化（容错版）：失败返回None而不是抛异常，调用方据此降级"""
        try:
            return self.embed_query(text)
        except Exception as e:
            print(f"[Embedding] 查询向量化失败（语义检索暂不可用）: {e}")
            return None


# ==================== 语义向量化器（硅基流动 bge-m3 API） ====================
class SiliconFlowEmbedder:
    """调用硅基流动 Embedding API（BAAI/bge-m3）生成稠密语义向量

    特性：
    - 1024维固定 → 新增数据只算新向量，旧向量永不重算（真增量同步）
    - 语义级检索：同义词、不同表述也能命中（如"蜂蜜"↔"蜂糖"）
    - 批量请求 + 失败重试 + 结果缓存
    """

    API_URL = "https://api.siliconflow.cn/v1/embeddings"
    BATCH_SIZE = 16          # 单次请求最大条数
    DIM = 1024               # bge-m3 固定输出维度

    def __init__(self):
        self.api_key = os.environ.get('EMBEDDING_API_KEY', '')
        self.model = os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-m3')
        self._cache = {}     # 文本哈希 -> 向量，避免重复请求

    def is_ready(self) -> bool:
        return bool(self.api_key)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用Embedding API，带3次重试"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(self.API_URL, json=payload, headers=headers,
                                     timeout=30, proxies=OUTBOUND_PROXIES)
                resp.raise_for_status()
                data = resp.json()
                # API按index返回，映射回输入顺序
                emb_map = {item['index']: item['embedding'] for item in data['data']}
                return [emb_map[i] for i in range(len(texts))]
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Embedding API调用失败: {last_err}")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """批量文档向量化（分批请求 + 缓存去重）"""
        vectors, to_query_idx, to_query_texts = [None] * len(documents), [], []
        for i, doc in enumerate(documents):
            key = hash(doc)
            if key in self._cache:
                vectors[i] = self._cache[key]
            else:
                to_query_idx.append(i)
                to_query_texts.append(doc)

        for start in range(0, len(to_query_texts), self.BATCH_SIZE):
            batch = to_query_texts[start:start + self.BATCH_SIZE]
            idxs = to_query_idx[start:start + self.BATCH_SIZE]
            embs = self._call_api(batch)
            for pos, idx in enumerate(idxs):
                vectors[idx] = embs[pos]
                self._cache[hash(batch[pos])] = embs[pos]

        # 缓存过大时清空（避免内存膨胀）
        if len(self._cache) > 5000:
            self._cache = {}
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """查询文本向量化（单条，带缓存）"""
        key = hash(text)
        if key not in self._cache:
            self._cache[key] = self._call_api([text])[0]
        return self._cache[key]

    def embed_query_safe(self, text: str) -> Optional[List[float]]:
        """查询向量化（容错版）：API失败返回None而不是抛异常，调用方据此降级"""
        try:
            return self.embed_query(text)
        except Exception as e:
            print(f"[Embedding] 查询向量化失败（语义检索暂不可用）: {e}")
            return None


# ==================== MySQL数据访问层 ====================
class MySQLNectarStore:
    """MySQL结构化蜜源数据存储"""

    def __init__(self, host=None, port=None, user=None,
                 password=None, database=None):
        self.db_config = {
            'host': host or os.environ.get('MYSQL_HOST', '127.0.0.1'),
            'port': int(port or os.environ.get('MYSQL_PORT', 3306)),
            'user': user or os.environ.get('MYSQL_USER', 'root'),
            'password': password or os.environ.get('MYSQL_PASSWORD', '123456'),
            'database': database or os.environ.get('MYSQL_DATABASE', 'bee_rag'),
            'charset': 'utf8mb4'
        }

    def _get_conn(self):
        return pymysql.connect(**self.db_config)

    def get_all_nectar_sources(self) -> List[Dict]:
        """获取全部蜜源记录"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute("SELECT * FROM nectar_sources ORDER BY id")
            rows = cursor.fetchall()
            cursor.close()
            return rows
        finally:
            conn.close()

    def get_all_knowledge(self) -> List[Dict]:
        """获取全部知识库文本"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute("SELECT * FROM bee_knowledge ORDER BY id")
            rows = cursor.fetchall()
            cursor.close()
            return rows
        finally:
            conn.close()

    def get_filter_vocab(self) -> Dict[str, List[str]]:
        """检索时用于从问题文本抽取结构化硬条件的词表：去重省份、去重植物名"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(
                "SELECT DISTINCT province FROM nectar_sources "
                "WHERE province IS NOT NULL AND province <> ''")
            provinces = [r['province'] for r in cursor.fetchall()]
            cursor.execute(
                "SELECT DISTINCT plant_name FROM nectar_sources "
                "WHERE plant_name IS NOT NULL AND plant_name <> ''")
            plants = [r['plant_name'] for r in cursor.fetchall()]
            cursor.close()
            return {"provinces": provinces, "plants": plants}
        finally:
            conn.close()

    def filter_by_location_and_time(self, province: str = None,
                                     month: int = None,
                                     plant_name: str = None) -> List[Dict]:
        """按地理位置和花期月份结构化筛选"""
        sql = "SELECT * FROM nectar_sources WHERE 1=1"
        params = []

        if province:
            sql += " AND province = %s"
            params.append(province)
        if plant_name:
            sql += " AND plant_name = %s"
            params.append(plant_name)
        if month:
            # 花期包含该月份
            sql += """ AND (
                (flowering_start <= flowering_end AND %s BETWEEN flowering_start AND flowering_end)
                OR
                (flowering_start > flowering_end AND (%s >= flowering_start OR %s <= flowering_end))
            )"""
            params.extend([month, month, month])

        sql += " ORDER BY honey_yield DESC"
        conn = self._get_conn()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            cursor.close()
            return rows
        finally:
            conn.close()

    def get_by_id(self, nectar_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute("SELECT * FROM nectar_sources WHERE id = %s", (nectar_id,))
            row = cursor.fetchone()
            cursor.close()
            return row
        finally:
            conn.close()

    def search_knowledge(self, keyword: str = None, category: str = None) -> List[Dict]:
        sql = "SELECT * FROM bee_knowledge WHERE 1=1"
        params = []
        if category:
            sql += " AND category = %s"
            params.append(category)
        if keyword:
            sql += " AND (title LIKE %s OR content LIKE %s OR tags LIKE %s)"
            kw = f"%{keyword}%"
            params.extend([kw, kw, kw])
        sql += " ORDER BY id"
        conn = self._get_conn()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            cursor.close()
            return rows
        finally:
            conn.close()

    def get_stats(self) -> Dict:
        conn = self._get_conn()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute("SELECT COUNT(*) as total FROM nectar_sources")
            total = cursor.fetchone()['total']
            cursor.execute("SELECT COUNT(DISTINCT plant_name) as cnt FROM nectar_sources")
            plant_cnt = cursor.fetchone()['cnt']
            cursor.execute("SELECT COUNT(DISTINCT province) as cnt FROM nectar_sources")
            prov_cnt = cursor.fetchone()['cnt']
            cursor.execute("SELECT COUNT(*) as cnt FROM bee_knowledge")
            know_cnt = cursor.fetchone()['cnt']
            cursor.execute("SELECT plant_name, COUNT(*) as cnt FROM nectar_sources GROUP BY plant_name ORDER BY cnt DESC")
            plant_dist = cursor.fetchall()
            cursor.execute("SELECT province, COUNT(*) as cnt FROM nectar_sources GROUP BY province ORDER BY cnt DESC")
            prov_dist = cursor.fetchall()
            cursor.close()
            return {
                "nectar_count": total,
                "plant_count": plant_cnt,
                "province_count": prov_cnt,
                "knowledge_count": know_cnt,
                "plant_distribution": plant_dist,
                "province_distribution": prov_dist
            }
        finally:
            conn.close()


# ==================== Chroma向量数据库索引 ====================
class ChromaVectorIndex:
    """基于Chroma的持久化向量数据库索引

    特性：
    - 向量持久化到磁盘（vector_db/目录），重启服务器无需重建
    - 增量同步：MySQL新增数据只向量化新增部分，自动入库
    - 支持按ID删除，与MySQL保持一致
    - cosine距离检索，支持按type过滤
    """

    COLLECTION_NAME = 'bee_rag'

    def __init__(self, persist_dir: str = None):
        if persist_dir is None:
            persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vector_db')
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir

        # 持久化客户端：数据自动落盘
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={'hnsw:space': 'cosine'}   # 余弦距离
        )
        # 语义Embedding：默认本地离线bge-small-zh；.env设置EMBEDDING_PROVIDER=api可切回硅基流动
        provider = os.environ.get('EMBEDDING_PROVIDER', 'local').strip().lower()
        if provider == 'api':
            self.embedder = SiliconFlowEmbedder()
            self.embedding_provider = 'api'
        else:
            self.embedder = LocalBGEEmbedder()
            self.embedding_provider = 'local'
        if not self.embedder.is_ready():
            print(f"[RAG] 警告: Embedding不可用（provider={self.embedding_provider}），"
                  f"语义检索将降级，SQL筛选/LLM问答功能不受影响")
        self._cache = {}
        self.is_built = False

    # ---------- 文本组装 ----------
    def _nectar_text(self, ns: Dict) -> str:
        return (f"{ns['plant_name']} {ns['province']} {ns['city']} {ns['location_name']} "
                f"{ns['description']} 气候类型{ns['climate_type']} 年均气温{ns['annual_avg_temp']}℃ "
                f"年降水量{ns['annual_precipitation']}mm 日照{ns['sunshine_hours']}小时 "
                f"开花{ns['flowering_start']}月到{ns['flowering_end']}月 "
                f"采蜜{ns['best_honey_start']}月到{ns['best_honey_end']}月")

    def _knowledge_text(self, k: Dict) -> str:
        return f"{k['title']} {k['category']} {k['content']} {k.get('tags') or ''}"

    def _nectar_meta(self, ns: Dict) -> Dict:
        """Chroma metadata不允许None，统一清洗"""
        def clean(v):
            return '' if v is None else v
        return {
            "id": ns['id'],
            "type": "nectar_source",
            "plant_name": ns['plant_name'],
            "province": ns['province'],
            "city": ns['city'],
            "location_name": clean(ns['location_name']),
            "latitude": float(ns['latitude']),
            "longitude": float(ns['longitude']),
            "flowering_start": ns['flowering_start'],
            "flowering_end": ns['flowering_end'],
            "best_honey_start": ns['best_honey_start'],
            "best_honey_end": ns['best_honey_end'],
            "honey_yield": float(ns['honey_yield']) if ns['honey_yield'] else 0.0,
            "nectar_sugar_content": float(ns['nectar_sugar_content']) if ns['nectar_sugar_content'] else 0.0,
            "description": ns['description'] or '',
            "climate_type": clean(ns['climate_type']),
            "annual_avg_temp": float(ns['annual_avg_temp']) if ns['annual_avg_temp'] else 0.0,
            "annual_precipitation": ns['annual_precipitation'] or 0,
            "sunshine_hours": ns['sunshine_hours'] or 0,
            "planting_area": ns['planting_area'] or 0,
        }

    def _knowledge_meta(self, k: Dict) -> Dict:
        return {
            "id": k['id'],
            "type": "knowledge",
            "title": k['title'],
            "category": k['category'],
            "tags": k.get('tags') or '',
        }

    # ---------- 从MySQL加载全部文档 ----------
    def _load_docs_from_mysql(self, mysql_store: MySQLNectarStore):
        docs, metas, ids = [], [], []
        nectar_sources = mysql_store.get_all_nectar_sources()
        for ns in nectar_sources:
            docs.append(self._nectar_text(ns))
            metas.append(self._nectar_meta(ns))
            ids.append(f"nectar_{ns['id']}")
        knowledge = mysql_store.get_all_knowledge()
        for k in knowledge:
            docs.append(self._knowledge_text(k))
            metas.append(self._knowledge_meta(k))
            ids.append(f"knowledge_{k['id']}")
        return docs, metas, ids

    def _existing_ids(self) -> set:
        if self.collection.count() == 0:
            return set()
        return set(self.collection.get()['ids'])

    def _reset_collection(self):
        """清空并重建collection（向量维度变化时必须，否则维度不匹配）"""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={'hnsw:space': 'cosine'})

    def _existing_dim(self) -> Optional[int]:
        """读取Chroma collection中实际向量维度（唯一可靠的判断依据）"""
        if self.collection.count() == 0:
            return None
        sample = self.collection.get(limit=1, include=['embeddings'])
        embs = sample.get('embeddings')
        if embs is not None and len(embs) > 0:
            return len(embs[0])
        return None

    def _full_upsert(self, docs, metas, ids):
        """全量向量化并写入向量库（维度变化时先重置collection）"""
        existing_dim = self._existing_dim()

        vectors = self.embedder.embed_documents(docs)
        new_dim = len(vectors[0]) if vectors else 0

        # 维度变化（换embedding模型）→ 必须重置collection（Chroma维度固定）
        if existing_dim is not None and new_dim != existing_dim:
            self._reset_collection()

        self._upsert(ids, docs, metas, vectors)
        self._cache = {}

    # ---------- 构建与同步 ----------
    def build(self, mysql_store: MySQLNectarStore) -> int:
        """启动时构建索引：优先从磁盘加载（无需重新向量化），增量补齐新数据

        Embedding 不可用时：已落盘向量仍可加载（若维度匹配），仅新增数据暂缓入库，
        不阻塞服务启动。
        """
        docs, metas, ids = self._load_docs_from_mysql(mysql_store)
        existing = self._existing_ids()
        dim_ok = self._existing_dim() == self.embedder.DIM

        # 磁盘状态与MySQL一致且维度匹配 → 直接加载，跳过重建
        if existing == set(ids) and dim_ok:
            self.is_built = True
            return -len(ids)   # 负数表示"磁盘命中，0计算"

        # Embedding 不可用且磁盘没有可用向量 → 无法构建，服务照常启动（语义检索返回空）
        if not self.embedder.is_ready() and not existing:
            print("[RAG] 警告: 向量库为空且Embedding不可用，语义检索暂不可用，"
                  "SQL筛选/LLM问答功能正常")
            self.is_built = True
            return 0

        # Embedding不可用但磁盘已有可用向量 → 降级加载已有部分，跳过增量
        if existing and dim_ok:
            if self._api_probe_failed():
                self.is_built = True
                print(f"[RAG] Embedding不可用：已加载磁盘{len(existing)}条向量，"
                      f"MySQL新增{len(set(ids)-existing)}条待模型恢复后同步")
                return -len(existing)
            try:
                result = self._apply_changes(docs, metas, ids)
                self.is_built = True
                return result["added"] if existing else len(ids)
            except RuntimeError as e:
                # Embedding在增量过程中失败：保留已入库部分，服务照常启动
                print(f"[RAG] 向量增量同步失败（{e}），已加载磁盘{len(existing)}条向量")
                self.is_built = True
                return -len(existing)

        # 空库或维度不匹配：必须调Embedding，失败则抛出（此时向量库完全不可用）
        result = self._apply_changes(docs, metas, ids)
        self.is_built = True
        return result["added"] if existing else len(ids)

    def _api_probe_failed(self) -> bool:
        """轻量探测Embedding API可用性（仅启动时调用一次）"""
        try:
            self.embedder.embed_query('探测')
            return False
        except Exception:
            return True

    def sync(self, mysql_store: MySQLNectarStore) -> Dict:
        """增量同步：只向量化新增数据（真增量，旧向量永不重算）"""
        docs, metas, ids = self._load_docs_from_mysql(mysql_store)
        return self._apply_changes(docs, metas, ids)

    def _apply_changes(self, docs, metas, ids) -> Dict:
        """将MySQL当前状态同步进向量库（增量优先，维度不匹配自动回退全量）"""
        existing = self._existing_ids()
        removed_ids = list(existing - set(ids))
        if removed_ids:
            self.collection.delete(ids=removed_ids)
            existing = existing - set(removed_ids)

        new_items = [(d, m, i) for d, m, i in zip(docs, metas, ids) if i not in existing]

        # 库内维度与当前模型不一致（如换过embedding模型）→ 全量重建
        if existing and self._existing_dim() != self.embedder.DIM:
            self._full_upsert(docs, metas, ids)
            return {"added": len(ids), "removed": len(removed_ids), "total": len(ids)}

        if not new_items:
            return {"added": 0, "removed": len(removed_ids), "total": len(ids)}

        if existing:
            # 库中已有同维度数据 → 只向量化新增部分（真增量，旧向量不动）
            new_docs = [x[0] for x in new_items]
            new_metas = [x[1] for x in new_items]
            new_ids = [x[2] for x in new_items]
            vectors = self.embedder.embed_documents(new_docs)
            existing_dim = self._existing_dim()
            if existing_dim is not None and len(vectors[0]) != existing_dim:
                # 模型切换导致维度变化 → 回退全量重建
                self._full_upsert(docs, metas, ids)
            else:
                self._upsert(new_ids, new_docs, new_metas, vectors)
        else:
            # 空库 → 全量
            self._full_upsert(docs, metas, ids)

        return {"added": len(new_items), "removed": len(removed_ids), "total": len(ids)}

    def rebuild(self, mysql_store: MySQLNectarStore) -> int:
        """全量重建：清空向量库后重新构建"""
        docs, metas, ids = self._load_docs_from_mysql(mysql_store)
        self._reset_collection()
        vectors = self.embedder.embed_documents(docs)
        self._upsert(ids, docs, metas, vectors)
        self._cache = {}
        self.is_built = True
        return len(ids)

    def _upsert(self, ids, docs, metas, vectors):
        """分批upsert（Chroma单批限制）"""
        batch = 50
        for i in range(0, len(ids), batch):
            self.collection.upsert(
                ids=ids[i:i+batch],
                documents=docs[i:i+batch],
                metadatas=metas[i:i+batch],
                embeddings=vectors[i:i+batch],
            )

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 5, filter_type: str = None) -> List[Dict]:
        """语义检索：Chroma余弦相似度Top-K；Embedding不可用时返回空（由上层降级）"""
        if not self.is_built or self.collection.count() == 0:
            return []
        query_vec = self.embedder.embed_query_safe(query)
        if not query_vec:
            return []

        where = {"type": filter_type} if filter_type else None
        n_results = min(top_k, self.collection.count())
        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            where=where,
            include=['metadatas', 'documents', 'distances']
        )

        results = []
        for doc, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
            score = 1.0 - dist   # cosine距离 → 相似度
            if score <= 0.01:
                continue
            item = dict(meta)
            item['document'] = doc
            item['id'] = int(meta.get('id', 0)) if meta.get('id') else None
            item['score'] = round(score, 4)
            results.append(item)
        return results

    def _vector_rank_within(self, query: str, id_list: List[str], top_k: int) -> List[Dict]:
        """在给定 id 候选集内做向量余弦排序（SQL 预过滤后的精排步骤）

        直接用 collection.get(ids=...) 取候选集向量，在 Python 内算余弦相似度排序，
        绕开 Chroma 对 id 字段做 where 过滤的不稳定性，保证"向量排序严格发生在 SQL 候选集内"。
        Embedding 不可用时返回空（由上层回退 SQL 排序）。
        """
        if not id_list:
            return []
        got = self.collection.get(ids=id_list, include=['embeddings', 'metadatas', 'documents'])
        embs = got.get('embeddings')
        if embs is None or len(embs) == 0:
            return []
        q = self.embedder.embed_query_safe(query)
        if q is None:
            return []
        embs = np.asarray(embs, dtype=float)
        q = np.asarray(q, dtype=float)
        sims = (embs @ q) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q) + 1e-9)
        order = sims.argsort()[::-1][:top_k]
        ranked = []
        for i in order:
            meta = got['metadatas'][i]
            item = dict(meta)
            item['document'] = got['documents'][i] if got['documents'] else ''
            item['id'] = int(meta.get('id', 0)) if meta.get('id') else None
            item['score'] = round(float(sims[i]), 4)
            ranked.append(item)
        return ranked

    def search_nectar_sources(self, query: str, top_k: int = 5,
                              province: str = None, month: int = None,
                              plant_name: str = None) -> List[Dict]:
        """混合检索（真正的预过滤）：SQL结构化硬筛选 → 候选集内向量语义排序

        流程：
          1. 若给定任一结构化条件，先由 MySQL 硬过滤得到候选蜜源 id 集；
          2. 向量排序仅在「该候选集」内进行（_vector_rank_within），输出 Top-K；
          3. 若候选集内无语义命中，回退为 SQL 结果（按产蜜量排序，score=0）；
          4. 若 SQL 无命中，退化为纯向量检索以扩大召回面。
        """
        if not self.is_built:
            return []

        if province or month or plant_name:
            mysql_store = MySQLNectarStore()
            filtered = mysql_store.filter_by_location_and_time(
                province=province, month=month, plant_name=plant_name)
            if filtered:
                # 第一步：SQL 硬过滤得到候选集 id 列表（Chroma 文档 id = nectar_<mysql_id>）
                id_list = [f"nectar_{ns['id']}" for ns in filtered]
                # 第二步：向量语义排序严格在候选集内进行
                results = self._vector_rank_within(query, id_list, top_k)
                if not results:
                    # 语义相关性低但 SQL 硬性命中 → 直接返回 SQL 结果（已按产蜜量降序）
                    for ns in filtered[:top_k]:
                        item = self._nectar_meta(ns)
                        item['score'] = 0.0
                        results.append(item)
                return results[:top_k]
            # SQL 无命中 → 退化为纯向量检索
            return self.search(query, top_k=top_k, filter_type='nectar_source')
        else:
            return self.search(query, top_k=top_k, filter_type='nectar_source')

    def count(self) -> int:
        return self.collection.count()

    def stats(self) -> Dict:
        existing = self._existing_ids()
        nectar = sum(1 for i in existing if i.startswith('nectar_'))
        knowledge = sum(1 for i in existing if i.startswith('knowledge_'))
        return {
            "total": len(existing),
            "nectar_vectors": nectar,
            "knowledge_vectors": knowledge,
            "persist_dir": self.persist_dir,
            "embedding": {
                "mode": "semantic",
                "provider": self.embedding_provider,
                "model": self.embedder.model,
                "dim": self.embedder.DIM,
            },
        }


# ==================== 答案生成器 ====================
class AnswerGenerator:
    """答案生成：调用LLM API或本地模板回退"""

    def __init__(self):
        self.api_key = os.environ.get('LLM_API_KEY', '')
        self.api_url = os.environ.get('LLM_API_URL', 'https://api.deepseek.com/v1/chat/completions')
        self.model = os.environ.get('LLM_MODEL', 'deepseek-chat')

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def generate(self, question: str, context_docs: List[Dict],
                 question_type: str = 'general',
                 history: List[Dict] = None) -> Tuple[str, str, dict]:
        """生成回答，返回(answer, method, token_usage)；history为多轮对话历史"""
        if self.is_enabled():
            answer, token_usage = self._call_llm(question, context_docs, question_type, history)
            if answer:
                return answer, 'llm', token_usage
        # 模板回退（无 LLM 调用，token 记为 0）
        return self._template_answer(question, context_docs), 'template', {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def generate_stream(self, question: str, context_docs: List[Dict],
                        question_type: str = 'general',
                        history: List[Dict] = None):
        """流式生成：逐块yield文本片段，最后yield ('__method__', method)标记生成方式"""
        if self.is_enabled():
            got_any = False
            try:
                for chunk in self._call_llm_stream(question, context_docs, question_type, history):
                    got_any = True
                    yield chunk
                if got_any:
                    yield ('__method__', 'llm')
                    return
            except Exception as e:
                print(f"[LLM流式调用失败] {e}")
        answer, method, _ = self.generate(question, context_docs, question_type, history)
        yield answer
        yield ('__method__', method)

    # ---------- 查询重写（Query Rewriting）----------
    def rewrite_query(self, question: str, history: List[Dict] = None) -> str:
        """将含指代的多轮追问改写为可独立理解的检索查询

        作用位置：召回层最前端——在问题被向量化之前，先把"那里产蜜量高吗"
        这类依赖上下文的问法补全为"陕西的蜜源地预计产蜜量是多少"。

        设计约束：
        - 仅当存在历史对话且开启了 LLM 时才触发，首轮提问零开销
        - 任何异常（无 Key、超时、返回异常）一律回退为原问题，不影响主流程
        """
        if not history or not self.is_enabled():
            return question

        system_prompt = (
            "你是智慧养蜂平台的查询理解模块。任务：结合历史对话，把用户最新的追问"
            "改写为一条**可独立理解**的检索查询语句。\n"
            "规则：\n"
            "1. 只补全指代（那里、这种蜜、它、那儿等）与省略的主语，不要臆造新信息\n"
            "2. 输出纯文本一条查询，不要解释、不要标点结尾、不要加引号\n"
            "3. 长度控制在 30 字以内\n"
            "4. 若当前问题本身已完整独立，原样输出即可"
        )
        messages = [{"role": "system", "content": system_prompt}]
        for h in history[-6:]:
            role, content = h.get('role'), (h.get('content') or '').strip()
            if role in ('user', 'assistant') and content:
                messages.append({"role": role, "content": content[:300]})
        messages.append({"role": "user", "content": f"请改写这条追问：{question}"})

        try:
            resp = requests.post(
                self.api_url,
                json={"model": self.model, "messages": messages,
                      "temperature": 0.0, "max_tokens": 80},
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
                timeout=8,                      # 检索前置步骤，超时从严
                proxies=OUTBOUND_PROXIES,
            )
            resp.raise_for_status()
            rewritten = (resp.json()['choices'][0]['message']['content'] or '').strip()
            rewritten = rewritten.strip('"\'` \n')
            # 安全兜底：改写结果异常（空/过长）时仍用原问题
            if not rewritten or len(rewritten) > 120:
                return question
            if rewritten != question:
                print(f"[查询重写] 「{question}」→「{rewritten}」")
            return rewritten
        except Exception as e:
            print(f"[查询重写失败，回退原问题] {e}")
            return question

    def _build_messages(self, question: str, context_text: str,
                        history: List[Dict] = None) -> List[Dict]:
        """组装多轮对话messages：system + 历史对话 + 带检索资料的本轮问题"""
        system_prompt = (
            "你是智慧养蜂平台的AI助手「智蜂专家」，专门为蜂农提供蜜源推荐和养蜂技术咨询。"
            "请根据检索到的蜜源数据或知识库内容回答用户问题。"
            "回答要准确、实用，如果有具体数据（温度、花期、产蜜量等）请引用。"
            "如果涉及蜜源推荐，请按推荐优先级排序，并说明推荐理由。"
            "如果检索资料与问题无关，请如实说明，再基于养蜂常识谨慎补充。"
            "请结合历史对话理解用户的指代（如'那里'、'这种蜜'、'它'）。"
        )
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for h in history[-8:]:          # 最近8条消息，控制上下文长度
                role = h.get('role')
                content = (h.get('content') or '').strip()
                if role in ('user', 'assistant') and content:
                    messages.append({"role": role, "content": content[:2000]})
        user_prompt = f"检索到的参考资料：\n{context_text}\n\n用户问题：{question}\n\n请基于以上资料回答："
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _call_llm(self, question: str, context_docs: List[Dict],
                  question_type: str, history: List[Dict] = None) -> Optional[str]:
        """调用DeepSeek/OpenAI API（非流式）"""
        context_text = self._build_context(context_docs, question_type)
        messages = self._build_messages(question, context_text, history)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500
        }

        try:
            resp = requests.post(self.api_url, json=payload, headers=headers,
                                 timeout=60, proxies=OUTBOUND_PROXIES)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get('usage') or {}
            content = data['choices'][0]['message']['content']
            # 透传 token 用量（供评测脚本上报 LangSmith）
            token_usage = {
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
            return content, token_usage
        except Exception as e:
            print(f"[LLM调用失败] {e}")
            return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _call_llm_stream(self, question: str, context_docs: List[Dict],
                         question_type: str, history: List[Dict] = None):
        """流式调用LLM：逐块yield生成内容（SSE协议解析）"""
        context_text = self._build_context(context_docs, question_type)
        messages = self._build_messages(question, context_text, history)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500,
            "stream": True,
            "stream_options": {"include_usage": True},   # 让 DeepSeek 在末尾推送 usage 块
        }

        resp = requests.post(self.api_url, json=payload, headers=headers,
                             stream=True, timeout=(10, 120), proxies=OUTBOUND_PROXIES)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith('data:'):
                continue
            data_str = line[5:].strip()
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                # 末尾 usage 块（开启 include_usage 后推送）：单独透传给上游用于记录 token
                usage = chunk.get('usage')
                if usage:
                    yield ('__usage__', {
                        "input_tokens": int(usage.get('prompt_tokens') or 0),
                        "output_tokens": int(usage.get('completion_tokens') or 0),
                        "total_tokens": int(usage.get('total_tokens') or 0),
                    })
                    continue
                choices = chunk.get('choices') or []
                if not choices:
                    continue
                delta = choices[0].get('delta', {})
                content = delta.get('content')
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def _build_context(self, docs: List[Dict], question_type: str) -> str:
        """将检索结果组装为LLM上下文"""
        parts = []
        for i, doc in enumerate(docs, 1):
            if doc.get('type') == 'nectar_source':
                parts.append(
                    f"[{i}] 蜜源植物：{doc['plant_name']}\n"
                    f"  位置：{doc['province']} {doc['city']} {doc['location_name']}\n"
                    f"  经纬度：({doc['latitude']}, {doc['longitude']})\n"
                    f"  气候：{doc.get('climate_type', 'N/A')}，年均气温{doc.get('annual_avg_temp', 'N/A')}℃，"
                    f"降水{doc.get('annual_precipitation', 'N/A')}mm，日照{doc.get('sunshine_hours', 'N/A')}小时\n"
                    f"  花期：{doc.get('flowering_start', '?')}月-{doc.get('flowering_end', '?')}月，"
                    f"最佳采蜜期：{doc.get('best_honey_start', '?')}月-{doc.get('best_honey_end', '?')}月\n"
                    f"  预计产蜜量：{doc.get('honey_yield', 'N/A')}kg/群，含糖量：{doc.get('nectar_sugar_content', 'N/A')}%\n"
                    f"  描述：{doc.get('description', 'N/A')}\n"
                    f"  相似度：{doc.get('score', 0)}"
                )
            else:
                content = doc.get('content') or doc.get('document', 'N/A')
                parts.append(
                    f"[{i}] 知识库：{doc.get('title', 'N/A')}（{doc.get('category', 'N/A')}）\n"
                    f"  内容：{content[:500]}\n"
                    f"  相似度：{doc.get('score', 0)}"
                )
        return "\n\n".join(parts)

    def _template_answer(self, question: str, docs: List[Dict]) -> str:
        """模板回退（LLM不可用时）"""
        if not docs:
            return "抱歉，未找到与您问题相关的信息。请尝试换个关键词提问，或检查数据库中是否有对应数据。"

        parts = [f"关于「{question}」，为您检索到以下相关内容：\n"]
        for i, doc in enumerate(docs, 1):
            if doc.get('type') == 'nectar_source':
                parts.append(
                    f"{i}. **{doc['plant_name']}** — {doc['province']} {doc['city']}\n"
                    f"   花期：{doc.get('flowering_start', '?')}-{doc.get('flowering_end', '?')}月 | "
                    f"采蜜期：{doc.get('best_honey_start', '?')}-{doc.get('best_honey_end', '?')}月\n"
                    f"   产蜜量：{doc.get('honey_yield', 'N/A')}kg/群 | 匹配度：{doc.get('score', 0)}\n"
                    f"   {doc.get('description', '')[:120]}..."
                )
            else:
                parts.append(
                    f"{i}. **{doc.get('title', 'N/A')}**（{doc.get('category', 'N/A')}）\n"
                    f"   {doc.get('content', '')[:200]}..."
                )
        return "\n\n".join(parts)


# ==================== RAG 主引擎 ====================
class RAGEngine:
    """RAG主引擎：MySQL结构化检索 + Chroma向量数据库 + LLM生成"""

    def __init__(self):
        self.mysql_store = MySQLNectarStore()
        self.vector_index = ChromaVectorIndex()
        self.generator = AnswerGenerator()
        self.is_initialized = False

    def initialize(self):
        """初始化：优先从磁盘加载向量库（未变化时不重建），增量补齐MySQL新数据"""
        stats = self.mysql_store.get_stats()
        print(f"[RAG] MySQL连接成功：{stats['nectar_count']}条蜜源记录，{stats['knowledge_count']}篇知识库")

        doc_count = self.vector_index.build(self.mysql_store)
        provider = self.vector_index.embedding_provider
        model = self.vector_index.embedder.model
        dim = self.vector_index.embedder.DIM
        print(f"[RAG] 向量化方案：{model}（{provider}模式，{dim}维）")
        if doc_count >= 0:
            print(f"[RAG] Chroma向量库构建完成：{doc_count}篇文档（全量向量化并持久化到磁盘）")
        else:
            print(f"[RAG] Chroma向量库磁盘命中：{-doc_count}篇文档（跳过重建，直接加载）")

        # LLM状态
        llm_status = "已启用" if self.generator.is_enabled() else "未启用(仅模板模式)"
        print(f"[RAG] LLM状态：{llm_status}")

        self.is_initialized = True

    def sync_vectors(self) -> Dict:
        """增量同步：MySQL新增/删除的数据同步到Chroma向量库（无需重启服务器）"""
        result = self.vector_index.sync(self.mysql_store)
        print(f"[RAG] 向量库增量同步：新增{result['added']}条，删除{result['removed']}条，总计{result['total']}条")
        return result

    def rebuild_vectors(self) -> int:
        """全量重建向量库"""
        count = self.vector_index.rebuild(self.mysql_store)
        print(f"[RAG] 向量库全量重建完成：{count}条")
        return count

    def vector_stats(self) -> Dict:
        return self.vector_index.stats()

    def query(self, question: str, top_k: int = 5,
              province: str = None, month: int = None,
              history: List[Dict] = None) -> Dict:
        """智能问答检索

        Args:
            question: 用户问题
            top_k: 返回结果数量
            province: 省份筛选（可选）
            month: 月份筛选（可选，用于花期匹配）
            history: 多轮对话历史（可选，[{role, content}]）

        Returns:
            {answer, method, sources, context_docs}
        """
        if not self.is_initialized:
            self.initialize()

        sources, _ = self.retrieve(question, top_k=top_k, province=province,
                                   month=month, history=history)

        # 生成回答
        answer, method, token_usage = self.generator.generate(question, sources, history=history)

        return {
            "answer": answer,
            "method": method,
            "sources": sources,
            "total_sources": len(sources),
            "token_usage": token_usage,
        }

    def _extract_structured_constraints(self, text: str) -> Dict:
        """从问题文本抽取结构化硬条件（省份/植物名/月份），供 SQL 预过滤使用

        仅做「精确命中已知词表」的匹配，避免误召回；无任何匹配时三项均为 None。
        词表来自数据库去重结果，首次调用后缓存到实例。
        """
        if not hasattr(self, '_filter_vocab'):
            self._filter_vocab = self.mysql_store.get_filter_vocab()
        provinces = self._filter_vocab.get('provinces', [])
        plants = self._filter_vocab.get('plants', [])

        out = {'province': None, 'plant_name': None, 'month': None}
        # 省份：命中已知省份名即采用（取首个）
        for p in provinces:
            if p and p in text:
                out['province'] = p
                break
        # 植物名：命中已知蜜源植物即采用（取首个）
        for pl in plants:
            if pl and pl in text:
                out['plant_name'] = pl
                break
        # 月份：阿拉伯数字 "N月" 模式
        m = re.search(r'(\d{1,2})\s*月', text)
        if m:
            mo = int(m.group(1))
            if 1 <= mo <= 12:
                out['month'] = mo
        return out

    def retrieve(self, question: str, top_k: int = 5,
                 province: str = None, month: int = None,
                 history: List[Dict] = None,
                 rewrite: bool = True) -> tuple:
        """只做检索不生成（供流式端点先推来源、再流式生成）

        返回 (sources, rewritten_question)：
          - rewritten_question 为实际用于检索的查询文本（未重写时等于原问题）
          - 调用方可用它做前端展示或效果追踪
        """
        if not self.is_initialized:
            self.initialize()

        # ===== 召回层第一步：查询重写（多轮指代消解）=====
        # 必须在意图判断与向量化之前完成，否则"那里"这类代词会污染查询向量
        search_query = question
        if rewrite:
            search_query = self.generator.rewrite_query(question, history)

        # 判断问题类型：蜜源推荐 vs 养蜂知识
        nectar_keywords = ['蜜源', '蜜源地', '采蜜', '花期', '开花', '推荐', '哪里有',
                          '哪个省', '什么时候采', '转场', '放蜂', '荆条', '椴树',
                          '油菜', '荔枝', '龙眼', '刺槐', '枣花', '荆条', '向日葵',
                          '棉花', '紫云英', '枇杷', '乌桕', '柑橘', '莽麦']
        is_nectar_query = any(kw in search_query for kw in nectar_keywords)

        # ===== 召回层：结构化硬条件抽取（SQL 预过滤用）=====
        # 从（重写后的）问题文本精确匹配省份/植物名/月份；与 UI 显式传入的筛选合并，
        # 显式传入优先。无任何匹配时三项为 None，退化为纯向量检索。
        constraints = self._extract_structured_constraints(search_query)
        eff_province = province or constraints['province']
        eff_month = month or constraints['month']
        eff_plant = constraints['plant_name']

        if is_nectar_query or eff_province or eff_month or eff_plant:
            # 蜜源推荐型检索（SQL 硬过滤 → 候选集内向量排序）
            sources = self.vector_index.search_nectar_sources(
                search_query, top_k=top_k,
                province=eff_province, month=eff_month, plant_name=eff_plant
            )
            # 如果结果不足，补充知识库检索
            if len(sources) < top_k:
                knowledge_results = self.vector_index.search(
                    search_query, top_k=2, filter_type='knowledge'
                )
                sources.extend(knowledge_results)
        else:
            # 养蜂知识型检索
            sources = self.vector_index.search(search_query, top_k=top_k)
            # 如果结果不足，补充蜜源检索
            if len(sources) < top_k:
                nectar_results = self.vector_index.search_nectar_sources(
                    search_query, top_k=2
                )
                sources.extend(nectar_results)
        return sources, search_query

    def recommend_nectar(self, province: str = None, month: int = None,
                         plant_name: str = None, top_k: int = 10) -> Dict:
        """蜜源推荐：基于结构化条件筛选 + 向量排序

        Args:
            province: 省份
            month: 月份（花期匹配）
            plant_name: 蜜源植物名称
            top_k: 返回数量

        Returns:
            {recommendations, total, filters}
        """
        if not self.is_initialized:
            self.initialize()

        # SQL结构化筛选
        filtered = self.mysql_store.filter_by_location_and_time(
            province=province, month=month, plant_name=plant_name
        )

        # 如果有筛选结果，按产蜜量和含糖量排序
        recommendations = []
        for ns in filtered[:top_k]:
            recommendations.append({
                "id": ns['id'],
                "plant_name": ns['plant_name'],
                "province": ns['province'],
                "city": ns['city'],
                "location_name": ns['location_name'],
                "latitude": float(ns['latitude']),
                "longitude": float(ns['longitude']),
                "flowering_start": ns['flowering_start'],
                "flowering_end": ns['flowering_end'],
                "best_honey_start": ns['best_honey_start'],
                "best_honey_end": ns['best_honey_end'],
                "honey_yield": float(ns['honey_yield']) if ns['honey_yield'] else None,
                "nectar_sugar_content": float(ns['nectar_sugar_content']) if ns['nectar_sugar_content'] else None,
                "climate_type": ns['climate_type'],
                "annual_avg_temp": float(ns['annual_avg_temp']) if ns['annual_avg_temp'] else None,
                "description": ns['description'],
                "planting_area": ns['planting_area']
            })

        return {
            "recommendations": recommendations,
            "total": len(filtered),
            "filters": {
                "province": province,
                "month": month,
                "plant_name": plant_name
            }
        }

    def get_map_data(self) -> List[Dict]:
        """获取全部蜜源点数据（供前端地图渲染）"""
        if not self.is_initialized:
            self.initialize()

        all_sources = self.mysql_store.get_all_nectar_sources()
        map_points = []
        for ns in all_sources:
            map_points.append({
                "id": ns['id'],
                "plant_name": ns['plant_name'],
                "province": ns['province'],
                "city": ns['city'],
                "location_name": ns['location_name'],
                "lat": float(ns['latitude']),
                "lng": float(ns['longitude']),
                "flowering_start": ns['flowering_start'],
                "flowering_end": ns['flowering_end'],
                "best_honey_start": ns['best_honey_start'],
                "best_honey_end": ns['best_honey_end'],
                "honey_yield": float(ns['honey_yield']) if ns['honey_yield'] else None,
                "is_primary": ns['is_primary']
            })
        return map_points

    def get_stats(self) -> Dict:
        """获取系统统计数据"""
        return self.mysql_store.get_stats()


# ==================== 测试入口 ====================
if __name__ == '__main__':
    # 加载.env
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

    engine = RAGEngine()
    engine.initialize()

    print("\n" + "=" * 60)
    print("测试1：蜜源推荐（广东 3月）")
    print("=" * 60)
    result = engine.recommend_nectar(province="广东", month=3)
    print(f"筛选到 {result['total']} 条蜜源")
    for rec in result['recommendations'][:3]:
        print(f"  - {rec['plant_name']} @ {rec['province']}{rec['city']}")
        print(f"    花期：{rec['flowering_start']}-{rec['flowering_end']}月，产蜜量：{rec['honey_yield']}kg/群")

    print("\n" + "=" * 60)
    print("测试2：智能问答（荆条蜜源在哪）")
    print("=" * 60)
    result = engine.query("荆条蜜源在哪些省份？什么时候开花？", top_k=3)
    print(f"方法: {result['method']}")
    print(f"回答: {result['answer'][:300]}")
    print(f"来源: {result['total_sources']} 条")
    for s in result['sources']:
        if s.get('type') == 'nectar_source':
            print(f"  - {s['plant_name']} @ {s['province']}{s['city']} (score={s['score']})")
        else:
            print(f"  - 知识库: {s.get('title', '')} (score={s['score']})")

    print("\n" + "=" * 60)
    print("测试3：地图数据")
    print("=" * 60)
    map_data = engine.get_map_data()
    print(f"地图点数: {len(map_data)}")
    for p in map_data[:3]:
        print(f"  - {p['plant_name']} ({p['lat']}, {p['lng']}) @ {p['province']}{p['city']}")

    print("\n" + "=" * 60)
    print("测试4：统计信息")
    print("=" * 60)
    stats = engine.get_stats()
    print(f"蜜源记录: {stats['nectar_count']}")
    print(f"蜜源植物: {stats['plant_count']}种")
    print(f"覆盖省份: {stats['province_count']}个")
    print(f"知识库: {stats['knowledge_count']}篇")
