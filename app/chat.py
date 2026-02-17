"""
Company Q&A Chat Engine.

Optimized with:
- Query caching (LRU, 1hr TTL)
- Streaming responses (SSE)
- TOP_K=3 with 0.4 threshold for better accuracy
- 500 max tokens for concise answers
"""

import hashlib
import json
import logging
import time
from typing import List, Dict, Optional
from collections import OrderedDict

from openai import OpenAI
from pinecone import Pinecone

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    COMPANY_QA_NAMESPACE,
)

logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ── Optimized retrieval config ──
TOP_K = 3
SIMILARITY_THRESHOLD = 0.4
MAX_TOKENS = 500


# ═══════════════════════════════════════════════════════════
# Query Cache (LRU, in-memory)
# ═══════════════════════════════════════════════════════════

class QueryCache:
    def __init__(self, max_size=200, ttl_seconds=3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache = OrderedDict()

    def _key(self, query):
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query):
        key = self._key(query)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self._cache.move_to_end(key)
                logger.info(f"Cache HIT for query: '{query[:40]}...'")
                return entry["result"]
            else:
                del self._cache[key]
        return None

    def put(self, query, result):
        key = self._key(query)
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = {"result": result, "timestamp": time.time()}

    def clear(self):
        self._cache.clear()
        logger.info("Query cache cleared")


cache = QueryCache(max_size=200, ttl_seconds=3600)


SYSTEM_PROMPT = """你是一位親切專業的客服助理，負責回答訪客關於公司服務的問題。

## 回答規則

1. **打招呼和寒暄**：如果訪客只是打招呼（如「你好」、「嗨」、「Hi」），請親切回應並詢問有什麼可以幫忙的。例如：「您好！很高興為您服務 😊 請問有什麼可以幫您的嗎？」
2. **僅根據提供的參考資料回答專業問題**。不要編造公司沒有的服務或功能。
3. **使用繁體中文回答**，語氣親切、專業。
4. 如果參考資料中有相關連結，自然地在回答中包含連結。格式範例：「您可以在這裡查看詳情：[連結]」
5. 如果參考資料中沒有連結，就正常回答文字內容即可，不要提到「沒有連結」。
6. 如果有多個相關結果，將它們整合成一個完整的回答，必要時列出選項。
7. 如果訪客的問題太廣泛，可以詢問進一步的細節以縮小範圍。
8. 如果找不到相關答案，禮貌地回覆：「很抱歉，我目前無法回答這個問題。請透過 support@example.com 或撥打 02-1234-5678 聯繫我們的客服團隊，我們會盡快為您服務。」
9. 保持回答簡潔明瞭，用2-3句話回答，除非問題需要更詳細的解釋。

## 重要
- 對打招呼、感謝、再見等社交性訊息，直接親切回應即可，不需要參考資料
- 絕對不要編造不在參考資料中的資訊
- 不要回答與公司服務無關的問題
- 如果不確定，寧可引導訪客聯繫客服"""


def embed_query(query):
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def retrieve_context(query):
    query_embedding = embed_query(query)
    results = index.query(
        vector=query_embedding, top_k=TOP_K,
        include_metadata=True, namespace=COMPANY_QA_NAMESPACE,
    )
    matches = []
    for match in results.get("matches", []):
        if match["score"] >= SIMILARITY_THRESHOLD:
            matches.append({"score": round(match["score"], 4), "metadata": match["metadata"]})
    logger.info(f"Retrieved {len(matches)} matches for query: '{query[:50]}...'")
    return matches


def build_context_block(matches):
    if not matches:
        return "（找不到相關的參考資料）"
    blocks = []
    for i, match in enumerate(matches, 1):
        meta = match["metadata"]
        block = f"【參考資料 {i}】（相關度：{match['score']}）\n"
        block += f"問題：{meta.get('question', '')}\n"
        block += f"答案：{meta.get('answer', '')}"
        if meta.get("link"):
            block += f"\n連結：{meta['link']}"
        if meta.get("category"):
            block += f"\n分類：{meta['category']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _build_messages(query, context_block, conversation_history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    user_message = f"""訪客問題：{query}

以下是從知識庫中檢索到的相關參考資料：

{context_block}

請根據以上參考資料回答訪客的問題。"""
    messages.append({"role": "user", "content": user_message})
    return messages


def _calc_confidence(matches):
    if matches:
        avg_score = sum(m["score"] for m in matches) / len(matches)
        return round((avg_score + matches[0]["score"]) / 2, 4)
    return 0.0


def _build_sources(matches):
    sources = []
    for match in matches:
        source = {
            "row_number": match["metadata"].get("row_number"),
            "question": match["metadata"].get("question"),
            "relevance_score": match["score"],
        }
        if match["metadata"].get("category"):
            source["category"] = match["metadata"]["category"]
        if match["metadata"].get("link"):
            source["link"] = match["metadata"]["link"]
        sources.append(source)
    return sources


# ═══════════════════════════════════════════════════════════
# Standard (non-streaming) chat
# ═══════════════════════════════════════════════════════════

def chat(query, conversation_history=None):
    start_time = time.time()

    # Check cache
    cached = cache.get(query)
    if cached and not conversation_history:
        cached["latency_seconds"] = round(time.time() - start_time, 3)
        cached["cached"] = True
        return cached

    # Retrieve and generate
    matches = retrieve_context(query)
    context_block = build_context_block(matches)
    messages = _build_messages(query, context_block, conversation_history)

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL, messages=messages,
        temperature=0.3, max_tokens=MAX_TOKENS,
    )

    answer = response.choices[0].message.content
    elapsed = round(time.time() - start_time, 3)

    result = {
        "answer": answer,
        "sources": _build_sources(matches),
        "confidence": _calc_confidence(matches),
        "latency_seconds": elapsed,
        "model": CHAT_MODEL,
        "matches_found": len(matches),
    }

    logger.info(f"Chat response — confidence: {result['confidence']}, latency: {elapsed}s")

    if not conversation_history:
        cache.put(query, result)

    return result


# ═══════════════════════════════════════════════════════════
# Streaming chat (SSE)
# ═══════════════════════════════════════════════════════════

def chat_stream(query, conversation_history=None):
    """
    Generator that yields SSE-formatted events:
    - event: metadata (sources, confidence)
    - event: chunk (text pieces as they arrive)
    - event: done (full answer, latency)
    """
    start_time = time.time()

    matches = retrieve_context(query)
    context_block = build_context_block(matches)
    messages = _build_messages(query, context_block, conversation_history)

    confidence = _calc_confidence(matches)
    sources = _build_sources(matches)

    # Yield metadata first so frontend can show sources immediately
    yield f"data: {json.dumps({'type': 'metadata', 'sources': sources, 'confidence': confidence}, ensure_ascii=False)}\n\n"

    # Stream LLM response
    stream = openai_client.chat.completions.create(
        model=CHAT_MODEL, messages=messages,
        temperature=0.3, max_tokens=MAX_TOKENS, stream=True,
    )

    full_answer = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_answer += text
            yield f"data: {json.dumps({'type': 'chunk', 'content': text}, ensure_ascii=False)}\n\n"

    elapsed = round(time.time() - start_time, 3)

    # Cache result
    if not conversation_history:
        result = {
            "answer": full_answer, "sources": sources,
            "confidence": confidence, "latency_seconds": elapsed,
            "model": CHAT_MODEL, "matches_found": len(matches),
        }
        cache.put(query, result)

    # Final event
    yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'latency_seconds': elapsed}, ensure_ascii=False)}\n\n"

    logger.info(f"Streamed response — confidence: {confidence}, latency: {elapsed}s")


def clear_cache():
    """Clear the query cache. Call after re-indexing."""
    cache.clear()
