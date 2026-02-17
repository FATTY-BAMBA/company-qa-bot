"""
Company Q&A Chat Engine.

Handles visitor queries by:
1. Embedding the query
2. Retrieving relevant chunks from Pinecone
3. Building a grounded prompt with retrieved context
4. Generating a natural response via OpenAI
"""

import logging
import time
from typing import List, Dict, Optional

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

# Retrieval config
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

SYSTEM_PROMPT = """你是一位親切專業的客服助理，負責回答訪客關於公司服務的問題。

## 回答規則

1. **僅根據提供的參考資料回答**。不要編造公司沒有的服務或功能。
2. **使用繁體中文回答**，語氣親切、專業。
3. 如果參考資料中有相關連結，自然地在回答中包含連結。格式範例：「您可以在這裡查看詳情：[連結]」
4. 如果參考資料中沒有連結，就正常回答文字內容即可，不要提到「沒有連結」。
5. 如果有多個相關結果，將它們整合成一個完整的回答，必要時列出選項。
6. 如果訪客的問題太廣泛，可以詢問進一步的細節以縮小範圍。
7. 如果找不到相關答案，禮貌地回覆：「很抱歉，我目前無法回答這個問題。請透過 support@example.com 或撥打 02-1234-5678 聯繫我們的客服團隊，我們會盡快為您服務。」
8. 保持回答簡潔明瞭，避免冗長重複。

## 重要
- 絕對不要編造不在參考資料中的資訊
- 不要回答與公司服務無關的問題
- 如果不確定，寧可引導訪客聯繫客服"""


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


def retrieve_context(query: str) -> List[Dict]:
    """
    Retrieve relevant Q&A entries from Pinecone.

    Returns a list of matches with metadata, filtered by similarity threshold.
    """
    query_embedding = embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True,
        namespace=COMPANY_QA_NAMESPACE,
    )

    matches = []
    for match in results.get("matches", []):
        if match["score"] >= SIMILARITY_THRESHOLD:
            matches.append({
                "score": round(match["score"], 4),
                "metadata": match["metadata"],
            })

    logger.info(f"Retrieved {len(matches)} matches for query: '{query[:50]}...'")
    return matches


def build_context_block(matches: List[Dict]) -> str:
    """Format retrieved matches into a context block for the LLM prompt."""
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


def chat(
    query: str,
    conversation_history: Optional[List[Dict]] = None,
) -> Dict:
    """
    Process a visitor's question and generate a grounded response.

    Args:
        query: The visitor's question
        conversation_history: Optional list of previous messages for multi-turn

    Returns:
        Dict with answer, sources, confidence, and latency
    """
    start_time = time.time()

    # ── Retrieve context ──
    matches = retrieve_context(query)
    context_block = build_context_block(matches)

    # ── Build messages ──
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history if provided (for multi-turn)
    if conversation_history:
        messages.extend(conversation_history)

    # Add current query with context
    user_message = f"""訪客問題：{query}

以下是從知識庫中檢索到的相關參考資料：

{context_block}

請根據以上參考資料回答訪客的問題。"""

    messages.append({"role": "user", "content": user_message})

    # ── Generate response ──
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content
    elapsed = round(time.time() - start_time, 3)

    # ── Calculate confidence ──
    # Hybrid: avg retrieval score + whether we found any matches
    if matches:
        avg_score = sum(m["score"] for m in matches) / len(matches)
        top_score = matches[0]["score"]
        confidence = round((avg_score + top_score) / 2, 4)
    else:
        confidence = 0.0

    # ── Build source references ──
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

    result = {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "latency_seconds": elapsed,
        "model": CHAT_MODEL,
        "matches_found": len(matches),
    }

    logger.info(
        f"Chat response generated — confidence: {confidence}, "
        f"sources: {len(sources)}, latency: {elapsed}s"
    )

    return result
