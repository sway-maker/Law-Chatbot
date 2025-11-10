# labor_law_rag.py (主程式：包含資料處理、索引建立、RAG 核心、K-Fold 評估)

# Python imports
import os
import re
import jieba
from docx import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity

# 偵測 GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 資料讀取與處理
def read_docx(filepath):
    if not os.path.exists(filepath):
        return []
    doc = Document(filepath)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

def normalize_text(s):
    s = s.replace('\u3000', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# 讀取條文
try:
    with open("labor_law_articles", "r", encoding="utf-8") as f:
        law_articles = [normalize_text(line) for line in f.readlines() if normalize_text(line)]
except FileNotFoundError:
    law_articles = []

# 讀取 QA
law_qas_raw = read_docx("labor_law_qa.docx")

# 解析 Q&A pairs
qa_pairs = []
i = 0
while i < len(law_qas_raw):
    para = law_qas_raw[i]
    if para.upper().startswith('Q:') or para.upper().startswith('Q：'):
        current_q = para
        current_a_parts = []
        i += 1
        while i < len(law_qas_raw) and not (law_qas_raw[i].upper().startswith('Q:') or law_qas_raw[i].upper().startswith('Q：')):
            a_part = law_qas_raw[i]
            if a_part:
                current_a_parts.append(a_part)
            i += 1
        if current_a_parts:
            combined_a = " ".join(current_a_parts)
            qa_pairs.append({"q": current_q, "a": combined_a})
        else:
            i += 1
    else:
        i += 1

# 最終合併索引文件 (僅法條)
docs = law_articles

# 文件切割 (Chunking)
chunk_size = 512
stride = 512

chunks = []
for doc in docs:
    doc = normalize_text(doc)
    if not doc:
        continue
    if len(doc) <= chunk_size:
        chunks.append(doc)
    else:
        for i in range(0, len(doc), stride):
            chunk = doc[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)

# Embedding 模型與 FAISS 索引
embedding_model_name = "intfloat/multilingual-e5-base"
embedding_model = SentenceTransformer(embedding_model_name, device=device)

if chunks:
    chunks_embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    d = chunks_embeddings.shape[1]
    global_index = faiss.IndexFlatL2(d)
    global_index.add(chunks_embeddings)
else:
    global_index = None

# BM25 索引
if chunks:
    tokenized_global_chunks = [list(jieba.cut(chunk)) for chunk in tqdm(chunks, desc="Tokenizing")]
    global_bm25_model = BM25Okapi(tokenized_global_chunks)
else:
    global_bm25_model = None

# 檢索函式 (BM25 & RRF)
def bm25_retrieve(query: str, chunks: list, bm25_model: BM25Okapi, top_k: int = 20) -> list:
    tokenized_query = list(jieba.cut(query))
    scores = bm25_model.get_scores(tokenized_query)
    top_n_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_n_indices]

def reciprocal_rank_fusion(*ranked_lists, k=60) -> list:
    scores = {}
    if not ranked_lists or all(not lst for lst in ranked_lists):
        return []

    for rl in ranked_lists:
        if not rl: continue
        for rank, doc_id in enumerate(rl, start=1):
            if not isinstance(doc_id, str): doc_id = str(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    if not scores: return []
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [d for d, _ in fused]

# 載入 Gemma-3-4B-it
generator_model_name = "unsloth/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
generator_pipeline = pipeline(
    "text-generation",
    model=generator_model_name,
    tokenizer=tokenizer,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
    device_map="auto",
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<end_of_turn>")
]

RELEVANCE_THRESHOLD = 1.0

# RAG 核心函式
def ask_laborlaw_gemma_conversational_v2(
    query: str,
    chat_history: list,
    faiss_index: faiss.Index,
    chunk_list: list,
    emb_model: SentenceTransformer,
    bm25_model: BM25Okapi,
    top_k: int = 20,
    chunks_to_feed: int = 5,
    debug: bool = False
):
    if faiss_index is None or faiss_index.ntotal == 0 or not chunk_list or bm25_model is None:
        return "抱歉，知識庫尚未準備就緒。", chat_history

    q_emb = emb_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(q_emb, top_k)
    best_distance = distances[0][0]
    is_relevant = best_distance < RELEVANCE_THRESHOLD

    system_prompt = ""
    user_message_content = ""

    if is_relevant:
        system_prompt = "你是一位熟悉台灣《勞動基準法》的專業法律助理。請根據提供的內容回答問題，不可憑空捏造。請以正式、條理分明的中文回答，並盡可能附上相關法條依據。"

        valid_indices = [i for i in indices[0] if i < len(chunk_list)]
        relevant_chunks_embedding = [chunk_list[i] for i in valid_indices]

        relevant_chunks_bm25 = bm25_retrieve(
            query=query,
            chunks=chunk_list,
            bm25_model=bm25_model,
            top_k=top_k
        )

        fused_chunks_list = reciprocal_rank_fusion(
            relevant_chunks_bm25,
            relevant_chunks_embedding
        )

        final_fused_chunks = fused_chunks_list[:chunks_to_feed]

        if not final_fused_chunks:
            relevant_chunks = "（RRF 融合後未檢索到相關資料）"
        else:
            relevant_chunks = "\n\n".join(final_fused_chunks)

        user_message_content = f"""
以下是勞動基準法相關資料：
---
{relevant_chunks}
---
請根據上面資料回答下列問題：
{query}
"""
    else:
        system_prompt = """
你是一位專業的 AI 助理，你的 "唯一" 職責是回答台灣《勞動基準法》的相關問題。
你 "絕對不可以" 回答任何與勞動基準法無關的問題。
如果使用者詢問無關問題（例如：天氣、食譜、閒聊、蛋糕），請你禮貌地拒絕，並清楚說明你的專長是勞動法規。
"""
        user_message_content = query

    messages_to_send = chat_history.copy()
    if not chat_history:
        user_message_content = f"{system_prompt}\n\n{user_message_content}"
    messages_to_send.append({"role": "user", "content": user_message_content})

    prompt = tokenizer.apply_chat_template(
        messages_to_send,
        tokenize=False,
        add_generation_prompt=True
    )

    try:
        out_list = generator_pipeline(
            prompt,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=terminators,
        )

        full_text = out_list[0]["generated_text"]
        answer = full_text[len(prompt):].strip()
        answer = answer.replace("<end_of_turn>", "").strip()

        if not answer:
            answer = "（模型沒有生成任何回應）"

        return answer, chat_history

    except Exception as e:
        return f"生成答案時發生錯誤：{e}", chat_history

# K-Fold 評估
N_SPLITS = 5

if qa_pairs:
    qa_pairs_array = np.array(qa_pairs)
else:
    qa_pairs_array = np.array([])

if qa_pairs_array.size > 0 and global_index is not None and global_bm25_model is not None:
    if len(qa_pairs_array) < N_SPLITS:
        N_SPLITS = len(qa_pairs_array)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    all_fold_scores = []
    all_fold_results = []

    for fold_num, (train_index_ignored, test_index) in enumerate(kf.split(qa_pairs_array)):
        evaluation_set = qa_pairs_array[test_index]
        fold_results = []

        for item in tqdm(evaluation_set, desc=f"Fold {fold_num+1} Evaluating"):
            question = item['q']
            ground_truth_answer = item['a']

            generated_answer, _ = ask_laborlaw_gemma_conversational_v2(
                query=question,
                chat_history=[],
                faiss_index=global_index,
                chunk_list=chunks,
                emb_model=embedding_model,
                bm25_model=global_bm25_model,
                top_k=20,
                chunks_to_feed=5,
                debug=False
            )

            fold_results.append({
                "question": question,
                "ground_truth": ground_truth_answer,
                "generated": generated_answer
            })

        ground_truth_list = [res['ground_truth'] for res in fold_results]
        generated_list = [res['generated'] for res in fold_results]

        truth_embeddings = embedding_model.encode(ground_truth_list, show_progress_bar=False)
        gen_embeddings = embedding_model.encode(generated_list, show_progress_bar=False)

        similarities = np.diag(cosine_similarity(truth_embeddings, gen_embeddings))
        average_similarity = np.mean(similarities)

        all_fold_scores.append(average_similarity)
        all_fold_results.extend(fold_results)

    final_mean = np.mean(all_fold_scores)
    final_std = np.std(all_fold_scores)

if __name__ == "__main__":
    # 在這裡執行 K-Fold 評估或其他主邏輯
    print("K-Fold 評估完成。")
    print(f"最終平均語意相似度 (Mean): {final_mean:.4f}")
    print(f"相似度標準差 (Std. Dev.): {final_std:.4f}")