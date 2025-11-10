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
import gradio as gr

# 偵測 GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Gradio 介面 CSS 與 HTML
footer_html = """
<div class='footer-info'>
    <p>⚠️ <strong>重要提醒</strong>：本服務由 AI 驅動，回答僅供參考，不構成正式法律意見</p>
    <p class='footer-credits'>🚀 Powered by API · 💚 Built with Gradio</p>
</div>
"""

adaptive_css = """
/* ========== 深淺色模式變數 (★ 已更新為 EY 色系) ========== */
:root {
    /* ★ 修正：背景改為中性淺灰白 */
    --bg-gradient-start: #f5f5f5;
    --bg-gradient-end: #ffffff;
    --card-bg: #ffffff;
    --card-border: #e2e8f0;
    --text-primary: #2E2E38; /* ★ 修正：主要文字改為深灰 */
    --text-secondary: #64748b;
    --text-tertiary: #94a3b8;
    --input-bg: #f8fafc;
    --input-border: #e2e8f0;
    --input-focus-border: #FFEB00; /* ★ 修正：焦點改為 EY 黃 */
    --chat-bg: #fafafa;
    --bot-bubble-bg: #ffffff;
    --bot-bubble-border: #e2e8f0;
    --example-bg: #ffffff;
    --example-hover-bg: #fffdeB; /* ★ 修正：範例 hover 改為淺黃 */
    --divider: #f1f5f9;
    --shadow-sm: rgba(0, 0, 0, 0.05);
    --shadow-md: rgba(0, 0, 0, 0.08);
    --shadow-lg: rgba(255, 235, 0, 0.4); /* ★ 修正：陰影改為 EY 黃 */
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-gradient-start: #0f172a;
        --bg-gradient-end: #1e293b;
        --card-bg: #1e293b;
        --card-border: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-tertiary: #94a3b8;
        --input-bg: #334155;
        --input-border: #475569;
        --input-focus-border: #FFEB00; /* ★ 修正：焦點改為 EY 黃 */
        --chat-bg: #0f172a;
        --bot-bubble-bg: #334155;
        --bot-bubble-border: #475569;
        --example-bg: #334155;
        --example-hover-bg: #3a3800; /* ★ 修正：範例 hover 改為深黃 */
        --divider: #334155;
        --shadow-sm: rgba(0, 0, 0, 0.3);
        --shadow-md: rgba(0, 0, 0, 0.4);
        --shadow-lg: rgba(255, 235, 0, 0.3); /* ★ 修正：陰影改為 EY 黃 */
    }
}

/* ========== 全域樣式 (★ 已更新背景) ========== */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: linear-gradient(to bottom, var(--bg-gradient-start), var(--bg-gradient-end)) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans TC', sans-serif !important;
    min-height: 100vh !important;
    padding: 2rem 1rem !important;
}

/* ========== 主容器 ========== */
.gradio-container > .contain {
    background: var(--card-bg) !important;
    border-radius: 24px !important;
    padding: 3rem 2.5rem !important;
    box-shadow: 0 20px 60px var(--shadow-md) !important;
    border: 1px solid var(--card-border) !important;
}

/* ========== 標題區 (★ 已更新標題顏色) ========== */
.title-section {
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 2px solid var(--divider);
}
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    /* ★ 修正：移除漸層，改為 EY 深灰色 */
    color: #2E2E38 !important;
    background: none !important;
    -webkit-background-clip: initial !important;
    -webkit-text-fill-color: initial !important;
    background-clip: initial !important;
    margin: 0 0 1rem 0;
    letter-spacing: -0.02em;
}
.subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    line-height: 1.6;
    margin: 0;
}

/* ========== 聊天區域 (★ 已更新使用者對話框) ========== */
.gradio-container .chatbot {
    border: 2px solid var(--card-border) !important;
    border-radius: 20px !important;
    background: var(--chat-bg) !important;
    padding: 0 !important;
    position: relative;
    z-index: 1;
}
.gradio-container .message-wrap { padding: 1rem !important; }
.gradio-container [data-testid="user"] { justify-content: flex-end !important; }
.gradio-container [data-testid="user"] .message {
    /* ★ 修正：使用者對話框改為 EY 深灰色 */
    background: #2E2E38 !important;
    color: white !important;
    border: none !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 0.75rem 1.25rem !important;
    box-shadow: 0 4px 12px var(--shadow-sm) !important; /* 陰影改為中性 */
    max-width: 80% !important;
}
.gradio-container [data-testid="bot"] .message {
    background: var(--bot-bubble-bg) !important;
    color: var(--text-primary) !important;
    border: 2px solid var(--bot-bubble-border) !important;
    border-radius: 20px 20px 20px 4px !important;
    padding: 0.75rem 1.25rem !important;
    box-shadow: 0 2px 8px var(--shadow-sm) !important;
    max-width: 80% !important;
}

/* ========== 輸入區域 (★ 已更新焦點顏色) ========== */
.input-row {
    display: flex !important;
    gap: 12px !important;
    align-items: stretch !important;
    margin-top: 1.5rem !important;
    position: relative;
    z-index: 10;
}
.input-row .gradio-textbox {
    flex: 6 1 0% !important;
}
.gradio-container textarea {
    background: var(--input-bg) !important;
    border: 2px solid var(--input-border) !important;
    border-radius: 16px !important;
    padding: 1rem 1.5rem !important;
    font-size: 1rem !important;
    color: var(--text-primary) !important;
    transition: all 0.2s ease !important;
    line-height: 1.5 !important;
    min-height: 56px !important;
}
.gradio-container textarea:focus {
    background: var(--card-bg) !important;
    border-color: var(--input-focus-border) !important; /* EY 黃 */
    outline: none !important;
    box-shadow: 0 0 0 4px rgba(255, 235, 0, 0.25) !important; /* EY 黃陰影 */
}
.gradio-container textarea::placeholder {
    color: var(--text-tertiary) !important;
}

/* ========== 按鈕 (★ 已更新發送鈕顏色) ========== */
.input-row .gradio-button {
    flex: 1 1 0% !important;
    max-width: fit-content !important;
}
.gradio-container button {
    border-radius: 16px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 1rem 2rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    border: none !important;
    white-space: nowrap !important;
    height: 100% !important;
}
.send-btn {
    /* ★ 修正：發送鈕改為 EY 黃，文字改為深灰 */
    background: #FFEB00 !important;
    color: #2E2E38 !important;
    box-shadow: 0 4px 12px var(--shadow-lg) !important; /* EY 黃陰影 */
}
.send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--shadow-lg) !important; /* EY 黃陰影 */
}
.clear-btn {
    background: var(--card-bg) !important;
    color: var(--text-secondary) !important;
    border: 2px solid var(--card-border) !important;
}
.clear-btn:hover {
    background: var(--input-bg) !important;
    color: var(--text-primary) !important;
}

/* ========== 範例問題 (★ 已更新 hover 顏色) ========== */
.examples-section {
    margin-top: 2rem;
    position: relative;
    z-index: 5;
}
.examples-section .gradio-label-wrap {
    padding: 0 !important;
    margin: 0 !important;
}
.examples-section .gradio-label-wrap label {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    margin-bottom: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    display: block !important;
}
.gradio-container .examples button {
    background: var(--example-bg) !important;
    border: 2px solid var(--card-border) !important;
    border-radius: 12px !important;
    padding: 0.875rem 1.25rem !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
}
.gradio-container .examples button:hover {
    /* ★ 修正：Hover 改為 EY 淺黃 */
    background: var(--example-hover-bg) !important;
    border-color: #FFEB00 !important; /* EY 黃 */
    color: #2E2E38 !important; /* 深灰文字 */
    transform: translateX(4px) !important;
}

/* ========== 底部資訊 ========== */
.footer-info {
    text-align: center;
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 2px solid var(--divider);
    color: var(--text-secondary);
    position: relative;
    z-index: 5;
}
.footer-info p { margin: 0.5rem 0; font-size: 0.95rem; }
.footer-credits { opacity: 0.7; margin-top: 1rem !important; }

/* ========== 響應式設計 ========== */
@media (max-width: 768px) {
    .gradio-container > .contain {
        padding: 2rem 1.5rem !important;
    }
    .main-title { font-size: 2rem !important; }
    .subtitle { font-size: 1rem !important; }
    .input-row { flex-wrap: wrap !important; }
    .input-row .gradio-textbox {
        flex-basis: 100% !important;
    }
    .input-row .gradio-button {
        flex: 1 !important;
    }
    .gradio-container button {
        min-width: 120px !important;
    }
}
"""

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

# RAG 核心函式 (Hybrid v2)
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

# Gradio 輔助函式
def handle_submit(message, history_tuples):
    chat_history_for_gemma = []
    for user_msg, bot_msg in history_tuples:
        if user_msg:
            chat_history_for_gemma.append({"role": "user", "content": user_msg})
        if bot_msg:
            chat_history_for_gemma.append({"role": "assistant", "content": bot_msg})

    try:
        response_text, _ = ask_laborlaw_gemma_conversational_v2(
            query=message,
            chat_history=chat_history_for_gemma,
            faiss_index=global_index,
            chunk_list=chunks,
            emb_model=embedding_model,
            bm25_model=global_bm25_model,
            top_k=20,
            chunks_to_feed=5,
            debug=False
        )
    except Exception as e:
        response_text = f"抱歉，處理您的請求時發生錯誤：{e}"

    history_tuples.append([message, response_text])
    return "", history_tuples

def clear_conversation():
    return "", []

# 建構並啟動 Gradio 介面
with gr.Blocks(css=adaptive_css, title="安永銀行勞動權益小助手", elem_classes="contain") as demo:
    gr.HTML("""
        <div class='title-section'>
            <h1 class='main-title'>🏢 安永銀行勞動權益小助手</h1>
            <p class='subtitle'>您的專屬勞動法律顧問 · 快速、準確、易懂的法規諮詢服務</p>
        </div>
    """)

    chatbot = gr.Chatbot(
        type="messages",
        height=800,
        show_copy_button=True,
        avatar_images=(None, None),
        elem_classes="chatbot"
    )

    with gr.Row(elem_classes="input-row"):
        message = gr.Textbox(
            placeholder="💬 請輸入您的勞動法規問題...",
            show_label=False,
            scale=6,
            lines=2,
            autoscroll=True
        )
        send_btn = gr.Button("發送", elem_classes="send-btn", scale=1)
        clear_btn = gr.Button("清除", elem_classes="clear-btn", scale=1)

    with gr.Column(elem_classes="examples-section"):
        gr.Examples(
            label="💡 常見問題",
            examples=[
                ["勞工每天最長可以工作幾小時？"],
                ["一個月最多可以加班多久？"],
                ["加班費應該如何計算？"],
                ["特休假的規定是什麼？"]
            ],
            inputs=message
        )

    gr.HTML(footer_html)

    message.submit(handle_submit, [message, chatbot], [message, chatbot])
    send_btn.click(handle_submit, [message, chatbot], [message, chatbot])
    clear_btn.click(clear_conversation, None, [message, chatbot])

demo.launch(share=True, debug=True)