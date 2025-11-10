# app.py (Gradio 介面部分：匯入主程式變數並建構介面)

import gradio as gr
from labor_law_rag import (  # 從主程式匯入必要變數與函式
    chunks,
    embedding_model,
    global_bm25_model,
    global_index,
    ask_laborlaw_gemma_conversational_v2,
)

# Gradio介面 CSS 與 HTML
footer_html = """
<div class='footer-info'>
    <p>⚠️ <strong>重要提醒</strong>：本服務由 AI 驅動，回答僅供參考，不構成正式法律意見</p>
    <p class='footer-credits'>🚀 Powered by API · 💚 Built with Gradio</p>
</div>
"""

adaptive_css = """
/* ========== 深淺色模式變數 ========== */
:root {
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

/* ========== 全域樣式 ========== */
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

/* ========== 標題區 ========== */
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

/* ========== 聊天區域 ========== */
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

# Gradio 輔助函式

def handle_submit(message, history_messages):
    """
    處理 Gradio 的提交事件
    - ★ 格式修正：history_messages 現在是 "messages" 格式:
    - [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    - 呼叫 RAG 核心，使用「全域索引」(僅法條)
    """
    # 1. 準備聊天歷史記錄
    chat_history_for_gemma = history_messages

    # 2. 呼叫 RAG 核心
    try:
        response_text, _ = ask_laborlaw_gemma_conversational_v2(
            query=message,
            chat_history=chat_history_for_gemma, # ★ 直接傳入
            faiss_index=global_index,
            chunk_list=chunks,
            emb_model=embedding_model,
            bm25_model=global_bm25_model,
            top_k=20,
            chunks_to_feed=5,
            debug=False
        )
    except Exception as e:
        print(f"!!! Gradio Handle Error: {e}")
        response_text = f"抱歉，處理您的請求時發生錯誤：{e}"

    history_messages.append({"role": "user", "content": message})
    history_messages.append({"role": "assistant", "content": response_text})

    return "", history_messages

def clear_conversation():
    """
    清除對話
    """
    return "", []

print("Gradio helper functions (handle_submit, clear_conversation) defined.")

# 建構 Gradio 介面
with gr.Blocks(css=adaptive_css, title="安永銀行勞動權益小助手", elem_classes="contain") as demo:

    # 標題區
    gr.HTML("""
        <div class='title-section'>
            <h1 class='main-title'>🏢 安永銀行勞動權益小助手</h1>
            <p class='subtitle'>您的專屬勞動法律顧問 · 快速、準確、易懂的法規諮詢服務</p>
        </div>
    """)

    # 聊天區
    chatbot = gr.Chatbot(
        type="messages",
        height=800,
        show_copy_button=True,
        avatar_images=(None, None),
        elem_classes="chatbot"
    )

    # 輸入區
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

    # 範例問題
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

    # 事件綁定
    message.submit(handle_submit, [message, chatbot], [message, chatbot])
    send_btn.click(handle_submit, [message, chatbot], [message, chatbot])
    clear_btn.click(clear_conversation, None, [message, chatbot])

if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(share=True, debug=True)