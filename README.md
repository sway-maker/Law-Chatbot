# 勞動權益小助手

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/unsloth/gemma-3-4b-it)

一個基於台灣《勞動基準法》的 **RAG 聊天機器人**，使用 **BM25 + FAISS + RRF 混合檢索**，搭配 **Gemma-3-4B-it** 生成回應。  
支援 K-Fold 交叉驗證評估，並提供美觀的 Gradio 網頁介面。

> ⚠️ **重要提醒**：本工具由 AI 驅動，回答僅供參考，**不構成正式法律意見**。

## 功能特色

- **混合檢索 (Hybrid Retrieval)**：BM25 關鍵字 + FAISS 語意 + RRF 融合。
- **路由機制**：自動判斷問題是否與勞動法相關，無關問題會禮貌拒絕。
- **K-Fold 評估**：使用完整 Q&A 資料集進行 5-Fold 驗證，計算語意相似度來評分。
- **Gradio 介面**：深淺色模式、範例問題、聊天歷史、複製按鈕。
- **多輪對話支援**：完整保留上下文，適合連續提問。

## 技術架構
### 混合檢索流程
- BM25 檢索：基於關鍵字匹配，使用 **jieba** 分詞，適合精確詞彙查詢
- FAISS 檢索：基於語意相似度，使用 **intfloat/multilingual-e5-base** 嵌入模型
- RRF 融合：Reciprocal Rank Fusion 整合兩種檢索結果，取 Top-K 相關法條

### 生成模型
- 模型：**unsloth/gemma-3-4b-it**
- 推理：使用 Hugging Face Transformers
- 優化：支援量化與 GPU 加速

## 評估方法
使用QA問答集（範例為切5份）來評估模型表現：
- 評估指標：語意相似度（Cosine Similarity）
- 資料來源：**labor_law_qa.docx** 中的 Q&A 對
- 輸出：每個 Fold 的平均分數與整體平均


## 介面預覽
<img src="image/interface.png" alt="Gradio 介面" width="70%">
<img src="image/evaluate.png" alt="K-Fold 評估結果" width="40%">

## 🛠️ 檔案結構
```
Law-Chatbot/
│
├── File/
│   ├── labor_law_articles/       # 法條文字檔
│   └── labor_law_qa.docx         # Q&A 評估資料
│
├── image/
│   ├── interface.png             # 介面截圖
│   └── evaluate.png              # 評估截圖
│
├── interface.py                  # Gradio 介面
├── main_code.py                  # 主程式（資料處理、索引、RAG、評估）
└── README.md                     # 本文件
```


## 免責聲明
本專案僅供學習與參考使用，不提供任何法律效力。如需正式法律意見，請諮詢專業律師。
