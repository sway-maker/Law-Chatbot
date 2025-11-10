# Law-Chatbot
這是一個使用 RAG (Retrieval-Augmented Generation，檢索增強生成) 技術的聊天機器人，專門用於回答台灣《勞動基準法》的相關問題。

＃專案簡介

本專案是一個基於 RAG 技術的台灣勞動法規問答系統。採用 FAISS (語意) 和 BM25 (關鍵字) 進行混合檢索，並透過 RRF 演算法融合結果。後端使用 Gemma 3 4B-it 模型生成答案，並以 Gradio 建構互動式網頁介面。

＃專案特色

-混合檢索 (Hybrid Retrieval)：同時使用 FAISS (intfloat/multilingual-e5-base) 進行語意檢索和 BM25 進行關鍵字檢索。

-RRF 融合 (Reciprocal Rank Fusion)：將兩種檢索結果的排序進行智慧融合，提高檢索準確性。

-RAG 路由 (RAG Router)：透過 L2 距離判斷問題是否與法規相關，若無關則禮貌拒絕回答。

-Gemma 3 4B-it：使用 unsloth/gemma-3-4b-it 作為回答生成的語言模型。

-Gradio 介面：提供一個乾淨、專業且支援深淺色模式的 Web UI。

-內建評估：app.py 內含一個使用 qa_pairs（Q&A）來評估 RAG 系統（僅法條索引）回答準確度的評分機制。


資料準備

您需要準備以下兩個檔案，並將它們放在 app.py 相同的目錄下：

labor_law_articles

一個純文字檔案 (.txt 或無副檔名)。

內容為法條原文，一行代表一條法規或一個段落。

此檔案將被用於建立 FAISS 和 BM25 索引。

labor_law_qa.docx

一個 Microsoft Word (.docx) 檔案。

內容為 Q&A 問答對，用於 K-Fold 評估。

格式必須嚴格遵守：

Q：勞工每天最長可以工作幾小時？
A：根據勞動基準法第 XX 條，勞工每日正常工作時間不得超過 8 小時...
A：（可選的第二行答案）
Q：加班費應該如何計算？
A：...


此檔案僅用於評估，不會被加入 Gradio 應用程式的索引中。

如何使用

1. 執行 Gradio 聊天機器人 (預設)

在 app.py 檔案頂部，確認 RUN_EVALUATION = False (預設應設為 False 或直接啟動)：

# app.py (頂部)
RUN_EVALUATION = False


然後在終端機執行：

python app.py


程式將會載入模型、建立索引，然後啟動 Gradio Web 服務。您會看到一個本地 URL (如 http://127.0.0.1:7860) 和一個公開分享的 URL。

2. 執行 K-Fold 評估 (可選)

如果您想在啟動 App 之前，先評估 RAG 系統（僅法條索引）回答 Q&A 的能力：

在 app.py 檔案頂部，將 RUN_EVALUATION 改為 True：

# app.py (頂部)
RUN_EVALUATION = True


執行 app.py：

python app.py


程式將會先執行 K-Fold 交叉驗證流程，在終端機中印出每個 Fold 的分數以及最終的平均語意相似度。

評估完成後，Gradio 應用程式將會接著啟動。
