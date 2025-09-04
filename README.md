# 人物行為偵測系統 (Abnormal Behavior Detection)

本專案利用 **YOLOv8** 模型與 **Raspberry Pi** 攝影機串流，實現人物異常行為偵測，並整合 **Telegram Bot** 與 **Google Gemini** 語言模型，提供即時通知與事件摘要，及提供github介面即時畫面，最後將結果儲存至個人資料庫。

---

## 📌 功能介紹
- 📷 **即時影像串流**：由 Raspberry Pi 3 負責攝影機串流 
- 📷 **即時查看影像串流**：透過 FastAPI 提供 API，於前端網頁即時觀看
- 🤖 **行為辨識**：使用自行訓練的 YOLOv8n 模型（best.pt，訓練 30 epochs），可辨識站立、奔跑、坐下、睡覺等行為。當信心值 > `0.5` 時，透過 **Telegram Bot** 發送通知
- 📝 **事件摘要**：結合 **Google Gemini** 大語言模型，將事件重點整理  
- 💾 **資料存儲**：偵測結果與摘要內容存入個人資料庫  

---

## 🔧 技術架構
- **硬體**  
  - Raspberry Pi 3 (影像串流)  
  - 電腦主機 (主力運算與 YOLO 偵測)  

- **軟體與框架**  
  - [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (行為偵測)  
  - OpenCV (影像處理與串流)  
  - FastAPI (後端串流與 API)  
  - Telegram Bot API (事件通知)  
  - Google Gemini (事件摘要)  
  - MySQL (資料儲存)  

---

## 📂 資料來源
- **模型訓練模組**： [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- **行為資料集**： [Abnormal Behavior of Person (Roboflow)](https://universe.roboflow.com/project-s41nz/abnormal-behavior-of-person)  

---

## 🚀 系統流程
1. Raspberry Pi 將攝影機影像串流至主機  
2. 主機透過 YOLOv8 進行行為偵測  
3. 當信心值 > `0.5` 且屬人物符合行為特徵(站立、坐立、睡覺等) → 觸發通知  
4. 通知透過 Telegram Bot 發送通知  
5. Gemini 語言模型整理異常事件摘要  
6. 偵測與摘要結果存入 MySQL 資料庫  

---

## 📊 架構圖 (Mermaid)
```mermaid
flowchart TD
    A[Raspberry Pi 3] -->|影像串流| B[主機 YOLOv8 偵測]
    B --> C{行為判斷}
    C -->|符合特徵描述| E[Telegram Bot 通知]
    E --> F[Gemini 事件摘要]
    F --> G[(MySQL 資料庫)]