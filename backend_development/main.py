import os
import time
import queue
import threading
import numpy as np
from dotenv import load_dotenv

import cv2
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from ultralytics import YOLO
import telebot
import google.generativeai as genai

#隱私設定
load_dotenv()
DB_USER = os.getenv("db_user")
DB_PASSWORD = os.getenv("db_password")
DB_HOST = os.getenv("db_host")
DATA_URL = os.getenv("DATABASE_URL")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

GENAI_KEY = os.getenv("genai_APIKEY")

COM_IP = os.getenv("API_IP")

MAIN_IP = os.getenv("MAIN_IP")

# FastAPI 初始化 
app = FastAPI()

# CORS 
origins = [
    "https://Kuan-crypto.github.io/Smart-Patrol-Car-Model",  # GitHub Pages 網址
    MAIN_IP,       # 本地開發
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入訓練好的 YOLO 模型 
model = YOLO("best.pt")
model.fuse()

# 影像來源 (樹莓派 MJPEG 串流) 
video = COM_IP
global cap
cap = cv2.VideoCapture(video)

#  MySQL 資料庫初始化設定
DB_URL = DATA_URL
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(50))
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

os.makedirs("Person_abnormality", exist_ok=True)

# LLM 大語言模型設定 (gemini)

genai.configure(api_key = GENAI_KEY)
modelai = genai.GenerativeModel(model_name="gemini-2.5-flash")

# Telegram Bot 初始化設定
telegram_token = TELEGRAM_TOKEN
chat_id = CHAT_ID
bot = telebot.TeleBot(telegram_token)

alert_cooldown = 30   # 每次警告間隔秒數
target = [ "Opp_Direction", "Standing", "Sleeping","Sitting"]
last_alert_time ={
    "Opp_Direction": 0,
    "Standing": 0,
    "Sleeping": 0,
    "Sitting": 0
}   # 紀錄上次發送通知時間

# 控制狀態 如:暫停、播放
is_running = True
is_streaming = True
latest_frame = None
lock = threading.Lock()

# Alert 任務隊列 排除卡頓異常
alert_queue = queue.Queue()

# gemini生成人性化訊息
def generate_humanized_alert(label, confidence):
    response = modelai.generate_content(f"你是一個監控助理。偵測到異常行為：{label}，信心度 {confidence:.2f}。請用你見到的畫面去做判斷。")
    return response.text.strip()

#  YOLO 偵測執行緒分析 
def detection_yolo():
    global latest_frame,is_running,cap
    while True:
        ret, frame = cap.read()
        if not ret:
            print("串接失敗，請稍等或檢查網路")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(video) #嘗試重新登入
            continue

        if is_running:
            results = model.track(frame,persist=True,conf = 0.5,tracker = "bytetrack.yaml",verbose=False)  # YOLOv8 回傳第一個 batch
        
            if results and len(results)>0:
                result = results[0]
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for i, class_id in enumerate(class_ids):
                    class_name = result.names[int(class_id)]
                    conf = float(confs[i])
                    print("偵測到:", class_name)

                    # ====== 偵測異常行為，發送 Telegram ======
                    if class_name in target: 
                        if (time.time() - last_alert_time[class_name]) > alert_cooldown:
                            # 存截圖
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            screenshot_path = f"Person_abnormality/{class_name}_{timestamp_str}.jpg"
                            annotated_frame = result.plot()
                            cv2.imwrite(screenshot_path, annotated_frame)

                            # 放入隊列，非阻塞
                            alert_queue.put({
                                "label": class_name,
                                "confidence": conf,
                                "screenshot": screenshot_path
                            })

                            last_alert_time[class_name] = time.time()

                        # 存入資料庫
                        with SessionLocal() as session:
                            detection = Detection(label=class_name, confidence=conf)
                            session.add(detection)
                            session.commit()

                # 更新最新畫面（含標註）
                with lock:
                    latest_frame = result.plot()

# Alert Worker 線程 
def alert_worker():
    print("alert_worker 線程正在運行")
    while True:
        alert = alert_queue.get()
        if alert is None:
            break
        print("Alert被觸發",alert)
        label = alert["label"]
        confidence = alert["confidence"]
        screenshot_path = alert["screenshot"]

        # 生成人性化訊息
        msg = generate_humanized_alert(label, confidence)

        # 發送 Telegram
        try:
            bot.send_message(CHAT_ID, msg)
            with open(screenshot_path, "rb") as photo:
                bot.send_photo(CHAT_ID, photo)
        except Exception as e:
            print("Telegram 發送失敗:", e)

# MJPEG 串流產生器
def generate_mjpeg():
    global latest_frame
    while True:
        if not is_streaming:
            time.sleep(0.1)
            continue
        if latest_frame is None:
            continue
        _, jpeg = cv2.imencode(".jpg", latest_frame)
        frame_bytes = jpeg.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)

# API 設定
@app.get("/video")
def video_feed():
    """回傳 YOLO 偵測後的 MJPEG 串流"""
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")
#暫停
@app.post("/pause")
def pause():
    global is_streaming
    is_streaming = False
    return {"status": "paused"}
#播放
@app.post("/resume")
def resume():
    global is_streaming
    is_streaming = True
    return {"status": "resumed"}
#查詢異常紀錄
@app.get("/detections")
def get_detections(limit: int = 10):
    """查詢最近 N 筆異常偵測紀錄"""
    session = SessionLocal()
    rows = session.query(Detection).order_by(Detection.timestamp.desc()).limit(limit).all()
    return [
        {"label": r.label, "confidence": r.confidence, "timestamp": r.timestamp.isoformat()}
        for r in rows
    ]

# WebSocket 控制 
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")
            if action == "pause_stream":
                global is_streaming
                is_streaming = False
            elif action == "resume_stream":
                is_streaming = True
            elif action == "pause_detect":
                global is_running
                is_running = False
            elif action == "resume_detect":
                is_running = True
            await ws.send_json({"status": "ok", "action": action})
    except Exception as e:
        print("WebSocket 關閉:", e)
    finally:
        await ws.close()

# 啟動
if __name__ == "__main__":
    threading.Thread(target=detection_yolo, daemon=True).start()
    threading.Thread(target=alert_worker, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8080)