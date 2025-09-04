from fastapi import FastAPI 
from fastapi.responses import StreamingResponse 
from picamera2 import Picamera2 
import cv2 
import time 
import uvicorn 

app = FastAPI() 
#初始化攝影機 
picam2 = Picamera2() 
config = picam2.create_preview_configuration(main={"size": (640, 480)}) 
picam2.configure(config) 
picam2.start() 
time.sleep(2) # 預熱 
# 產生 MJPEG 串流 
def generate_frames(): 
    try: 
        while True: 
            frame = picam2.capture_array() # 將畫面轉成 JPEG bytes 
            _, buffer = cv2.imencode(".jpg", frame) 
            frame_bytes = buffer.tobytes() 
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n") #幀數的封包格式 每個段落內容型態是JPEG圖片格式 最後加入真正JPEG影像二進位資料(tobytes)
    except Exception as e: 
        print("Stream error:", e) # FastAPI route 
@app.get("/stream") 
def stream(): 
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame") #瀏覽器取得新資源後，將顯示的圖片使用新的圖片替代上去 不斷更新圖片。  Sever單向傳送資料給瀏覽器 適合運用在樹莓派鏡頭抓取畫面上
if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000)