import cv2
import numpy as np
import torch
import os
from fastapi import FastAPI, Request, Response, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

app = FastAPI()

# 檢查 GPU
device = "0" if torch.cuda.is_available() else "cpu"
print(f"伺服器運行裝置: {'GPU (CUDA)' if device == '0' else 'CPU'}")

# 載入 YOLO 模型
model = YOLO("weights/best.pt")

# 全域變數：暫存最新的一張手機鏡頭畫格
latest_mobile_frame = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/mobile", response_class=HTMLResponse)
async def mobile_page(request: Request):
    with open("templates/mobile.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload_frame")
async def upload_frame(data: str = Body(...)):
    global latest_mobile_frame
    try:
        if "," in data:
            header, encoded = data.split(",", 1)
            img_bytes = __import__('base64').b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                latest_mobile_frame = frame
    except Exception as e:
        print(f"上傳畫格錯誤: {e}")
    return {"status": "ok"}

def generate_frames(source: str):
    global latest_mobile_frame
    if source == "mobile":
        while True:
            if latest_mobile_frame is not None:
                frame = latest_mobile_frame.copy()
                results = model.predict(frame, conf=0.5, verbose=False, device=device)
                annotated_frame = results[0].plot()
                
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for Mobile Camera...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            import time
            time.sleep(0.03)
    else:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = model.predict(frame, conf=0.5, verbose=False, device=device)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

@app.get("/video_feed")
async def video_feed(source: str = "test.mp4"):
    return StreamingResponse(generate_frames(source), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    import subprocess
    
    cert_file = "cert.pem"
    key_file = "key.pem"
    
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("正在自動產生 SSL 憑證...")
        from datetime import datetime, timedelta
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"localhost")])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
            critical=False,
        ).sign(private_key, hashes.SHA256())

        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        print("SSL 憑證產生成功！")

    print("啟動 HTTPS 伺服器中...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, ssl_keyfile=key_file, ssl_certfile=cert_file, reload=True)