import cv2
from ultralytics import YOLO

# 1. 載入模型權重
MODEL_PATH = "F:/py/runs/detect/train-5/weights/best.pt"
print(f"載入模型：{MODEL_PATH} ...")
model = YOLO(MODEL_PATH)

# 2. 測試影片路徑
VIDEO_PATH = "test.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"錯誤：找不到影片 {VIDEO_PATH}")
    exit()

print("\n--- 操作說明 ---")
print("【空白鍵 (Space)】：暫停 / 繼續播放")
print("【D 鍵】：暫停時單步前進一幀 (Step Forward)")
print("【Q 鍵】：退出程式\n")

is_paused = False
frame = None

while cap.isOpened():
    # 非暫停狀態才讀取下一影格
    if not is_paused:
        ret, frame = cap.read()
        if not ret:
            print("影片播放完畢。")
            break

        # 執行推論
        results = model.predict(frame, conf=0.5, verbose=False)
        display_frame = frame.copy()

        if results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clses = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, clses, confs):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names.get(cls_id, str(cls_id))

                if class_name.lower() == "accident":
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"! ACCIDENT ! ({conf*100:.1f}%)"
                    cv2.putText(display_frame, label, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    label = f"{class_name} {conf*100:.1f}%"
                    cv2.putText(display_frame, label, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 暫停時在畫面上加註文字
    render_frame = display_frame.copy()
    if is_paused:
        cv2.putText(render_frame, "[ PAUSED ] Press Space to resume, 'D' to step", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Accident Detection", render_frame)

    # 按鍵監聽邏輯
    key = cv2.waitKey(0 if is_paused else 25) & 0xFF

    if key == ord(' '):  # 空白鍵切換暫停/播放
        is_paused = not is_paused
    elif key == ord('d') and is_paused:  # 暫停時按 d 讀取下一影格
        ret, frame = cap.read()
        if not ret:
            print("影片播放完畢。")
            break
        # 重新針對單幀執行推論
        results = model.predict(frame, conf=0.5, verbose=False)
        display_frame = frame.copy()
        if results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clses = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, cls_id, conf in zip(boxes, clses, confs):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names.get(cls_id, str(cls_id))
                color = (0, 0, 255) if class_name.lower() == "accident" else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
    elif key == ord('q'):  # 按 q 退出
        break

cap.release()
cv2.destroyAllWindows()