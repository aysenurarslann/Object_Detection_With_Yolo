import cv2
from ultralytics import YOLO
import numpy as np
from sort.sort import Sort

# Çizgiyi geçişleri saymak için sayaç
crossing_counter = 0
crossed_ids = set()

def draw_boxes(frame, objects, line_start, line_end):
    global crossing_counter, crossed_ids

    for obj in objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Nesnenin merkezi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Nesnenin merkezi çizgiyi geçiyor mu kontrol et
        if obj_id not in crossed_ids and cy < line_start[1]:
            crossed_ids.add(obj_id)
            crossing_counter += 1

    # Çizgiyi çiz
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # Sayacı ekrana yaz
    cv2.putText(frame, f'Counter: {crossing_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame

def process_video(input_path, output_path, model):
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternatif olarak 'XVID' veya 'X264' deneyebilirsiniz
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)  # SORT algoritması parametreleri
    frame_index = 0

    # Çizgi koordinatlarını tanımla (görüntünün ortasından biraz yukarıda)
    line_start = (0, height // 2 - 50)
    line_end = (width, height // 2 - 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            if confidence > 0.5:  # Güven eşiği
                detections.append([x1, y1, x2, y2, confidence])  # [x1, y1, x2, y2, confidence]

        if len(detections) > 0:
            detections = np.array(detections)
            tracked_objects = tracker.update(detections)
            frame = draw_boxes(frame, tracked_objects, line_start, line_end)
        else:
            tracked_objects = tracker.update()  # Boş güncellemelerle tracker'ı güncelle

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

if __name__ == "__main__":
    model = YOLO('yolov8n.pt')  # YOLOv8 nano modelini yükleyin
    input_video = "input.mp4"   # Giriş video dosyası
    output_video = "output.mp4" # Çıkış video dosyası

    process_video(input_video, output_video, model)
