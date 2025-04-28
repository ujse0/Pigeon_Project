import cv2
from ultralytics import YOLO
from copy import deepcopy

# from playsound import playsound
import threading
import time 
import pygame
model = YOLO("C:/Users/joons/workspace/final_project/runs/mixed/mixed/weights/best.pt")

video_path = "C:/Users/joons/workspace/final_project/github/jongnoB_vid_9_trim.mp4"
mp3_path = "C:/Users/joons/workspace/final_project/github/noise.mp3"


# def play_music():
#     playsound(mp3_path)

def play_music():
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

cap = cv2.VideoCapture(video_path)
font = cv2.FONT_HERSHEY_SIMPLEX
blue_color = (255, 100, 50)
red_color = (50, 50, 255)
threshold = 10


last_play_time = 0
cooldown_sec = 5


while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame_copied = deepcopy(frame)
        results = model.predict(frame, verbose=False)
        result = results[0]
        boxes = result.boxes
        pigeon_number = 0
        
        for box in boxes:
            cls_ = box.cls
            if cls_ == 0:
                pigeon_number += 1
        current_time = time.time()

        if pigeon_number > threshold:
            text = f"Alert! Pigeon number: {pigeon_number}" 
            color = red_color
            
            if current_time - last_play_time > cooldown_sec:
                t = threading.Thread(target=play_music)
                t.start()
                last_play_time = current_time
                
        else:
            text = f"Pigeon number: {pigeon_number}"
            color = blue_color

        for box in boxes:
            cls_ = box.cls
            
            if cls_ == 0:
                xyxy = box.xyxy
                p1 = (int(xyxy[0, 0]), int(xyxy[0, 1]))
                p2 = (int(xyxy[0, 2]), int(xyxy[0, 3]))
                cv2.rectangle(frame_copied, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
        
        cv2.putText(frame_copied, text, (50, 50), font, 1, color, 2, cv2.LINE_4)
        cv2.imshow("YOLOv11 Inference", frame_copied)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()