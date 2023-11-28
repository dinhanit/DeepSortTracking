from utils import load_session
from Predict import prediction
import cv2,requests
from tracker import Tracker
from configs import *

video_path = "sample/dance.mp4"

# video_path = 0

cap = cv2.VideoCapture(video_path)
cap.set(3,1920)
cap.set(4,1080)

ret, frame = cap.read()

session = load_session(PATH_MODEL)

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.05
    iou_thres = 0.3

cfg = CFG()

tracker = Tracker()

detection_threshold = 150

while ret:
    ret, frame = cap.read()
    results = prediction(
        session=session,
        image=frame,
        cfg=cfg
    )    
    detections = []
    for r in results:
        x1, y1, x2, y2, score, class_id = r
        if class_id != 0:
            continue
        if score < detection_threshold:
            continue
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        class_id = int(class_id)
        detections.append([x1, y1, x2, y2, score])
        
    if len(detections)>0:
        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(x1), int(y1)-10)
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2

            image = cv2.putText(frame,str(track_id), org, font,fontScale, color, thickness, cv2.LINE_AA)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
