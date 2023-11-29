import face_recognition
import os
import numpy as np
from sklearn import neighbors
import cv2
from Face_detection_module import FaceDetector
import joblib
import pickle
from collections import Counter

model_filename = "Models/face_recognition_model.pkl"
# Load the saved model
classifier = joblib.load(model_filename)

# Load known_face_names from the file
with open("Models/known_face_names.pkl", "rb") as file:
    known_face_names = pickle.load(file)

# Step 5: Recognize Faces
def recognize_faces(image):
    """
    Recognizes faces in the given image.

    Args:
    - image: Image to recognize faces from.

    Returns:
    - List[str]: List of recognized face names.
    """
    face_locations = face_recognition.face_locations(image)
    unknown_face_encodings = face_recognition.face_encodings(image, face_locations)
    face_names = []

    for face_encoding in unknown_face_encodings:
        distances, indices = classifier.kneighbors([face_encoding], n_neighbors=5)
        closest_face_names = [known_face_names[i] for i in indices[0]]
        if distances[0][0] < 0.6:  # You can adjust this threshold
            most_common_name = max(set(closest_face_names), key=closest_face_names.count)
            face_names.append(most_common_name)
        else:
            face_names.append("Unknown")

    return face_names

face_detector = FaceDetector()
cap = cv2.VideoCapture(0)

recognized_names_list = []
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Detect faces in the current frame using the FaceDetector
    bounding_boxes = face_detector.find_face(frame, draw=False)
    faces_found = len(bounding_boxes)

    if faces_found > 1:
        # Handling multiple faces scenario
        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 255, 255), thickness=1, lineType=2)
    elif faces_found > 0:
        # Processing single face
        det = bounding_boxes[:, 0:4]
        bb = np.zeros((faces_found, 4), dtype=np.int32)
        for i in range(faces_found):
            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]
            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                recognized_names = recognize_faces(cropped)
                print(recognized_names)
                text_x = bb[i][0]
                text_y = bb[i][3] + 20
                
                try:
                    if len(recognized_names_list) == 10:
                        # Step 1: Count the occurrences of each item
                        item_counts = Counter(recognized_names_list)
                        # Step 2: Find the items that appear the most (best appearing items)
                        max_appearances = max(item_counts.values())
                        best_appearing_items = [item for item, count in item_counts.items() if count == max_appearances]
                        if max_appearances/len(recognized_names_list) >= 0.7 and max_appearances!="Unknown":
                            with open("recognized_name.txt","w+") as f:
                                f.write(recognized_names[0])
                        recognized_names_list = []
                    recognized_names_list.append(recognized_names[0])
                    cv2.putText(frame, recognized_names[0], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0,255), thickness=1, lineType=2)
                except:
                    pass
    # Display the frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
