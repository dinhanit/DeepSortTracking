import face_recognition
import os
import numpy as np
from sklearn import neighbors
import joblib
import pickle
# Step 1: Data Collection and Preprocessing
# Organize your dataset with labeled face images in separate directories for each individual.

# Step 2: Encode Known Faces
def encode_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)

        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) > 0:
                known_face_encodings.append(face_encoding[0])
                known_face_names.append(person_name)

    return known_face_encodings, known_face_names

known_faces_dir = "DataSet/FaceData/processed"
known_face_encodings, known_face_names = encode_known_faces(known_faces_dir)

# Step 3: Train a KNN classifier
classifier = neighbors.KNeighborsClassifier(n_neighbors=5, metric='euclidean')
classifier.fit(known_face_encodings, known_face_names)

# Save known_face_names to a file
with open("Models/known_face_names.pkl", "wb") as file:
    pickle.dump(known_face_names, file)

# Step 4: Save the trained model
model_filename = "Models/face_recognition_model.pkl"
joblib.dump(classifier, model_filename)
print(f"Model saved as {model_filename}")
