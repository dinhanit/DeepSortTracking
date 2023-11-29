from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import face_recognition
from typing import List
import joblib
import pickle

app = FastAPI()

# Load the saved model
model_filename = "Models/face_recognition_model.pkl"
classifier = joblib.load(model_filename)

# Load known_face_names from the file
with open("Models/known_face_names.pkl", "rb") as file:
    known_face_names = pickle.load(file)

def recognize_faces(image):
    """
    Recognizes faces in the given image.

    Args:
    - image: Image to recognize faces from.

    Returns:
    - Recognized face names.
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

@app.post("/recognize/")
async def recognize_faces_endpoint(file: UploadFile):
    """
    Endpoint to recognize faces in an uploaded image.

    Args:
    - file: Uploaded file containing an image.

    Returns:
    - Dict[str, List[str]]: Recognized face names.
    """
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        recognized_names = recognize_faces(image)
        return {"recognized_names": recognized_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_recognized_name/")
async def get_recognized_name():
    """
    Endpoint to retrieve the recognized name.

    Returns:
    - str: Recognized name.
    """
    try:
        with open("recognized_name.txt", "r") as f:
            recognized_name = f.read()
        # Clear the content of the file after reading it
        with open("recognized_name.txt", "w") as f:
            f.truncate(0)
        return recognized_name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
