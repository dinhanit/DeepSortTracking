import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os

# Import your FaceNetModel definition
from FaceNet import FaceNetModel
from Face_detection_module import FaceDetector

# Load the saved reference embeddings and labels
reference_embeddings = torch.load('Models/reference_embeddings.pth')
reference_labels = torch.load('Models/reference_labels.pth')
# Load the saved model
model = FaceNetModel()
model_load_path = 'Models/facenet_model.pth'
model.load_state_dict(torch.load(model_load_path))
model.eval()

# Define a custom dataset for the reference database
class ReferenceFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for filename in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, target
# Define transformation for the reference dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# Set the data directory for the reference database
reference_data_dir = 'DataSet/FaceData/processed'
# Create a custom dataset for the reference database
reference_dataset = ReferenceFaceDataset(root_dir=reference_data_dir, transform=transform)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# for i in range(len(reference_embeddings)):
#     reference_embeddings[i] = reference_embeddings[i].to(device)
def recognize_face(image):
    global model, recognized_object, reference_embeddings
    # Define the transformation for inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocess the input image
    input_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Compare the new face embeddings with the reference database to recognize the person
    min_distance = float('inf')
    recognized_object = None
    
    with torch.no_grad():
        embeddings = model(input_tensor)
        for i, reference_embedding in enumerate(reference_embeddings):
            distance = torch.norm(embeddings - reference_embedding, p=2)
            if distance < min_distance:
                min_distance = distance
                recognized_object = reference_labels[i]
    print(min_distance)
    if 1==1:
        return recognized_object
    else:
        return None

if __name__ == '__main__':
    # Define a recognition threshold
    # recognition_threshold = torch.Tensor([1.5000])
    face_detector = FaceDetector()

    # Open a connection to the webcam (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Detect faces in the current frame using the FaceDetector
        bounding_boxes = face_detector.find_face(frame, draw=False)
        faces_found = len(bounding_boxes)

        if faces_found > 1:
            cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (255, 255, 255), thickness=1, lineType=2)
        elif faces_found > 0:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]

                    # Perform face recognition on the cropped face
                    recognized_object = recognize_face(cropped)
                    if recognized_object:
                        recognized_object = reference_dataset.classes[recognized_object]
                    else:
                        recognized_object = "Unknown"
                    

                    # Display the bounding box and confidence
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20

                    cv2.putText(frame, recognized_object, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0,255), thickness=1, lineType=2)

        # Display the frame
        cv2.imshow('Webcam Face Recognition', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
