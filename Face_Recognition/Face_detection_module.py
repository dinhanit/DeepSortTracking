import cv2
import mediapipe as mp
import time
import os
import numpy as np

#[[276.81801327 309.08723465 500.57483001 595.94007038   0.99825972]]
class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.FaceDetection = self.mpFaceDetection.FaceDetection(model_selection=1)
    
    def find_face(self, img, draw = True):
        """
        Detect faces in the given image.

        Args:
        - img (numpy.ndarray): Input image.
        - draw (bool): Flag to draw the detected faces on the image.

        Returns:
        - numpy.ndarray: Array containing detected faces' coordinates.
        """
        try:
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            self.results = self.FaceDetection.process(imgRGB)
            boxs = []
            
            if self.results.detections:
                for id,detection in enumerate(self.results.detections):
                    # self.mpDraw.draw_detection(img, detection)
                    if int(detection.score[0]*100) >= 90:
                        boxC = detection.location_data.relative_bounding_box
                        ih, iw, ic = img.shape
                        x = boxC.xmin*iw
                        y = boxC.ymin * ih
                        w = boxC.width * iw
                        h = boxC.height * ih
                        box = boxC.xmin*iw, boxC.ymin * ih, \
                        boxC.width * iw + boxC.xmin*iw, boxC.height * ih + boxC.ymin * ih
                        for i in box:
                            boxs.append(i)
                        boxs.append(detection.score[0])
                        boxs = np.array([boxs])
                        
            return boxs
        except:
            pass
