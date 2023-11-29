import cv2,os
from Face_detection_module import FaceDetector
def GetData():
    """
    Capture facial data using the webcam and save the detected faces.

    This function prompts the user to enter an ID, captures facial data from the webcam,
    detects faces, and saves the detected face images in a specified directory.
    """
    detector = FaceDetector()
    name = input('Enter id: ')
    inf = name
    os.makedirs('DataSet/FaceData/processed/' + inf)
    cap = cv2.VideoCapture(0) # mở kết nối đến camera
    i=0
    j=0
    while True:
        ret, frame = cap.read()
        j+=1
        if i > 100:
            break
        if not ret:
            break
        if j % 5 == 0:
            faces = detector.find_face(frame)
            if len(faces) > 0:
                for face in faces:
                    x, y, w, h ,c = face
                    x = int(x)
                    y= int(y)
                    w = int(w)
                    h = int(h)
                    cv2.rectangle(frame, (x, y, w-x, h-y), (0, 255, 0), 2)

                    if c > 0.8:
                        if w>h:
                            k = w
                        else:
                            k = h
                        d = k//2
                        cropped_face = frame[y:h, x:w]
                        path = 'DataSet/FaceData/raw/'+inf+'/'+str(i)+'.jpg'
                        print(path)

                        cv2.imshow('face', cropped_face)
                        cv2.imwrite(path, cropped_face)
                        i+=1
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Run the function to capture data
GetData()
