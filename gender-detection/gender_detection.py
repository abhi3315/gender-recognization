import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cvlib as cv

# load model
model = load_model("gender_detection_pre.model")

# open webcam
cap = cv2.VideoCapture(0)

classes = ["man", "woman"]

# loop through frames
while cap.isOpened():

    # read from from webcam
    success, frame = cap.read()

    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points for face
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectange over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]  # [[.8775,.95543]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10

        cv2.putText(
            frame,
            label,
            (startX, Y),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Gender Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
