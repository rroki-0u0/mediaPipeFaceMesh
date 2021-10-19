import time

import cv2
import mediapipe

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mediapipe.solutions.drawing_utils
mpFaceMesh = mediapipe.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL, drawSpec, drawSpec)
            # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_LEFT_EYEBROW, drawSpec, drawSpec)

            for no, lm in enumerate(faceLms.landmark):
                print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 120), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.putText(img, f'Time: {int(cTime)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.putText(img, f'X: {lm.x}', (440, 270), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.putText(img, f'Y: {lm.y}', (440, 320), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.putText(img, f'Z: {lm.z}', (440, 370), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.putText(img, f'Z: {lm.z}', (440, 370), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.imshow('image', img)
    cv2.waitKey(1)
