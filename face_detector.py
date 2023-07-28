import cv2

face_casc_path = "Source\\face.xml"
smile_casc_path = "Source\\smile.xml"
eye_casc_path = "Source\\eye.xml"

faceCascade = cv2.CascadeClassifier(face_casc_path)
smileCascade = cv2.CascadeClassifier(smile_casc_path)
eyeCascade = cv2.CascadeClassifier(eye_casc_path)

label = ['Face', 'Eye', 'Smile']

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (ex, ey, ew, eh) in eyes:
            center = (int(ex + 0.5 * ew), int(ey + 0.5 * eh))
            radius = int(0.3 * (ew + eh))  
            cv2.circle(roi_color, center, radius, (0, 0, 255), 2)

        # Menampilkan teks Face, Eye, dan Smile pada frame yang sesuai
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        for (ex, ey, ew, eh) in eyes:
            center = (int(ex + 0.5 * ew), int(ey + 0.5 * eh))
            radius = int(0.3 * (ew + eh))
            cv2.putText(frame, "Eye", (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        for (sx, sy, sw, sh) in smiles:
            cv2.putText(frame, "Smile", (x + sx, y + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('FACE DETECTOR', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()

cv2.destroyAllWindows()