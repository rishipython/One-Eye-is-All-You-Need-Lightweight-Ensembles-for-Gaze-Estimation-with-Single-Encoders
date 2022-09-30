import cv2
video_capture = cv2.VideoCapture(0)

cv2.namedWindow("Window")

while True:
    ret, frame = video_capture.read()
    cv2.imshow("Window", frame)

    #This breaks on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()