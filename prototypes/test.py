import numpy as np
import cv2

cap = cv2.VideoCapture('../images/movie.avi')
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

while(cap.isOpened()):
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    circles = cv2.HoughCircles(
        image=grey,
        method=cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=2000,
        param1=50,
        param2=30,
        minRadius=250,
        maxRadius=280)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()