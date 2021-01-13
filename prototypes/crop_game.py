import numpy as np
import cv2

cap = cv2.VideoCapture('../images/VID_game.mp4')
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
crop_x = -1
crop_y = -1
radius = -1
f = open("results.txt", "w")
while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, channels = frame.shape
    if radius == -1: 



        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (11, 11), 0)     
        circles = cv2.HoughCircles(
            image=grey,
            method=cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=1000,
            param1=90,
            param2=50,
            minRadius=10,
            maxRadius=40)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            crop_x = circles[0,0,0]
            crop_y = circles[0,0,1]
            radius = circles[0,0,2]
            f.write('a' + str(crop_x) + ' ' + str(crop_y) + ' ' + str(radius) + '\n')
        #else:
        #    radius = -1
    else:
        crop_indices = [[max(crop_x-2*radius,0), min(crop_x+2*radius,width)],[max(crop_y-2*radius,0), min(crop_y+2*radius,height)] ]
        cropped = frame[crop_indices[1][0]:crop_indices[1][1], crop_indices[0][0]:crop_indices[0][1]]
        grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (11, 11), 0)
        circles = cv2.HoughCircles(
            image=grey,
            method=cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=1000,
            param1=90,
            param2=50,
            minRadius=10,
            maxRadius=40)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                crop_x = i[0]+crop_indices[0][0]
                crop_y = i[1]+crop_indices[1][0]
                radius = i[2]
                f.write(str(crop_x) + ' ' + str(crop_y) + ' ' + str(radius) + '\n')
                cv2.circle(cropped,(i[0],i[1]),radius,(0,255,0),2)
                cv2.imshow('cropped', cropped)
                cv2.circle(frame,(crop_x,crop_y),radius,(0,255,0),2)
                cv2.circle(frame,(crop_x,crop_y),2,(0,0,255),3)
        else:
            radius = -1
    cv2.imshow('frame', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
f.close()
cv2.waitKey(0)