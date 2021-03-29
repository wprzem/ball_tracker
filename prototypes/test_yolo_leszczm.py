import cv2
import time
import numpy as np
from math import sqrt
tic = time.perf_counter()
net = cv2.dnn.readNet("yolov3_training_last(2).weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
vel = 0
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# cap = cv2.VideoCapture('../images/movie.avi')
cap = cv2.VideoCapture('../images/VID_20201120_190736361~2.mp4')

detected_file = open("detected_ball.txt",'a')
frame = 0 
prev_found = None
while(cap.isOpened()):
    ret, img = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)

    if img is not None:
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        #img = cv2.flip(img, 0)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
                cv2.circle(img, (x+int(w/2),y+int(h/2)), int((w+h)/4) ,(0,0,255))
                if prev_found and frame - prev_found[0] < 10:
                    dp = sqrt((prev_found[1]- x+int(w/2))**2 + (prev_found[2] - y+int(h/2))**2)
                    dm = dp * 0.21*2/(prev_found[3]+int((w+h)/4))
                    vel = dm * fps        
                prev_found = (frame, x+int(w/2),y+int(h/2), int((w+h)/4) )
               #detected_file.write(str(frame) + " " + str( x+int(w/2)) + " " + str(y+int(h/2)) + " " + str(int((w+h)/4)) + "\n")
                
        cv2.putText(img, "velocity: " + str(vel), (0, 30), font, 1, (0, 255, 0), 1)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    frame += 1
    

toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds")
cap.release()
cv2.destroyAllWindows()