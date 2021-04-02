import cv2
import time
from math import sqrt

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("../images/VID_20201120_190736361~2.mp4")
fr_no=0
net = cv2.dnn.readNet("yolov4-obj_best.weights", "yolov4-obj.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
vel = 0 
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
prev_found = None
while cv2.waitKey(1) < 1:
    fps = vc.get(cv2.CAP_PROP_FPS)
    (grabbed, frame) = vc.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame,None, fx=0.4,fy=0.4)
    if not grabbed:
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(frame, box, color, 2)
        
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (box[0]+int(box[2]/2),box[1]+int(box[3]/2)), int((box[2]+box[3])/4) ,(0,0,255))
    if prev_found and fr_no - prev_found[0] < 10:
        dp = sqrt((prev_found[1]- box[0]+int(box[2]/2))**2 + (prev_found[2] - box[1]+int(box[3]/2))**2)
        dm = dp * 0.21*2/(prev_found[3]+int((box[2]+box[3])/4))
        vel = dm * fps        
    prev_found = (fr_no, box[0]+int(box[2]/2),box[1]+int(box[3]/2), int((box[2]+box[3])/4) )
    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "velocity: " + str(vel), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow("detections", frame)
    fr_no += 1