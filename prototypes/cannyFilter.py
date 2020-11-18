import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r'C:\pytong\ball_tracker\frame1108.jpg',0)
img = cv2.medianBlur(img,3)
edges = cv2.Canny(img,100,200)

cv2.imwrite(r'C:\pytong\ball_tracker\CannyFilter.jpg', edges)


