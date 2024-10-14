import easyocr
import cv2
import matplotlib.pyplot as plt

reader=easyocr.Reader(['en'])

image_path='the-data-analysis-process-1.jpg'
image=cv2.imread(image_path)

image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

