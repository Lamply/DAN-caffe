import numpy as np
import cv2
import utils
from time import time
from FaceAlignment import FaceAlignment


net_work_path = './prototxt/stage2/deploy.prototxt'
weight_path = './output/stage2/DAN_s2_iter_196000.caffemodel'

with np.load("./DAN.npz") as f:
    meanImg = f["meanImg"]
    stdDevImg = f["stdDevImg"]
    initLandmarks = f["initLandmarks"]

model = FaceAlignment(net_work_path, weight_path, 112, 112, 1, 2)
model.setNorm(meanImg, stdDevImg, initLandmarks)

vidIn = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("./data/haarcascade_frontalface_alt.xml")

reset = True
landmarks = None

print ("Press space to detect the face, press escape to exit")

while True:
    vis = vidIn.read()[1]
    if len(vis.shape) > 2:
        img = np.mean(vis, axis=2).astype(np.uint8)
    else:
        img = vis.astype(np.uint8)

    if reset:
        rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
        if len(rects) > 0:
            minX = rects[0][0]
            maxX = rects[0][0] + rects[0][2]
            minY = rects[0][1]
            maxY = rects[0][1] + rects[0][3]
            cv2.rectangle(vis, (minX, minY), (maxX, maxY), (255, 0, 0))
            initLandmarks = utils.bestFitRect(None, model.initLandmarks, [minX, minY, maxX, maxY])
            reset = False

            landmarks = model.processImg(img[np.newaxis], initLandmarks, video_flag=0)
            landmarks = landmarks.astype(np.int32)
            for i in range(landmarks.shape[0]):
                cv2.circle(vis, (landmarks[i, 0], landmarks[i, 1]), 2, (0, 255, 0))
    else:
        initLandmarks = utils.bestFitRect(landmarks, model.initLandmarks)
        start = time()
        landmarks = model.processImg(img[np.newaxis], initLandmarks, video_flag=0)
        end = time()
        print("time: " + str((end-start)*1000) + 'ms')
        landmarks = np.round(landmarks).astype(np.int32)

        for i in range(landmarks.shape[0]):
            cv2.circle(vis, (landmarks[i, 0], landmarks[i, 1]), 2, (0, 255, 0))

    cv2.imshow("image", vis)
    # cv2.imwrite("image.jpg", vis)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord(' '):
        reset = True

