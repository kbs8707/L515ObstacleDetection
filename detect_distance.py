import cv2
import pyrealsense2 as rs
import numpy as np

from realsense_depth import *

point = (400, 300)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

# Initialize Camera Intel Realsense
dc = DepthCamera()

# Create mouse event
cv2.namedWindow("depth frame")
cv2.namedWindow("post process")
cv2.setMouseCallback("depth frame", show_distance)
cv2.setMouseCallback("post process", show_distance)

while True:
    ret, depth_frame, color_frame, depth_preprocess, postprocess, preprocess_color, postprocess_color = dc.get_frame()

    # Show distance for a specific point
    cv2.circle(depth_frame, point, 4, (0, 0, 255))
    # distance = depth_frame[point[1], point[0]] / 4
    # distance_processed = postprocess[point[1], point[0]] / 4

    # cv2.putText(preprocess_color, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    # cv2.putText(postprocess_color, "{}mm".format(distance_processed), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # cv2.imshow("depth frame", preprocess_color)
    # cv2.imshow("post process", postprocess_color)

    postprocess = dc.removeFloor(postprocess, 9)
    postprocess = dc.setMaxDistance(postprocess, 10000)

    # labels = dc.connectedComponent(postprocess, 40)
    # lbls, lbl_count = np.unique(np.array([i for i in labels.values()]),return_counts=True)

    # uf = UFarray()
    # croppedLabels = []
    # croppedLblsWithCentroids={} 
    
    # for i in range(len(lbls)):
    #     if(lbl_count[i]>100):
    #         croppedLabels.append(lbls[i])
    #         centroids,width,height = uf.getStatsOfLabel(postprocess, lbls[i], labels)
    #         if(lbl_count[i] not in croppedLblsWithCentroids):
    #             newLblCount=lbl_count[i] + i * 0.001
    #             croppedLblsWithCentroids[newLblCount] = centroids, (width,height),lbls[i]
    labelled = dc.CCM(postprocess)
    box = dc.drawBox(labelled, depth_preprocess)

    width = int(postprocess.shape[1] * 4)
    height = int(postprocess.shape[0] * 4)
    dim = (width, height)
    
    # resize image
    postprocess = cv2.resize(postprocess, dim, interpolation = cv2.INTER_AREA)
    labelled = cv2.resize(labelled, dim, interpolation = cv2.INTER_AREA)
    box = cv2.resize(box, dim, interpolation = cv2.INTER_AREA)
    color = cv2.resize(color_frame, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("depth frame", depth_preprocess)
    cv2.imshow("post process", postprocess)
    cv2.imshow("labelled", labelled)
    cv2.imshow("box", box)
    cv2.imshow("RGB", color)
    key = cv2.waitKey(1)
    if key == 27:
        break
