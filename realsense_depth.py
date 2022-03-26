from pkgutil import get_data
import pyrealsense2 as rs
import numpy as np
import cv2
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt

from ufarray import *

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        colorizer = rs.colorizer()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # depth_image = self.setMaxDistance(depth_image, 1500)

        # filter_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.07), cv2.COLORMAP_JET)
        filter_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
        temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise
        decimation = rs.decimation_filter()


        #spatial filtering
        decimation.set_option(rs.option.filter_magnitude, 4)
        filtered = decimation.process(depth_frame)

        filtered = spat_filter.process(filtered)

        filtered = temp_filter.process(filtered)

        # hole_filling = rs.hole_filling_filter()
        # filtered = hole_filling.process(filtered)
        
        # filtered = self.setMaxDistance(np.asanyarray(filtered.get_data()), 1500*4)

        # filtered = np.asanyarray(filtered.get_data())
        preprocess_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        postprocessed_color = np.asanyarray(colorizer.colorize(filtered).get_data())
        # filtered = np.round(filtered/4)

        if not depth_frame or not color_frame:
            return False, None, None
        # return True, depth_image, color_image, filter_image, processed_image
        return True, depth_image, color_image, depth_image, np.asanyarray(filtered.get_data()), preprocess_color, postprocessed_color
        

    def release(self):
        self.pipeline.stop()

    def setMaxDistance(self, map, max):
        localmax = np.amax(map)
        res, out = cv2.threshold(map, max, localmax, cv2.THRESH_TOZERO_INV)

        return out

    def removeFloor(self, map, threshold):
        height = map.shape[0]
        width = map.shape[1]
        cy = int(height/2)  #center y
        fy = 580  #focal length y
        z = 5000  #initial min depth
    
        croppedRow = map[-1][np.nonzero(map[-1])]
        if (len(croppedRow)>0):
            z = stats.mode(croppedRow).mode[0]      #initial floor depth
        else:
            z = 75
    
        Y = z*(height-cy)/fy                         #actual y of floor
        Y = Y-threshold
        
        for i in range(height-1,cy-1,-1):
            z1 = (Y*fy)/((i+1)-cy)                   #depth of Y height
    
            for j in range(width):             
                if (map[i][j] > z1):
                    map[i][j] = 0
        return map

    def CCM(self, img):
        img = np.uint8(img)
        # Converting those pixels with values 1-127 to 0 and others to 1
        # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        # Applying cv2.connectedComponents() 
        num_labels, labels = cv2.connectedComponents(img)
        num = labels.max()

        for i in range(1, num+1):
            pts =  np.where(labels == i)
            if len(pts[0]) < 50:
                labels[pts] = 0

        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0
        
        return labeled_img

    def drawBox(self, img, raw):
        img = np.uint8(img)
        
        rawRes = raw.copy()
        rawRes = cv2.cvtColor(rawRes,cv2.COLOR_GRAY2BGR)

        result = img.copy()

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            cv2.rectangle(result, (x, y), (x+w, y+h), 80000, 2)
            # print("x,y,w,h:",x,y,w,h)
            cv2.rectangle(rawRes, (x*4, y*4), (x*4+w*4, y*4+h*4), 80000, 2)
        return rawRes

        # val = np.unique(label)
        # boxUpperLeft = np.zeros(val.size)
        # for i in range(len(label)):
        #     for j in range(len(label[i])):
        #         if label[i][j] != 0:
        #             if boxUpperLeft[val.index(label[i][j])] == 0:
        #                 boxUpperLeft[val.index(label[i][j])] = 
