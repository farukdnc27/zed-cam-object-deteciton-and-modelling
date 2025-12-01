import cv2
import numpy as np
import pyzed.sl as sl

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.UNIT_METER
init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE

err = zed.open(init_params)

obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = True
obj_param.enable_segmentation = True
obj_param.detection_model = sl.OBJECT_CLASS.MULIT_CLASS_BOX_MEDIUM

if obj_param.enable_tracking:
    positional_tracking_param = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_param)

err = zed.enable_object_detection(obj_param)

objects = sl.Objects()
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
obj_runtime_param.detection_confidence_threshold = 30

cv2.namedWindow("ZED Object Detection", cv2.WINDOW_NORMAL)

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_objects(objects, obj_runtime_param)
        
        img = sl.Mat() 
        zed.retrieve_image(img, sl.VIEW_LEFT)

        