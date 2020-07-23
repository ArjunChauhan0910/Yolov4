import os
import time
os.environ['TRIDENT_BACKEND'] =  'tensorflow'
import glob
from trident import *
from tf_yolo import *
from tf_darknet import *


detector=YoloDetectionModel(input_shape=(608,608,3),output=yolo4_body(80, 608))
detector.load_model('Models/pretrained_yolov4_mscoco_tf.pth.tar')



detector.eval()
detector.summary()

#image preprocess
detector.preprocess_flow=[resize((608,608),keep_aspect=True),to_bgr(),normalize(0,255) ]
#detection threshold
detector.detection_threshold=0.4
#iou_threshold
detector.iou_threshold=0.5

is_small_item_enhance=True
#enable small item enhancement
if is_small_item_enhance:
    for module in detector.model.modules():
        if isinstance(module,YoloLayer):
            module.small_item_enhance=True



#freeze model
detector.trainable=False
#evaluation
detector.eval()


#setting classname
with open( 'pretrained/modified_classes.txt', 'r', encoding='utf-8-sig') as f:
    labels = [l.rstrip() for l in f]
    detector.class_names = labels
#if u want you also can set detector.palette (list of color)


cam = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = detector.infer_then_draw_single_image(frame, verbose=True)
    img =  array2image(img)
    cv2.imshow('Detected',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()
