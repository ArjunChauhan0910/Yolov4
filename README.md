## Yolo v4 for pytorch , tensorflow eager mode and onnx (by trident api)


This code is an adaptation of https://github.com/AllanYiin/YoloV4
### How to install 

trident only support python 3.x     
If you want to use pytorch api, you need pytorch 1.2 or higher
If you want to use tensorflow api (trident version>=0.5.5), you need tensorflow 2.2.0rc0 or higher (because trident tensorflow  api  is pure eager mode "without keras, without static graph"")
You can install it from pip

    pip install tridentx   --upgrade  
    


after installed trident, you can use following syntax to import it and assign the backed.
    
    import os  
    os.environ['TRIDENT_BACKEND'] = 'pytorch'  
    import trident as T  
    from trident import *  

### How to use
1. pytorch_yolo.py: it is just a basic library derived from trident , to define darknet and yolo basic block.      
2. pytorch_darknet.py: we can construction yolo v4 network and load pretrained weights here.    
3. pytorch_infer_yolo4.py: It's a demo to show how to do object detection by yolo v4 model and how trident api to make things easy.
4. tf_yolo.py: it is just a basic library derived from trident , to define darknet and yolo basic block.      
5. tf_darknet.py: we can construction yolo v4 network and load pretrained weights here.    
6. tf_infer_yolo4.py: It's a demo to show how to do object detection by yolo v4 model and how trident api to make things easy.
7. onnx_video_infer.py: it's a demo how to use onnxruntime to infer video near real-time.

[pytorch pretrained model](https://drive.google.com/open?id=1-3FFc38AeySJpO9wKIAj6V9VZl3MITKW)   
[pytorch pretrained state_dict](https://drive.google.com/open?id=1JAmyPZ71cizlDGsFXKuDOP5EE8MxEqJZ)   
[tensorflow pretrained model](https://drive.google.com/open?id=1_ASrZdGbZZFerL1Fe8jEhwvrRzjyaSuo)      
[tensorflow pretrained state_dict](https://drive.google.com/open?id=1RO247iasDCBLgWrMigVmcgTv8ZKLn7ej)   
[onnx pretrained model](https://drive.google.com/open?id=12e9bJ-QnZ0nZnKUiGSiJha7_aJG3EGMQ)   
