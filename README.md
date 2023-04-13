# YOLOv7_Face_Mask_Detection
Object Detection project created to detect face mask using YOLOv7 trained on a custom dataset 

<img src="face-mask.gif"/>

### Dataset and Preperation


https://www.kaggle.com/datasets/andrewmvd/face-mask-detection


All 853 images were manually annotated using labelimg, two labels were used to classify the images, "Mask" and "No Mask".

The dataset containing the images and labels was split into train/test/val using 

```
python split_dataset.py --folder face_mask --train 80 --validation 10 --test 10 --dest face_mask_dataset
```
split_dataset.py provided by https://github.com/pHidayatullah/yolov7

### Yolo Training
Training was performed over 300 epochs and a batch size of 8 using google colab in the YOLOv7 Training.ipynb file.

```
# Train
!python train.py --batch-size 8 --device 0 --data data/face-mask.yaml --img 640 640 --cfg cfg/training/yolov7-face_mask.yaml --weights yolov7_training.pt --name yolov7-face-mask --hyp data/hyp.scratch.custom.yaml --epochs 300
```



### YOLOv7 Performance Measurement

```
python test.py --weights runs/train/yolov7-face-mask4/weights/best.pt --batch-size 2 --device 0 --data data/face-mask.yaml --img 640 --conf-thres 0.01 --iou 0.5 --name yolov7-face-mask-val --task val
```


![](screenshots\test_performance.png)


### Test Batch 0 results

![](runs\test\yolov7-face-mask-val2\test_batch0_pred.jpg)

### Confusion Matrix

![](runs\test\yolov7-face-mask-val2\confusion_matrix.png)

```
python test.py --weights runs/train/yolov7-face-mask4/weights/best.pt --batch-size 2 --device 0 --data data/face-mask.yaml --img 640 --conf-thres 0.01 --iou 0.5 --name yolov7-face-mask-test --task test
```


![](screenshots\test_performance.png)


### Test Batch 0 results

![](runs\test\yolov7-face-mask-test\test_batch0_pred.jpg)

#### Confusion Matrix

![](runs\test\yolov7-face-mask-test\confusion_matrix.png)