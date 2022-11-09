# 形态学计算
# 计算气孔的周长、面积和偏心率

from PIL import Image
import numpy as np
import random
import copy
import os
import cv2
from semantic_nets.unet import mobilenet_unet
import time
from numpy.core.fromnumeric import reshape
import paddlehub as hub

# 加载移动端预训练模型
ocr = hub.Module(name="chinese_ocr_db_crnn_server")

def recognition_number(image,ocr):
    """
    识别图片的数字，并输出为str文本
    image:输入的图片
    ocr:百度飞浆的线上ocr模型，调用方式:
    ocr = hub.Module(name="chinese_ocr_db_crnn_server") 
    """
    image = [image]
    text = []
    results = ocr.recognize_text(
                        images=image,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                        use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                        visualization=False,       # 是否将识别结果保存为图片文件；
                        box_thresh=0.7,           # 检测文本框置信度的阈值；
                        text_thresh=0.5)          # 识别中文文本置信度的阈值；

    for result in results:
        data = result['data']
        save_path = result['save_path']
        for infomation in data:
            text.append(infomation['text'])
    return text

#单张图片测试，测试气孔的开闭
def yolo_detect(image,weightsPath,configPath,labelsPath):

    #初始化一些参数
    LABELS = open(labelsPath).read().strip().split("\n")  #物体类别
    COLORS = [(255,255,0)]#颜色
    boxes = []
    confidences = []
    classIDs = []
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # 获取一帧
    (H, W) = image.shape[:2]
    # 得到 YOLO需要的输出层
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # 从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # 初始化bboxes
    new_bboxes = []

    # 在每层输出上循环
    for output in layerOutputs:
        # 对每个检测进行循环
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # 过滤掉那些置信度较小的检测结果
            if confidence > 0.5:
                # 框后接框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # 边框的左上角
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # 更新检测出来的框
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    # 极大值抑制
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
    #print(idxs)
    if len(idxs) > 0:
        for i in idxs.flatten():
            new_confidence = confidences[i]
            new_classID = classIDs[i]
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            new_bboxes.append([x1,y1,x2,y2,new_confidence,new_classID])

    return new_bboxes

def segment_img(img):
    """
    对opencv格式图像进行语义分割
    识别气孔复合体区域
    """

    # 一些基本参数
    random.seed(0)
    class_colors = [[0,0,0],[255,0,0]]
    NCLASSES = 2
    HEIGHT = 416
    WIDTH = 416

    # 加载模型
    model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    model.load_weights("semantic_weights/last1.h5")

    # 将opencv图像格式转换为PIL.Image的格式
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    # 将图片resize成416*416
    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)

    # 归一化，然后reshape成1，416，416，3
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)

    # 进行一次正向传播，得到（43264，2)
    pr = model.predict(img)[0]
    # print(pr.shape)

    pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT/2), int(WIDTH/2),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
    # 将将PIL.Image图像格式转换为opencv的格式
    seg_img = cv2.cvtColor(np.asarray(seg_img),cv2.COLOR_RGB2GRAY)
    return seg_img

def stomata_area(mask_img):
    """
    计算气孔面积
    """
    area = 0
    w,h = mask_img.shape
    for i in range(w):
        for j in range(h):
            if mask_img[i][j] != 0:
                area += 1
    return area

def stomata_perimeter(mask_img):
    """
    计算气孔周长
    """

    pass

def stomata_eccentricity(mask_img):
    """
    计算偏心率
    """
    pass

if __name__ == "__main__":

    #加载已经训练好的模型
    weightsPath="yolov3_tiny//stoma-yolov3-tiny.weights"
    configPath="yolov3_tiny//stoma-yolov3-tiny.cfg"
    labelsPath = "yolov3_tiny//stomata_class.txt"

    #读入待检测的图像
    image = cv2.imread('test/4.jpg')

    results = yolo_detect(image,weightsPath,configPath,labelsPath)
    print(results)
    for result in results:
        # 在原图上绘制边框和类别
        color = (255,255,20)
        cv2.rectangle(image, (result[0], result[1]), (result[2], result[3]), color, 2)
        text = "{}: {:.2f}".format('stoma',result[-1])
        cv2.putText(image, text, (result[0], result[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()