# 使用yolov3-tiny-stoma进行跟踪
# 使用yolov3-tiny-open-close进行气孔状态开闭识别
# 使用距离最小进行开闭状态的匹配
# 为了加快处理，设置间隔帧数，interval越大，跳跃的帧数越多
# 增加结果输出，保存为csv文件
# 增加OCR模块，记录时间信息，并输出到csv文件中
# 增加语义分割模块，识别气孔复合体区域：面积、周长等参数

import cv2
import numpy as np
from sort import *
from tqdm import tqdm
from utils import yolo_detect,segment_img,recognition_number
import sys
import pandas as pd
import paddlehub as hub

#加载已经训练好的模型
# 读取跟踪的yolo-tiny模型参数，只识别气孔个数
weightsPath="yolov3_tiny_stoma/stoma-yolov3-tiny_11000.weights"
configPath="yolov3_tiny_stoma/stoma-yolov3-tiny.cfg"
labelsPath = "yolov3_tiny_stoma/stomata_class.txt"

# 读取开闭状态的yolo-tiny-open-close模型参数
weightsPath1 = "yolov3_tiny_open_close/stoma-yolov3-tiny_40000.weights"
configPath1 ="yolov3_tiny_open_close/stoma-yolov3-tiny.cfg"
labelsPath1 = "yolov3_tiny_open_close/stomata_class.txt"

# 加载移动端预训练模型
ocr = hub.Module(name="chinese_ocr_db_crnn_server")

# 超参数
interval = 1000 #间隔帧数，改
video = cv2.VideoCapture("test/20210908-1-4leaves-water-yangmai16-moon1-times550.avi")#输入视频路径，改

# 初始化帧数值i
i = 0
# 初始化结果储存字典
csv_result = []

bar = tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT))
# Exit if video not opened.
if not video.isOpened():
    print
    "Could not open video"
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print
    'Cannot read video file'
    sys.exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
result = cv2.VideoWriter('results/20210908-1-4leaves-water-yangmai16-moon1-times550.mp4',fourcc, 30.0, (frame.shape[1], frame.shape[0])) #输出视频路径，改

mot_tracker = Sort()

while True:
    ok, frame = video.read()
    bar.update(1)
    if ok:
        if i%interval == 0:
            # 找到日期在原图上的坐标
            date_image = frame[10:45,5:295]
            cv2.rectangle(frame, (10, 5), (295, 45), (255, 255, 255), 2)
            #用飞桨OCR来识别日期
            now_time = recognition_number(date_image,ocr)

            #语义分割，识别气孔复合体区域,此时是灰度图像
            seg_img = segment_img(frame)
     
            # 二值化
            retval, binary_img = cv2.threshold(seg_img, 50, 255, cv2.THRESH_BINARY)
            # cv2.imshow('binary',binary_img)
            # cv2.waitKey(0)

            # sort            
            match_result = yolo_detect(frame,weightsPath,configPath,labelsPath)
            match_result1 = yolo_detect(frame,weightsPath1,configPath1,labelsPath1)
            # print("match_result1:",match_result1)
            # print("\n")
            match_result = np.array(match_result)
            trackers = mot_tracker.update(match_result)
            for tracker in trackers:
                # print("tracker:",tracker)
                color_tracker = (255,255,0)
                cv2.rectangle(frame, (int(tracker[0]), int(tracker[1])), (int(tracker[2]), int(tracker[3])), color_tracker, 2)
                
                # 截图
                img = seg_img[int(tracker[1]):int(tracker[3]),int(tracker[0]):int(tracker[2])]
                #得到轮廓信息
                contours,hierarchy = cv2.findContours(img,cv2.RETR_LIST,2)
                # 判断，如果该区域没有气孔复合体区域，则直接为0
                if contours == []:
                    area = 0
                    perimeter = 0
                else:    
                    #高级函数的用法，key直接调用函数，取面积最大的轮廓
                    cnt = sorted(contours,key=cv2.contourArea)[-1]
                    #计算轮廓所包含的面积
                    area = cv2.contourArea(cnt)
                    # print("面积：",area)

                    #计算轮廓的周长
                    # 第二参数可以用来指定对象的形状是闭合的(True),还是打开的(一条曲线)。
                    perimeter = cv2.arcLength(cnt,True)
                    # print("周长：",perimeter)

                    # frame_draw = cv2.drawContours(frame, [cnt], -1, (255,255,255), 1)#把所有轮廓画出来

                # 获取classID
                distance1 = 100000
                for match_bbox in match_result1:
                    distance2 = (match_bbox[0]-tracker[0])**2+(match_bbox[1]-tracker[1])**2
                    if distance2 < distance1:
                        distance1 = distance2
                        classID = match_bbox[-1]
                if classID == 0:
                    classID_name = 'close'
                else:
                    classID_name = 'open'

                # print("ID:",classID)
                # 在屏幕上显示气孔信息
                text = "ID{}:{}".format(str(int(tracker[4])),classID_name)
                cv2.putText(frame, text, (int(tracker[0]), int(tracker[1]) - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,220,0), 2)
                text1 = "S:{}/C:{}".format(int(area),int(perimeter))
                cv2.putText(frame, text1, (int(tracker[0]), int(tracker[3]) + 15),cv2.FONT_HERSHEY_SIMPLEX, 0.4,color_tracker, 2)
                result_dic = {'time1':now_time,'time2':now_time[0],'frame_number':i,
                                'x1':int(tracker[0]),'y1':int(tracker[1]),'x2':int(tracker[2]),'y2':int(tracker[3]),
                                'Area':int(area),'Perimeter':int(perimeter),
                                'ID':int(tracker[4]),'classID':classID}
                csv_result.append(result_dic)

            # 语义分割mask_img和img混合
            seg_img = cv2.cvtColor(seg_img,cv2.COLOR_GRAY2BGR)
            mixed_img = cv2.addWeighted(frame,1,seg_img,1,0)
            # result.write(mixed_img)
            # cv2.imshow('frame', mixed_img)

            result.write(mixed_img)
            # cv2.imshow('frame', frame)
            cv2.imshow('mixed_img',mixed_img)
            
        else:
            pass         

        i += 1
        if cv2.waitKey(3) == 27:
            break
    else:
        break 

df = pd.DataFrame(csv_result)
df.to_csv('results/20210908-1-4leaves-water-yangmai16-moon1-times550.csv',index=False) #输出文本文件，改            

result.release()
video.release()
cv2.destroyAllWindows()