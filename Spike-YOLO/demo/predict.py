#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
from cProfile import label
from itertools import count
import time
import math
from tkinter import N
import cv2
import numpy as np
from PIL import Image

from yolo import YOLO
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "VOCdevkit/VOC2007/JPEGImages/"
    dir_save_path   = "示例/结果"

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref,frame=capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm
        import xml.etree.ElementTree as ET

        def parse_obj(xml_path, filename):
           tree=ET.parse(xml_path+filename)
           objects=[]
           for obj in tree.findall('object'):
               obj_struct={}
               obj_struct['name']=obj.find('name').text
               objects.append(obj_struct)
           return objects
        

        l=0.0
        k=0.0
        j=0.0
        pr=0
        labels=[]
        pred=[]
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                img_= Image.open(image_path)
                
                #真是标签
                xml_name=img_name.replace('.png','')
                objects=parse_obj('VOCdevkit/VOC2007/Annotations/',xml_name+'.xml')
                num_objs=0
                for object in objects:
                
                          num_objs+=1
                
               #----------------------------------------------------------------------------------
                n=0
                r_image  ,n  = yolo.detect_image(image,n)
                labels.append(num_objs)
                pred.append(n)

                w=abs(n-num_objs)
                l=l+w
                k=w*w+k
                j=(w/num_objs)*(w/num_objs)+j
                pr=pr+n/num_objs
        

                print('{}:{}'.format('预测',n))
                print('{}:{}'.format('真实',num_objs))
                with open((dir_save_path+".txt"), "a") as new_f:
                     new_f.write('预测'+str(n))
                     new_f.write('真实'+str(num_objs)+'\n')
            
            
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
       


        MAE=l/len(img_names)
        RMSE=math.sqrt(k/len(img_names))
        RMSPE=math.sqrt(j/len(img_names))
        pre=pr/len(img_names)

        labels=np.array(labels).reshape(-1, 1)
        pred=np.array(pred).reshape(-1, 1) 
        R2=r2_score(labels,pred)
   
     
        print('{}:{}'.format('MAE',MAE))
        print('{}:{}'.format('RMAE',RMSE))
        print('{}:{}'.format('RMSPE',RMSPE))
        print('{}:{}'.format('pre',pre))
        print('{}:{}'.format('R2',R2))
        
        
        with open((dir_save_path+".txt"), "a") as new_f:
           
            new_f.write('\n'+"MAE"+":"+str(MAE)+"\n")
            new_f.write("RMSE"+":"+str(RMSE)+"\n")
            new_f.write("RMSPE"+":"+str(RMSPE)+"\n")
            new_f.write("pre"+":"+str(pre)+"\n")
            new_f.write("R2"+":"+str(R2)+"\n")
        

        mode=LinearRegression()
        mode.fit(labels,pred)
        pred_y=mode.predict(labels)
        

        p1=plt.scatter(labels, pred,c='blue', marker='o', edgecolor='white',s=50,label='pre')
        p2=plt.plot(labels,pred_y,label='pre_line',c='green')
        p3=plt.plot([0,150],[0,150],label='y=x line',c='red')
        plt.ylabel("pre_count")#添加横轴标签\n",
        plt.xlabel(' lable_count')#添加纵轴标签\n",
        plt.text(2,150,'R2 ='+str(R2),c='green')
        plt.text(2,144,'MAE ='+str(MAE),c='green')
        plt.text(2,138,'RMSE ='+str(RMSE),c='green')
        plt.text(2,132,'RMSPE ='+str(RMSPE),c='green')
        plt.legend(loc="lower right")
        path='示例/结果/'
        
        if not os.path.exists(path):
           os.makedirs(path)
        plt.savefig(path+'imA1_imA_mul_asyloss_0_1_0.03.jpg')#保存图片
        plt.show()
  
       

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")