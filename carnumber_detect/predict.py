import cv2
import tensorflow as tf
import numpy as np

from carnumber_detect.model import LPRNet
from carnumber_detect.loader import resize_and_normailze
import carnumber_detect.carnumber


classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
              "가", "나", "다", "라", "마", "거", "너", "더", "러",
              "머", "버", "서", "어", "저", "고", "노", "도", "로",
              "모", "보", "소", "오", "조", "구", "누", "두", "루",
              "무", "부", "수", "우", "주", "허", "하", "호"
              ]
tf.compat.v1.enable_eager_execution()
net = LPRNet(len(classnames) + 1)
net.load_weights("carnumber_detect/saved_models/weights_best.pb")  #가중치 로드

def carnumber_detection(img):
    height, width, channel = img.shape
    carnumber = []
    carnumber_candidate,plate_infos= carnumber_detect.carnumber.carnumber_feature(img)  #차량에서 번호판 7~8글자가 연속으로 이어진 것을 찾게함
    #print(plate_infos)
    a = 0
    chars = ""
    for i in carnumber_candidate:
        try:
            img_rotation =cv2.warpAffine(img, M=i, dsize=(width, height))
            img_crop = img_rotation[plate_infos[a]['y']:plate_infos[a]['y'] + plate_infos[a]['h'], plate_infos[a]['x']:plate_infos[a]['x'] + plate_infos[a]['w']]
            a = a + 1
            x = np.expand_dims(resize_and_normailze(img_crop), axis=0)#번호판을 찾게되면 이미지를 정규화함
            chars = net.predict(x, classnames)  # 번호판에 글씨를 찾게함
            chars =''.join(chars)
        except:
            pass
        has_digit = False
        if len(chars) >=7 and len(chars) <= 8:
            for c in chars:
                if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                    if c.isdigit():
                        has_digit = True
            if has_digit ==True:
                cv2.imshow('carnumber',img_crop)
                cv2.waitKey(1)
                return chars


#carnumber_detection(img = cv2.imread("carnumber_detect/test_images/20.jpg"))