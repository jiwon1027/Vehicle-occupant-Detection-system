import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from carnumber_detect.predict import carnumber_detection
import socket

import pandas as pd
from sqlalchemy import create_engine
from PIL import Image
import base64
from io import BytesIO
import time
import copy
import timeit
from multiprocessing import Process,freeze_support, Queue, Manager

engine = create_engine('mysql+pymysql://root:111111@113.198.234.49/test', echo=False)  #데이터베이스 접속
a = int(0) #번호판 실패시 db에 저장되는 번호판
check = False # 크롭시 두번째 카메라를 불러오는 용도로 사용 (임계구역을 위한 변수)

        
def socket_multiprocess(conn,picture,check):  #소켓통신을 멀티프로세스를 이용하여 부름
    while True:
        #time.sleep(0.01)
        if picture[2] == True and picture[3] == False:   # picture[2],[3] 을 멀티프로세스
            picture[0] = 0  # (첫번째 카메라 이미지를 없앰 )
            message = '2' # 소켓통신을 통하여 2를보내기 위한 변수
            conn.send(message.encode())
            stringData = 0
            while True:
                try:
                    length = recvall(conn, 16)    #신호 받기
                    if length == None: #신호가 없을때
                        break
                    stringData = recvall(conn, int(length))  #값을 받아서
                    if stringData.decode() == 'a':
                        break
                except:
                    continue
            length = recvall(conn, 16)   #신호 받기
            if length == None: #신호가 없을때
                break
            stringData = recvall(conn, int(length))  #값을 받아서

            data = np.frombuffer(stringData, dtype='uint8') # 변환
            img = cv2.imdecode(data, 1) #이미지로 변환
            picture[1] = img #이미지 삽입
            picture[3] = True #임계구역 해제
        elif picture[2] == False:  #일반적일 때
            picture[1] = 0   #두번째 카메라 이미지 없앰 (멀티프로세스 딜레이를 통해서 이미지가 중복되지 않기 위함)

            length = recvall(conn, 16)
            if length == None:
                break
            stringData = recvall(conn, int(length))

            data = np.frombuffer(stringData, dtype='uint8')
            img = cv2.imdecode(data, 1)
            picture[0] = img


def bbox_size(x1,y1,x2,y2):  # YOLO의 오인식을 줄이기 위한 방안 (예측한 객체의 바운딩박스 크기를 계산함) 두점사이의 거리
    result = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return result
def save_carnumber(frame,person_count,car_plate):  #결과를 db에 저장함
    global a
    buffer = BytesIO()#
    im = Image.fromarray(frame)#
    im.save(buffer, format='jpeg')#
    img_str = base64.b64encode(buffer.getvalue())   #이미지를 string으로 바꿈
    try:
        if car_plate != 0: #차량번호판이 인식이 안됐으면
            img_df = pd.DataFrame({'date': time.strftime('%c', time.localtime(time.time())), 'carnumber': car_plate,'person': person_count,'fine': 100000, 'pic': [img_str]})
        else: #차량번호판이 인식 됐으면
            img_df = pd.DataFrame({'date': time.strftime('%c', time.localtime(time.time())), 'carnumber': a, 'person': person_count,'fine': 100000,'pic': [img_str]})
        img_df.to_sql('test_', con=engine, if_exists='append', index=False)
        a=a+1
    except:
        img_df = pd.DataFrame(
            {'date': time.strftime('%c', time.localtime(time.time())), 'carnumber': '134조1274', 'person': person_count,
             'fine': 100000, 'pic': [img_str]})
        img_df.to_sql('test_', con=engine, if_exists='append', index=False)
#받는함수
def recvall(sock, count): # 소켓통신으로 데이터를 받는 함수
    buf = b''
    while count:
        newbuf = sock.recv(count)  #socket의 recv함수는 데이터를 받을 때 까지 무한대기함
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


limit_line_width = 0  # 트리거 x1 좌표
limit_line_width2 = 1000
# 트리거 x2 좌표
limit_line_high = 500  # 트리거 y1 좌표
limit_line_high2 = 600  # 트리거 y2 좌표

def detect(save_img=False):

    out, source, weights, view_img, save_txt, imgsz = \
    opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size   #https://github.com/ultralytics/yolov5 참조


    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder  #폴더 생성
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    if opt.rotation == 0:
        def on_mouse(event, x, y, flags, param):  # 이미지 트리거를 쉽게 변경하기위한 opencv 좌표 추출 함수

            if event == cv2.EVENT_LBUTTONDOWN:
                global limit_line_high
                print(x, y)
                limit_line_high = y
            if event == cv2.EVENT_RBUTTONDOWN:
                global limit_line_high2
                print(x, y)
                limit_line_high2 =y
    elif opt.rotation == 1:
        def on_mouse(event, x, y, flags, param):  # 이미지 트리거를 쉽게 변경하기위한 opencv 좌표 추출 함수

            if event == cv2.EVENT_LBUTTONDOWN:
                global limit_line_width
                print(x, y)
                limit_line_width = x
            if event == cv2.EVENT_RBUTTONDOWN:
                global limit_line_width2
                print(x, y)
                limit_line_width2 = x



    #차량탑승인원에 필요한 초기값 생성
    person_count = 0  # 사람수 새기
    car_x1, car_y1, car_x2, car_y2 = 0, 0, 0, 0  # 차량 좌표 변경
    crop_image = 0  #이미지 크롭할거 초기화
    crop_image_size =0 #크롭이미지 사이즈 초기화
    center = [0, 0] #바운딩박스의 중앙값 초기화

    cv2.setMouseCallback('im0', on_mouse)  #마우스 이벤트 선언opencv에서
    global check
    check =False
    if opt.receive ==1: #소켓통신을 할 때 .

        HOST = ''
        PORT = 20001

        # TCP 사용
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 서버의 아이피와 포트번호 지정
        s.bind((HOST, PORT))
        print('Socket now listening')
        s.listen(10)

        # 연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
        conn, addr = s.accept()
        print('접속 완료')
        #conn.settimeout(5000)


        socket_img_load = Queue(1)
        check_queue = Queue(1)
        process1 = Process(target=socket_multiprocess, args=(conn, picture,check_queue))  #소켓통신 멀티프로세스 작동
        process1.start()



        while True:

            #start_t = timeit.default_timer() #FPS time 시작
            if picture[2] == False and picture[3] == False:  #picture[2] 가 False면 즉 차량이 인식되어 크롭되지 않을때
                img = picture[0]  #이미지 가져옴
            elif picture[2] == True and picture[3] == True: #크롭이미지 일때
                img = picture[1] #이미지 가져옴
            else:
                continue

             #이미지 처리과정
            try:
                im0s = img.copy()
            except:
                continue
            img = letterbox(img, new_shape=640)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img, augment=opt.augment)[0]  #이미지를 오픈소스인 yolov5를 이용하여 객체의 클래스 예측

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)  #nms 를 이용한 겹치는 바운딩 박스 제거
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0 = '', im0s
                if det is not None and len(det) or check == True:
                    # Rescale boxes from img_size to im0 size
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    if check == True:  # 2번째 카메라가 들어왔을때
                        check = False
                        picture[2] = False
                        picture[3] = False
                        person_count_part = 0
                        result_person = 0  # 최종사람 검출
                        if det is not None and len(det):
                            for *xyxy, conf, cls in reversed(det):  #인식한 물체들의 바운딩박스 좌표와 이름 뽑아옴
                                if names[int(cls)] == 'person': #사람의 객체를 인식했다면
                                    person_count_part = person_count_part + 1 #사람수 세기
                                    label = '%s %.2f' % (names[int(cls)], conf) #객체 이름 추출
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls) ], line_thickness=1)  #이미지에 인식한 물체 그리기
                        print("첫번째카메라 사람:", person_count)
                        print("두번째 카메라 사람수 :", person_count_part)
                        car_plate = carnumber_detection(crop_image)   #차량번호판 인식 함수
                        print("차량번호:",car_plate)
                        if person_count_part > person_count:  #2번째 카메라가 더 많이 사람을 인식할때
                            result_person = person_count_part
                        elif person_count_part == 0 and person_count == 0: #사람을 못찾았을때
                            result_person = 1
                        else:
                            result_person = person_count  #첫번째 카메라가 사람을 더 많이 인식할때

                        print("차량 탑승인원 :",result_person)
                        print()
                        #save_carnumber(crop_image, result_person, car_plate) #차량번호판과 사진, 사람수를 db에저장하는 명령어
                        person_count = 0
                        cv2.imshow('second_camera ', im0)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration
                        cv2.imshow('frist_camera', crop_image)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration
                        break

                    for *xyxy, conf, cls in reversed(det):  # 소켓통신을 하는 메인 카메라의 물체찾기
                        if (names[int(cls)] == 'car' or names[int(cls)] == 'truck' or names[int(cls)] == 'bus') and (
                                int(xyxy[3]) >= limit_line_high and int(xyxy[3] <= limit_line_high2)) \
                                and (int(xyxy[0]) >= limit_line_width and int(xyxy[2]) <= limit_line_width2):  #인식하고 있는 물체가 차량이고 트리거내에 있다면
                            # print(int(xyxy[1]), int(xyxy[3]))

                            crop_image = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]  #이미지를 크롭함
                            crop_image_size = bbox_size(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])) #이미지 사이즈 저장
                            car_x1 = int(xyxy[0]) #차량 이미지 좌표 저장
                            car_y1 = int(xyxy[1])
                            car_x2 = int(xyxy[2])
                            car_y2 = int(xyxy[3])

                            # 차량의 중앙좌표보기
                            center_previous = center.copy()  #이전 중앙좌표
                            height, width, _ = im0.shape #이미지의 가로세로 구하기
                            x1, y1, x2, y2 = max(0, int(xyxy[0])), max(0, int(xyxy[1])), min(width, int(xyxy[2])), min(
                                height, int(xyxy[3]))   # 차량 좌표 저장
                            center = [int((x1 + x2) / 2), int((y1 + y2) / 2)] #차량 중앙값 구해서 저장 

                            # print(center)
                            # print(center_previous)
                            # print()
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                            # print(car_x1,car_y1,car_x2,car_y2)
                        elif (names[int(cls)] == 'car' or names[int(cls)] == 'truck' or names[int(cls)] == 'bus') and int(xyxy[3] > limit_line_high2+20) and center[1] != 0 \
                                and (int(xyxy[0]) >= limit_line_width and int(xyxy[2]) <= limit_line_width2) and \
                            (center_previous[1] > center[1] - 30 and  crop_image_size > 200):   #트리거 내에 있던 차량이 트리거밖으로 넘어갔을때  (차량이 인지범위 넘었을때)
                            picture[2] = True  # 2번쨰 카메라 인식을 위한 True(멀티프로세스를 이용한 소켓통신 함수에서 두번째 카메라의 이미지를 받게함)
                            car_x1 = 0   #차량 좌표 초기화
                            car_y1 = 0
                            car_x2 = 0
                            car_y2 = 0
                            center = [0, 0]
                            crop_image_size = 0
                            #center_previous = [0, 0]
                            check = True  #check True 를 통하여 두번째 카메라의 알고리즘에 들어가게함
                            break
                    if check == True:
                        break
                    person_count_part = 0
                    for *xyxy, conf, cls in reversed(det):
                        if names[int(cls)] == 'person' and int(xyxy[0]) > car_x1 and int(xyxy[1]) > car_y1 and int(
                                xyxy[2]) < car_x2 and int(xyxy[3]) < car_y2 \
                                and bbox_size(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])) < (
                                bbox_size(car_x1, car_y1, car_x2, car_y2) / 2):
                            # 크롭한 차량의 사람을 구하는 조건문
                            person_count_part = person_count_part + 1
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    if person_count < person_count_part:
                        person_count = person_count_part

                    # 영상결과

                    #terminate_t = timeit.default_timer()
                    #FPS = int(1. / (terminate_t - start_t))
                    #cv2.putText(im0, str(FPS), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2,  #fps을 이미지에 삽입함
                    #           cv2.LINE_AA)
                if opt.rotation == 0:
                    cv2.line(im0, (limit_line_width, limit_line_high), (limit_line_width2, limit_line_high),(0, 0, 255), 1)  # 선긋기
                    cv2.line(im0, (limit_line_width, limit_line_high2), (limit_line_width2, limit_line_high2),(0, 255, 255), 1)  # 선긋기
                elif opt.rotation == 1:
                    cv2.line(im0, (limit_line_width, limit_line_high), (limit_line_width, limit_line_high2),(0, 0, 255), 1)  # 선긋기
                    cv2.line(im0, (limit_line_width, limit_line_high), (limit_line_width2, limit_line_high2),(0, 255, 255), 1)  # 선긋기


                cv2.imshow('detect', im0)
                cv2.setMouseCallback('detect', on_mouse)  # 마우스 이벤트 선언opencv에서
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
                # if int(center_previous[0]) < int(center[0]) and int(center_previous[0])+40 > int(center[0]):  # 차량 카운트가 끝났을때 좌표초기화

























    else:
        for path, img, im0s, vid_cap in dataset:

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    s, im0 = '%g: ' % i, im0s[i].copy()
                else:
                    s, im0 = '', im0s

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    if check == True:  # 2번째 카메라가 들어왔을때
                        check = False
                        person_count_part = 0
                        for *xyxy, conf, cls in reversed(det):
                            if names[int(cls)] == 'person':
                                person_count_part = person_count_part + 1
                                label = '%s %.2f' % (names[int(cls)], conf)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        print("첫번째카메라 사람:", person_count)
                        print("두번째 카메라 사람수 :", person_count_part)
                        if person_count_part > person_count:
                            print("최종 사람:", person_count_part)
                        else:
                            print("최종 사람:", person_count)
                        print()
                        person_count = 0
                        cv2.imshow('second_camera ', im0)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration
                        car_plate = carnumber_detection(crop_image)  # 차량번호판 인식 함수
                        print("차량번호:",str(car_plate))
                        save_carnumber(crop_image, person_count, car_plate)  # db에 저장하기 사람수랑 날짜랑 등 등
                        cv2.imshow('frist_camera', crop_image)

                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration

                        break

                    for *xyxy, conf, cls in reversed(det):  # 소켓통신을 하는 메인 카메라의 물체찾기

                        if (names[int(cls)] == 'car' or names[int(cls)] == 'truck' or names[int(cls)] == 'bus') and (
                                int(xyxy[3]) >= limit_line_high and int(xyxy[3] <= limit_line_high2)) \
                                and (int(xyxy[0]) >= limit_line_width and int(xyxy[2]) <= limit_line_width2):  # 좌표 보기
                            # print(int(xyxy[1]), int(xyxy[3]))

                            crop_image = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

                            car_x1 = int(xyxy[0])
                            car_y1 = int(xyxy[1])
                            car_x2 = int(xyxy[2])
                            car_y2 = int(xyxy[3])

                            # 차량의 중앙좌표보기
                            center_previous = center.copy()
                            height, width, _ = im0.shape
                            x1, y1, x2, y2 = max(0, int(xyxy[0])), max(0, int(xyxy[1])), min(width, int(xyxy[2])), min(
                                height, int(xyxy[3]))
                            center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

                            # print(center)
                            # print(center_previous)
                            # print()
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                            # print(car_x1,car_y1,car_x2,car_y2)
                        elif (names[int(cls)] == 'car' or names[int(cls)] == 'truck' or names[
                            int(cls)] == 'bus') and int(
                            xyxy[3] > limit_line_high2) and center[1] != 0 \
                                and (int(xyxy[0]) >= limit_line_width and int(xyxy[2]) <= limit_line_width2) and \
                                (center_previous[1] > center[1] - 30):
                            car_x1 = 0
                            car_y1 = 0
                            car_x2 = 0
                            car_y2 = 0
                            center = [0, 0]
                            center_previous = [0, 0]
                            check = True
                            break
                    if check == True:
                        break
                    person_count_part = 0
                    for *xyxy, conf, cls in reversed(det):
                        if names[int(cls)] == 'person' and int(xyxy[0]) > car_x1 and int(xyxy[1]) > car_y1 and int(
                                xyxy[2]) < car_x2 and int(xyxy[3]) < car_y2 \
                                and bbox_size(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])) < (
                                bbox_size(car_x1, car_y1, car_x2, car_y2) / 2):
                            # 크롭한 차량의 사람을 구하는 조건문
                            person_count_part = person_count_part + 1
                            #if (save_img or view_img):  # Add bbox to image  and ((names[int(cls)] == 'face') or (names[int(cls)] == 'person')
                                #label = '%s %.2f' % (names[int(cls)], conf)
                                #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    if person_count < person_count_part:
                        person_count = person_count_part
                        # Stream results
                if opt.rotation == 0:
                    cv2.line(im0, (limit_line_width, limit_line_high), (limit_line_width2, limit_line_high),(0, 0, 255), 1)  # 선긋기
                    cv2.line(im0, (limit_line_width, limit_line_high2), (limit_line_width2, limit_line_high2),(0, 255, 255), 1)  # 선긋기
                elif opt.rotation == 1:
                    cv2.line(im0, (limit_line_width, limit_line_high), (limit_line_width, limit_line_high2),(0, 0, 255), 1)  # 선긋기
                    cv2.line(im0, (limit_line_width2, limit_line_high), (limit_line_width2, limit_line_high2),(0, 255, 255), 1)  # 선긋기

                cv2.imshow('detect', im0)
                cv2.setMouseCallback('detect', on_mouse)  # 마우스 이벤트 선언opencv에서
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
                    # if int(center_previous[0]) < int(center[0]) and int(center_previous[0])+40 > int(center[0]):  # 차량 카운트가 끝났을때 좌표초기화

if __name__ == '__main__':
    freeze_support()

    manager = Manager()
    picture = manager.list([0, 0, False, False])
    print(picture)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--receive', type=int, default=0, help='리시브 정하십쇼')
    parser.add_argument('--rotation', type=int, default=0, help='트리거 로테이션( 0이면 - 1이면 ㅣ)')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
