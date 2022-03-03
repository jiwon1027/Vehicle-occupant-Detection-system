# 차량 탑승 인원 감지 시스템
# Vehicle occupant Detection System
#### Project Execution : 2020.06 ~ 2021.06
#### Project Manager : Prof. Sung Jin Jang
#### Project Engineer : Ji won Lee, Dong Jin Lee
# Description
현재 대한민국 경제 성장과 가구의 생활 수준 향상으로 한 가정의 차량의 수요가 증가하여 차량의 탑승 인원은 적어지고 도로의 차량 수는 증가하는 추세다. 이러한 이유로 차량의 등록 대수가 늘어나는 만큼 도로의 차량은 많아지고 교통 체증은 더욱 심해지게 되는데 이를 해결하기 위해  **다인승 전용차로**,  **HOV(High-Occupancy Vehicle) Lane**을 운영하고 있다.

이 차로는 9인승 이상 승용자동차 및 승합자동차에 6명 이상 탑승해 있어야만 이용가능한데 법규를 지키지 않고 이용하는 사람들이 계속 늘어나  **차량 탑승 인원 감지 시스템**을 구축하였다.

## Environment
|                |                          |
|-------------------|-------------------------------|
|Server OS	 		|`Ubuntu 18.04 	`				 |    
|Language			|`Python`
|Prototype      	|`Raspberry Pi 4 Model B 4GB`
|Camera         	|`42W 850nm IR illuminater Ip Camera`
|Deep Learning Model|`Yolov5 - Yolov5s`
|DataBase			|`MariaDB`


## 1.차량 탑승 인원 탐지 기술
||| 
|:---:|:---:|
|![image01](https://user-images.githubusercontent.com/68945145/156527837-883b5d11-640b-4464-a50f-1b8db8873a4b.png)|![image02](https://user-images.githubusercontent.com/68945145/156527865-f49cc9c2-dcca-47dd-a621-dc99ac475c79.png)|
|현장 시스템 구성도(1)|현장 시스템 구성도(2)|

### 2.Block Diagram & Flow chart
||
|:---:|
| ![image03](https://user-images.githubusercontent.com/68945145/156528526-b68a9d09-7616-4fe9-830a-e0ee6592562b.png) |
|Block Diagram|

||
|:---:|
| <img width="400" alt="image04" src="https://user-images.githubusercontent.com/68945145/156528010-a887aa67-d9c0-4be6-9b8c-b26cefd12c4b.png">|
|Flow Chart|

|||
|:---:|:---:|
|![image05](https://user-images.githubusercontent.com/68945145/156529390-3b399927-e142-471f-87b1-b17b23a1a2f2.png)| ![image06](https://user-images.githubusercontent.com/68945145/156529404-3200e48c-5883-4144-a351-6f94d8e36094.png)|
|차량 접근 시 전면 카메라 시점(1)|차량 접근 시 전면 카메라 시점(2)|
|![image07](https://user-images.githubusercontent.com/68945145/156529465-91873efd-5ba7-4564-bbdc-6fb34bae47d3.png)| ![image08](https://user-images.githubusercontent.com/68945145/156529495-d4e6ff51-d66c-4041-846b-c77d916db3fb.png)|
|차량 접근 시 전면 카메라 시점(3)|Bounding Box 중심점 이동|

## Result
|||
|:---:|:---:|
| ![image11](https://user-images.githubusercontent.com/68945145/156529978-c21c485b-edb3-4820-87a9-1fa2ec8e7216.png)|  ![image12](https://user-images.githubusercontent.com/68945145/156529992-8c56630c-83d7-4a8f-98e9-f2dc82506756.png)  |







