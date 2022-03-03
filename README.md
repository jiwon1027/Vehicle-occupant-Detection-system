# 차량 탑승 인원 감지 시스템
# Vehicle occupant Detection System
#### Project Execution : 2020.06 ~ 2021.06
#### Project Manager : Prof. Sung Jin Jang
#### Project Engineer : Ji won Lee, Dong Jin Lee
# Description
현재 대한민국 경제 성장과 가구의 생활 수준 향상으로 한 가정의 차량의 수요가 증가하여 차량의 탑승 인원은 적어지고 도로의 차량 수는 증가하는 추세다. 이러한 이유로 차량의 등록 대수가 늘어나는 만큼 도로의 차량은 많아지고 교통 체증은 더욱 심해지게 되는데 이를 해결하기 위해  **다인승 전용차로**,  **HOV(High-Occupancy Vehicle) Lane**을 운영하고 있다.

이 차로는 9인승 이상 승용자동차 및 승합자동차에 6명 이상 탑승해 있어야만 이용가능한데 법규를 지키지 않고 이용하는 사람들이 계속 늘어나  **차량 탑승 인원 감지 시스템**을 구축하였다.


### 1.차량 탑승 인원 탐지 기술
||| 
|:---:|:---:|
|![](../../image01.png)|![](../../image02.png)|
|현장 시스템 구성도(1)|현장 시스템 구성도(2)|

### 2.Block Diagram & Flow chart
||
|:---:|
|![](../../image03.png)|
|Block Diagram|
|![](../../image04.png)|
|Flow Chart|

|||
|:---:|:---:|
| ![](../../image05.png)|![](../../image06.png)|
|차량 접근 시 전면 카메라 시점(1)|차량 접근 시 전면 카메라 시점(2)|
|![](../../image07.png)|![](../../image08.png)|
|차량 접근 시 전면 카메라 시점(3)|Bounding Box 중심점 이동|



## Environment
|                |                          |
|-------------------|-------------------------------|
|Server OS	 		|`Ubuntu 18.04 	`				 |    
|Language			|`Python`
|Prototype      	|`Raspberry Pi 4 Model B 4GB`
|Camera         	|`42W 850nm IR illuminater Ip Camera`
|Deep Learning Model|`Yolov5 - Yolov5s`
|DataBase			|`MariaDB`




