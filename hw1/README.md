VFX hw1 report
===
b06504104 化工四 石子仙
b06303032 資管三 黃佳文

## 1. Description of this project 
拍攝數張不同快門速度的照片還原成High Dynamic Range照片，並畫出radiance map來表示場景的能量。再進行tone mapping，製成較貼近我們的視覺效果的LDR照片。
## 2. Our Images
#### - 設備：
- 相機：Canon EOS 600D
- ISO：ISO-3200
- 光圈：f/3.5
#### - 場景一 總圖：
（快門紀錄在第8點）

<img src="https://i.imgur.com/o3HqqjB.jpg" alt="" width="170"> <img src="https://i.imgur.com/oFBY1fZ.jpg" alt="" width="170"> <img src="https://i.imgur.com/nqgi6vD.jpg" alt="" width="170"> <img src="https://i.imgur.com/o4M5Yte.jpg" alt="" width="170">
<img src="https://i.imgur.com/Tuf2GT4.jpg" alt="" width="170"> <img src="https://i.imgur.com/FZcqKcv.jpg" alt="" width="170"> <img src="https://i.imgur.com/PO0hGcJ.jpg" alt="" width="170"> <img src="https://i.imgur.com/DL6JwI5.jpg" alt="" width="170">
-----
#### - 場景二 社科院走廊：
（快門紀錄在第8點）

<img src="https://i.imgur.com/XlN4PQC.jpg" alt="" width="170"> <img src="https://i.imgur.com/wOsRig4.jpg" alt="" width="170"> <img src="https://i.imgur.com/uAq1OyJ.jpg" alt="" width="170"> <img src="https://i.imgur.com/6li24Ib.jpg" alt="" width="170">
<img src="https://i.imgur.com/WNTj2pm.jpg" alt="" width="170"> <img src="https://i.imgur.com/VP0DW6V.jpg" alt="" width="170"> <img src="https://i.imgur.com/9zKGdbK.jpg" alt="" width="170"> <img src="https://i.imgur.com/XM32Zkt.jpg" alt="" width="170">
<img src="https://i.imgur.com/Pcvngt4.jpg" alt="" width="170"> <img src="https://i.imgur.com/eNJpD4U.jpg" alt="" width="170">
---

## 3. 程式流程 
Language: python
1. Load Images
2. Image alignment: MTB algorithm
3. Conructing HDR radiance map: Debevec method and Robertson method
4. Tone mapping: Global Opertor、Local Opertor and Drago

## 4. MTB Alignment
我們在拍攝總圖照片時，因為電池快沒電一直關機加上有非常多人經過，拍得很緊張。Canon這台相機沒有remote control，我們只有接一條快門操控線，快門調整是用手去轉動，總圖是我們拍攝的三個場景中晃動最明顯的一個。

運用Median Threshold Bitmap Alignment方法，先將照片轉成灰階，並計算每張照片亮度的中位數，將整張照片每個分成大於中位數(1)、小於中位數(0)。先選定一張照片固定，拿其他照片來與其比較。每次先將照片縮小成1/32，比對上下左右各移動1 pixel中哪張照片會使兩者間的差距最小。之後再放大2倍，變為原本的1/16，再次檢查其上下左右的pixel，如此重複到原本的大小便可較有效率的對齊照片。

我們是用相鄰的兩張快門相片計算相對移動，最後再算出每張照片需的絕對移動。
     
- 尚未進行Alignment前：

  <img src="https://i.imgur.com/AnpNiWJ.gif" alt="" width="200">

- 經過MTB Alignment後：

  <img src="https://i.imgur.com/s6Uliki.gif" alt="" width="200">

可以看到Alignment後晃動比較不明顯，雖然還是有一點晃動，我們覺得是因為照片旁邊的樹葉一直晃所以可能有些影響。

## 5. HDR Algorithm and Result of Radiance Map

### Computing HDR
我們實作了兩種方法：
#### 1. Debevec's method：
出自[Recovering High Dynamic Range Radiance Maps from Photographs](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/papers/Debevec1997RHD.pdf) 
*    選擇一張代表照片，隨機選擇picture intensity為0到256的各一個點作為sample
* 執行最小化以下目標方程式
    <img src="https://i.imgur.com/zt4AoRt.jpg" alt="" width="380">
* 加入權重，去除noise
    <img src="https://i.imgur.com/Oh0eFON.jpg" alt="" width="300">
    
#### Radiance Map and Response Curve
* 總圖：

  <img src="https://i.imgur.com/7NjhVgV.png" alt="" width="400"> <img src="https://i.imgur.com/KMFT3t6.png" alt="" width="250">

* 社科院走廊：

  <img src="https://i.imgur.com/441sDyO.png" alt="" width="400"><img src="https://i.imgur.com/NPtdAzt.png" alt="" width="250">

#### 2. Robertson's method：
出自[Estimation- Theoretic Approach to Dynamic Range Enhancement using Multiple Exposures](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/papers/Robertson2003ETA.pdf) 
* 先設<img src="https://i.imgur.com/wDMuecd.jpg" alt="" width="22">為已知反覆利用以下公式計算<img src="https://i.imgur.com/reQnFni.jpg" alt="" width="22">和<img src="https://i.imgur.com/wDMuecd.jpg" alt="" width="22">的值，直到converge (此處直接設定做8個循環)

  <img src="https://i.imgur.com/TcW1EFG.jpg" alt="" width="200">
  <img src="https://i.imgur.com/0jobIpy.jpg" alt="" width="200">

#### Radiance Map and Response Curve
* 總圖：

  <img src="https://i.imgur.com/YZgQJ6k.png" alt="" width="400"> <img src="https://i.imgur.com/dXFfRl1.png" alt="" width="250">

* 社科院走廊：

  <img src="https://i.imgur.com/TP37pAm.png" alt="" width="400"> <img src="https://i.imgur.com/dOPADBX.png" alt="" width="250">

## 6. Tone mapping
我們實作了兩種方法：

#### 1. Global 
  由HDR結果計算整張Image的平均亮度，再Normalized各個pixel，計算LDR的值。

#### 2. Local Operator（Dodging and burning）
  Local Operator考慮了鄰近區域的平均亮度，使得亮的地方會更亮，暗的地方更暗，加強對比。

#### 3. OpenCV: Drago
  使用OpenCV內建tone mapping的Drago method，我們認為自己做出來的方法飽和度比較不漂亮，所以找了一個現有的函式來運用。

## 7. Tone mapping results

- Global Opertor

  <img src="https://i.imgur.com/ivkbPnB.jpg" alt="" width="500">

- Local Operator（Dodging and burning）

  可以看到經過Local Operator的調整後，總圖的建築框線與窗戶的線條變得比原本更明顯。
  <img src="https://i.imgur.com/vHe1rJ9.jpg" alt="" width="500">

- Drago

  <img src="https://i.imgur.com/y6DPSxK.jpg" alt="" width="500">

| Global | Local | Drago |
| -------- | -------- | -------- |
| <img src="https://i.imgur.com/FTeENnw.jpg" alt="" width="235"> | <img src="https://i.imgur.com/Fp3VBLX.jpg" alt="" width="235"> | <img src="https://i.imgur.com/bemYupw.jpg" alt="" width="235">





| Global | Local | Drago |
| -------- | -------- | -------- |
| <img src="https://i.imgur.com/nwlwE2x.jpg" alt="" width="150">      | <img src="https://i.imgur.com/rOU8DZg.jpg" alt="" width="150">     | <img src="https://i.imgur.com/jIosxwU.jpg" alt="" width="150">      |

## 8. 照片的快門秒數：
[檔名] [快門秒數]
- 總圖
```=
IMG_1_5s.jpg 5
IMG_2_2s.JPG 2
IMG_3_2.JPG 0.5 
IMG_4_4.JPG 0.25
IMG_5_10.JPG 0.1
IMG_6_25.JPG 0.04
IMG_7_30.JPG 0.0333
IMG_8_60.JPG 0.01667
```
- 社科院走廊：
```=
IMG_6772.JPG 0.000625
IMG_6774.JPG 0.001
IMG_6776.JPG 0.0025
IMG_6778.JPG 0.004
IMG_6780.JPG 0.00625
IMG_6782.JPG 0.001
IMG_6784.JPG 0.001667
IMG_6786.JPG 0.025
IMG_6788.JPG 0.04
```



## 9. What extensions we have implemented
1. 實作MTB algorithm
2. 實作Debevec、Robertson method
3. 實作Tone mapping的Global Opertor、Local Opertor
