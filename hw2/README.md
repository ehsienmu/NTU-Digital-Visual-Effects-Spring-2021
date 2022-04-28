VFX hw2 report
===
**b06303032 資管三 黃佳文
b06504104 化工四 石子仙**


## 1. Description of this project
依序拍攝同一定點不同方向之照片，運用feature detection將每張照片的feature找出，再使用feature matching與image matching將照片接在一起，並使用blending使得邊界不那麼明顯來得到一張全景圖。

## 2. Our images
### - 場景一：頂樓
<img src="https://i.imgur.com/jNvfpRb.jpg" alt="" width="120">  <img src="https://i.imgur.com/UCy0L8I.jpg" alt="" width="120">  <img src="https://i.imgur.com/Ze0vaKw.jpg" alt="" width="120">  <img src="https://i.imgur.com/hm4KD5p.jpg" alt="" width="120">  <img src="https://i.imgur.com/8kN8Wqz.jpg" alt="" width="120">  <img src="https://i.imgur.com/VfFvPpc.jpg" alt="" width="120">  <img src="https://i.imgur.com/jasUndh.jpg" alt="" width="120">  <img src="https://i.imgur.com/DrQKU9M.jpg" alt="" width="120">  <img src="https://i.imgur.com/CyYIlmz.jpg" alt="" width="120">  <img src="https://i.imgur.com/rOK08gG.jpg" alt="" width="120">  <img src="https://i.imgur.com/eZMt7tn.jpg" alt="" width="120">  <img src="https://i.imgur.com/QT1Xzx6.jpg" alt="" width="120">  <img src="https://i.imgur.com/DKmDAZV.jpg" alt="" width="120">  <img src="https://i.imgur.com/LQFsbCl.jpg" alt="" width="120">   <img src="https://i.imgur.com/wSXceeR.jpg" alt="" width="120">  

### - 場景二：廣場
<img src="https://i.imgur.com/BTgtxau.jpg" alt="" width="120">  <img src="https://i.imgur.com/Ku7iHvm.jpg" alt="" width="120">  <img src="https://i.imgur.com/F3GCP7Y.jpg" alt="" width="120">  <img src="https://i.imgur.com/UUiGmva.jpg" alt="" width="120">  <img src="https://i.imgur.com/eTdNOED.jpg" alt="" width="120">  <img src="https://i.imgur.com/GcVfKX9.jpg" alt="" width="120">  <img src="https://i.imgur.com/1wbINIS.jpg" alt="" width="120">  <img src="https://i.imgur.com/fSqrKB0.jpg" alt="" width="120">  <img src="https://i.imgur.com/Njhm8nz.jpg" alt="" width="120">  <img src="https://i.imgur.com/w3ep0dp.jpg" alt="" width="120">  <img src="https://i.imgur.com/ziXKEmO.jpg" alt="" width="120">  <img src="https://i.imgur.com/dOQLAnF.jpg" alt="" width="120">  <img src="https://i.imgur.com/ODTxjoR.jpg" alt="" width="120">  <img src="https://i.imgur.com/Bo5wflR.jpg" alt="" width="120">  <img src="https://i.imgur.com/C0wsPp9.jpg" alt="" width="120">  <img src="https://i.imgur.com/Yk6TpJ2.jpg" alt="" width="120">

## 3. 程式流程 

1. Use autostitch to get focal length of all image.
2. Apply cylindrical projection,
3. Feature detection: Harris Corner Detector,
4. Feature matching, 
5. Image matching, 
6. Blending. 

### - Feature Detection: Harris Corner Detector
- 偵測特徵點採用Harris Corner Detector方法如下：
    1. 照片專灰階，經過gaussian_filter，並計算x、y方向gradient。
    2. 計算x、x，x、y，y、y方向像素乘積。
    3. 將以上乘積合再將過gaussian_filter，形成Ｍ矩陣
    4. 經過$R=det(M)-k(traceM)^2$算出R
    5. 當R大於threshold且是local maximum就視為feature
- 特徵點照片範例：

    <img src="https://i.imgur.com/kG0aEed.jpg" alt="" width="350">  <img src="https://i.imgur.com/1PROMu6.jpg" alt="" width="350"> 

    <img src="https://i.imgur.com/BTVJ1Hu.jpg" alt="" width="350">  <img src="https://i.imgur.com/lGHOZ7o.jpg" alt="" width="350">








### - Feature Matching
- 先為每個feature做descriptor：

    1. 取此feature周圍的一塊影像
    
    <img src="https://i.imgur.com/a3Ax6yj.png" alt="" width="250">
    
    2. 做gaussian blur
    
    <img src="https://i.imgur.com/dcoH74a.png" alt="" width="250">
    
    3. 再將他轉成intensity
    
    <img src="https://i.imgur.com/FFBi7Em.png" alt="" width="250">
    
    4. 最後再flatten成一維向量來作為此點的feature descriptor

- 對於一個image中的每個feature，我們會計算在另一張image中距離最小的點，並且規定此mse要小於一定的threshold才算match。

- Matching Result

<img src="https://i.imgur.com/f9iJFEq.jpg"> 

### - Image Matching
          
這部份我們使用RANSAC演算法來決定如何align two images：
          
- 取n=1（一個Pair）決定位移
- 設定threshold，計算經過此位移後，其他matching pairs的距離，若距離小於threshold，則投票給這個位移量。
- 最後選最多投票數的結果作為最佳位移量。

### - Blending
          
- 我們採用一般的線性插值來blending。

  <img src="https://i.imgur.com/cCH7nw4.png" alt="" width="250">
                                                             
  此圖來自課程投影片lec07_ImageStitching第41頁
                                                             
## Our Panorama
### 1. Parrington（老師提供的dataset）
                                                             
```
detect_threshold = 180000
match_threshold = 400
```
                                                             
<img src="https://i.imgur.com/tfjlwVB.jpg"> 

### 2. 頂樓
                                      
```
detect_threshold = 200000
match_threshold = 600
```
                                      
<img src="https://i.imgur.com/5Ml0fX6.jpg"> 
（天氣不好很像廢墟）
                                      
### 3. 廣場
                                      
```
detect_threshold = 4000
match_threshold = 200
```
                                      
<img src="https://i.imgur.com/jwI9mY2.jpg"> 


