# from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import cv2
import os
import glob


blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 40  # Canny edge detection low threshold
canny_hthreshold = 120  # Canny edge detection high threshold

# Hough transform parameters

rho = 1 #rho的步長，即直線到影象原點(0,0)點的距離
theta = np.pi / 180 #theta的範圍
threshold = 20 #累加器中的值高於它時才認為是一條直線
min_line_length = 20 #線的最短長度，比這個短的都被忽略
max_line_gap = 30 #兩條直線之間的最大間隔，小於此值，認為是一條直線

# rho = 1 #rho的步長，即直線到影象原點(0,0)點的距離
# theta = np.pi / 180 #theta的範圍
# threshold = 25 #累加器中的值高於它時才認為是一條直線
# min_line_length = 25 #線的最短長度，比這個短的都被忽略
# max_line_gap = 20 #兩條直線之間的最大間隔，小於此值，認為是一條直線

def roi_mask(img, vertices): # img是輸入的影象，verticess是興趣區的四個點的座標（三維的陣列）
  mask = np.zeros_like(img) # 生成與輸入影象相同大小的影象，並使用0填充,影象為黑色
  # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
  if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    mask_color = (255,) * channel_count#如果 channel_count=3,則為(255,255,255)
  else:
    mask_color = 255
  cv2.fillPoly(mask, vertices, mask_color)#使用白色填充多邊形，形成蒙板
  masked_img = cv2.bitwise_and(img, mask)#img&mask，經過此操作後，興趣區域以外的部分被矇住了，只留下興趣區域的影象
  return masked_img

def roi_mask2(img, vertices): # img是輸入的影象，verticess是興趣區的四個點的座標（三維的陣列）
  mask2 = np.zeros_like(img) # 生成與輸入影象相同大小的影象，並使用0填充,影象為黑色
  # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
  if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    mask_color = (255,) * channel_count #如果 channel_count=3,則為(255,255,255)
  else:
    mask_color = 255
  cv2.fillPoly(mask2, vertices, mask_color) #使用白色填充多邊形，形成蒙板 on mask2
  masked_img2 = cv2.bitwise_and(img, mask2) #img & mask2，經過此操作後，興趣區域以外的部分被蓋住了，只留下興趣區域的影象 https://blog.csdn.net/LaoYuanPython/article/details/109148867
  return masked_img2

def draw_roi(img, vertices):
  cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        print("No line detected in image")

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
  lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)#函式輸出的直接就是一組直線點的座標位置（每條直線用兩個點表示[x1,y1],[x2,y2]）
  print("lines",lines)
  # cv2.imshow("img",img)
  # cv2.waitKey(0)
  line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)#生成繪製直線的繪圖板，黑底
  # draw_lines(line_img, lines)
  draw_lanes(line_img, lines)
  
  return line_img

def lines_active(img, rho, theta, threshold, min_line_len, max_line_gap):
  lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)#函式輸出的直接就是一組直線點的座標位置（每條直線用兩個點表示[x1,y1],[x2,y2]）
  return lines

def perspective_transform(img2):
    # img2 = img.copy()
    cv2.imshow("img2",img2)
    cv2.waitKey(20)
    img_size2 = (img2.shape[1], img2.shape[0])  # (1892, 874) x,y
    print(img_size2)
  # offset =150
    src = np.float32([[(10, img2.shape[0]-20),
                       (100,80), 
                       (300,80), 
                       (img2.shape[1]-20, img2.shape[0]-20)]])
    dst = np.float32([
            [0,img2.shape[0]],            # bottom-left corner
            [0,0],                       # top-left corner
            [img2.shape[1],0 ],           # top-right corner
            [img2.shape[1], img2.shape[0]]  # bottom-right corner
        ])
    M = cv2.getPerspectiveTransform( src,dst)  # src --> dst  => begin_matrix --> end_matrix  perspective matrix  (3 dimenation matrix)
    print("M" , M)
    warped = cv2.warpPerspective(img2, M, img_size2)
    # cv2.imshow("warped",warped)
    # cv2.waitKey(1)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)#影象轉換為灰度圖
    blur_warped = cv2.GaussianBlur(warped_gray, (blur_ksize, blur_ksize),5) # 使用高斯模糊去除noise , sigma =0.3*((ksize-1)*0.5-1)+0.8 = 1.1
    # cv2.imshow("blur_warped",blur_warped)
    # cv2.waitKey(1)
    # roi_edges = roi_mask(blur_warped, np.array([[(300, 874), (720 ,450), (1150 ,450),(1500 , 874)]])) #對邊緣檢測的影象生成影象蒙板，去掉不感興趣的區域，保留興趣區
    edges_warped = cv2.Canny(blur_warped, canny_lthreshold, canny_hthreshold) #使用Canny進行邊緣檢測  if point_binary value > hthreshold , 則此點也視為邊緣
    # cv2.imshow("edges_warped",edges_warped )
    # cv2.waitKey(1)

    # cv2.imshow("roi_edges2",roi_edges2 )
    cv2.waitKey(20)

        
    # lines3 = lines_active(roi_edges2, rho, theta, threshold, min_line_length, max_line_gap)   # HoughLinesP
    # coordinate2 = draw_lanes(warped, lines3, color=[255, 0, 0], thickness=8)  # average_slope_intercept
    # # print("coordinate2",coordinate2)
    # coordinate2=list(coordinate2) #tuple轉list
    # coordinate2=np.array(coordinate2) #list轉array
    # coordinate2[[0,1],:] = coordinate2[[1,0],:] #第0列與第一列互換
    # print("coordinate2: \n",coordinate2)
    # res_img_warp=cv2.fillPoly(warped , [coordinate2], (0, 255, 0))  # 將線合成至原影像
    # cv2.imshow("res_img_warp",res_img_warp )
    # cv2.waitKey(0)
    # line_warped_res_img = cv2.addWeighted(warped  , 0.8  ,res_img_warp, 0.2, 1)#將處理後的影象與原圖做融合
    # cv2.imshow('line_warped_res_img',line_warped_res_img)
    return edges_warped

def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
  left_lines, right_lines = [], []#用於儲存左邊和右邊的直線
  if lines is not None:
    for line in lines:#對直線進行分類
      for x1, y1, x2, y2 in line:
        k = (y2 - y1) / (x2 - x1)
        if k < 0:
          left_lines.append(line)
        else:
          right_lines.append(line)
  else:
        print("No line detected in image")

  if (len(left_lines) <= 0 or len(right_lines) <= 0):
    return img

  clean_lines(left_lines, 0.1)#彈出左側不滿足斜率要求的直線
  clean_lines(right_lines, 0.1)#彈出右側不滿足斜率要求的直線
  left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line]#提取左側直線族中的所有的第一個點
  left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]#提取左側直線族中的所有的第二個點
  right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]#提取右側直線族中的所有的第一個點
  right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]#提取右側側直線族中的所有的第二個點

  left_vtx = calc_lane_vertices(left_points, 100, img.shape[0])#擬合點集，生成直線表示式，並計算左側直線在影象中的兩個端點的座標
  right_vtx = calc_lane_vertices(right_points, 100, img.shape[0])#擬合點集，生成直線表示式，並計算右側直線在影象中的兩個端點的座標
  
  color_l= [255,0,0]
  color_r =[0,0,255]
  left_vtx1 = np.reshape(left_vtx , (2,2))  # col,row
  right_vtx1 = np.reshape(right_vtx , (2,2))   # col,row
  color_o =[200,200,220]  # 淡藍色
  print(" right_vtx1[0,1]" ,  right_vtx1[1,0])
  print(" left_vtx1[0,1]" ,  left_vtx1[1,0])

  right1 = right_vtx1[1,0]
  left1 = left_vtx1[1,0]  # (col , row)
  my_x = int((right1 + left1)/2)  # 兩線的中線(bottom)
  
  right2 = right_vtx1[0,0]
  left2 = left_vtx1[0,0]
  my_x2 = int((right2 + left2)/2)  # 兩線的中線(top)

  cv2.line(img, (my_x , right_vtx1[1,1]) , (my_x2  , right_vtx1[0,1])  , color_o, 8)#畫出中間直線  (1892, 874) x,y
  cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)#畫出直線
  cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)#畫出直線
  coordinate=left_vtx[0],left_vtx[1],right_vtx[0],right_vtx[1]
  
  print(" my_x" ,  my_x , "l" ,  my_x - left_vtx1[1,0] , "r" , my_x - right_vtx1[1,0]  )
  
  
  return coordinate


#將不滿足斜率要求的直線彈出
def clean_lines(lines, threshold):
    slope=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            k=(y2-y1)/(x2-x1)
            slope.append(k)
    #slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)#計算斜率的平均值，因為後面會將直線和斜率值彈出
        diff = [abs(s - mean) for s in slope]#計算每條直線斜率與平均值的差值
        idx = np.argmax(diff)#計算差值的最大值的下標
        if diff[idx] > threshold:#將差值大於閾值的直線彈出
          slope.pop(idx)#彈出斜率
          lines.pop(idx)#彈出直線
        else:
          break

#擬合點集，生成直線表示式，並計算直線在影象中的兩個端點的座標
def calc_lane_vertices(point_list, ymin, ymax):
  x = [p[0] for p in point_list]#提取x
  y = [p[1] for p in point_list]#提取y
  fit = np.polyfit(y, x, 1)#用一次多項式x=a*y+b擬合這些點，fit是(a,b)
  fit_fn = np.poly1d(fit)#生成多項式物件a*y+b

  xmin = int(fit_fn(ymin))#計算這條直線在影象中最左側的橫座標
  xmax = int(fit_fn(ymax))#計算這條直線在影象中最右側的橫座標
  
  return [xmin, ymin],[xmax, ymax]

def process_an_image(img):
  # roi_vtx = np.array([[(0-40, img.shape[0]-20),
  #                      (100,80), 
  #                      (300,80), 
  #                      (img.shape[1]-20, img.shape[0]-20)]])#目標區域的四個點座標，roi_vtx是一個三維的陣列
  # print(roi_vtx)++
  #一:(canny 50 150)  15 15 40 擬和點集90   (roi (0, img.shape[0]),(115,80), (210,80), (img.shape[1], img.shape[0]))
  #二:(canny 40 120)  20 20 30 擬和點集100  (0-50 -15 roi(100, 80), (220, 80) +10 -15)
  #三:(canny 40 120)  20 20 30 擬和點集100  (0-40  0   roi(90, 82), (235, 82) +10  0)
  img = perspective_transform(img)
  
#   gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#影象轉換為灰度圖
#   cv2.imshow('original img',img)
# #   cv2.waitKey(0)
# #   gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#影象轉換為灰度圖
#   cv2.imshow('RGB graph',gray)
#   cv2.waitKey(0)
#   blur_gray = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0, 0)#使用高斯模糊去除noise
#   cv2.imshow('GaussianBlur',blur_gray)
  
#   edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)#使用Canny進行邊緣檢測
#   cv2.imshow('Canny edges',edges)
# #   cv2.waitKey(0)
#   roi_edges = roi_mask(edges, roi_vtx)#對邊緣檢測的影象生成影象蒙板，去掉不感興趣的區域，保留興趣區
#   cv2.imshow("roi_mask_edges",roi_edges)
#   cv2.waitKey(0)
  line_img = hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap)#使用霍夫直線檢測，並且繪製直線
  cv2.imshow("Hough lines",line_img)
#   cv2.waitKey(0)
  lines = lines_active(img, rho, theta, threshold, min_line_length, max_line_gap)
  res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)#將處理後的影象與原圖做融合
  img1=res_img.copy()
  coordinate = draw_lanes(img, lines, color=[255, 0, 0], thickness=8)
  print("coordinate",coordinate)
  coordinate=list(coordinate) #tuple轉list
  coordinate=np.array(coordinate) #list轉array
  coordinate[[0,1],:] = coordinate[[1,0],:] #第0列與第一列互換
  print("coordinate: \n",coordinate)

  # 車道半透明填充
  res_img1=cv2.fillPoly(img1, [coordinate], (0,255, 0))
  cv2.imshow("res_img1",res_img1)

  alpha = 0.8
  beta = 1-alpha
  gamma = 0
  res_img = cv2.addWeighted(res_img, alpha,res_img1, beta, gamma)
  
  # cv2.fillPoly(res_img, [coordinate], (0, 255, 0)) #填滿車道
  return res_img

# cap = cv2.VideoCapture("car3.avi")  # test2.mp4  road_car_view.mp4   

# while True:
#     ret , frame = cap.read()  

#     if ret :
#         cv2.imshow('video', frame)

#         res_img=process_an_image(frame)
#         # # sliding_window_polyfit = sliding_window_polyfit(frame)
#         # cv2.imshow('window_title', res_img)
#         # cv2.imshow("sliding_window_polyfit" , sliding_window_polyfit )
#         # cv2.imshow("edges" , canny)
#         if cv2.waitKey(100) == ord('q'):
#             break
i = 0
array_of_img = []

for filename in os.listdir("./test_image/03/"):
    img = cv2.imread("./test_image/03/" + "/" + filename)
    cv2.imshow("img",img)
    i =i+1
    array_of_img.append(img)
    print(array_of_img)
    res_img=process_an_image(img)
    # Img_Name = "./test_image/lane2/2"+str(i)+".jpg"  # 要存結果再打開
    # cv2.imwrite(Img_Name,res_img)
    cv2.imshow("test_img",res_img)
    # line_warped_res_img = perspective_transform(img)
    # cv2.imshow('line_warped_res_img', line_warped_res_img)

    cv2.waitKey(0)

# for filename in os.listdir("./test_image/lane1/1/"):
#     img = cv2.imread("./test_image/lane1/1/" + "/" + filename)
#     cv2.imshow("img",img)
#     i =i+1
#     array_of_img.append(img)
#     print(array_of_img)
#     res_img=process_an_image(img)
#     Img_Name = "./test_image/lane1/2/"+str(i)+".jpg"
#     cv2.imwrite(Img_Name,res_img)
#     cv2.imshow("res_img",res_img)

#     cv2.waitKey(0)








# img= mplimg.imread("C:/Users/User/Desktop/python_work/3/3/1361.jpg")
# res_img=process_an_image(img)
# cv2.imshow("res_img",res_img)
# cv2.waitKey(0)
# cv2.imwrite('C:/Users/User/Desktop/python_work/1 result/595.jpg', res_img)
# print("show you the image....")


# print("start to process the video....")
# output = 'video_2_xlt.mp4'#ouput video
# clip = VideoFileClip("video_2.mp4")#input video
# out_clip = clip.fl_image(process_an_image)#對視訊的每一幀進行處理
# out_clip.write_videofile(output, audio=True)#將處理後的視訊寫入新的視訊檔案