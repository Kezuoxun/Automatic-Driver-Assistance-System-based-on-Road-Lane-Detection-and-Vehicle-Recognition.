# https://medium.com/typeiqs/advanced-lane-finding-c3c8305f074

# https://medium.com/typeiqs/advanced-lane-finding-c3c8305f074

import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
import matplotlib.pyplot as plt # Used for plotting and error checking
 
def abs_sobel_thresh(img, orient='x', thresh=(0, 255), sobel_kernel=3):
    thresh_min, thresh_max = thresh
    s_thresh = (120, 255)
    l_thresh = (40, 255)
    
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls_image[:,:,0]
    L = hls_image[:,:,1]
    S = hls_image[:,:,2]
    
    if orient == 'x':
        sobel = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    l_sobel = np.zeros_like(scaled_sobel)
    l_sobel[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    s_binary = np.zeros_like(S)
    s_binary[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 1
    
    l_binary = np.zeros_like(L)
    l_binary[(L >= l_thresh[0]) & (L <= l_thresh[1])] = 1
    
    binary_output = np.zeros_like(l_sobel)
    binary_output[((l_binary == 1) & (s_binary == 1) | (l_sobel==1))] = 1  
    
    return binary_output

def mag_thresh(img, sobel_kernel=5, thresh=(0,255)):
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
    gaussian_blur = cv2.GaussianBlur(img, (3, 3), 0)  # img process of gaussian_blur
    sobelx = cv2.Sobel(gaussian_blur, cv2.CV_32F, 1, 0, ksize=sobel_kernel)  # cv2.CV_16S
    sobely = cv2.Sobel(gaussian_blur, cv2.CV_32F, 0, 1, ksize=sobel_kernel)  # cv2.CV_16S
    
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    
    return binary_output

def dir_threshold(img, sobel_kernel=9, thresh=(0, np.pi/2)):   
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    dirs = np.arctan2(abs_sobely, abs_sobelx)
    
    binary_output = np.zeros_like(dirs)
    binary_output[(dirs >= thresh[0]) & (dirs <= thresh[1])] = 1
    
    return binary_output
def combined_thresh(img, sobel_kernel=5, abs_thresh=(40,220), _mag_thresh=(50,220), dir_thresh=(0, np.pi/2)):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=sobel_kernel, thresh=abs_thresh)
    
    mag_binary = mag_thresh(img, sobel_kernel=sobel_kernel, thresh=_mag_thresh)
    
    dir_binary = dir_threshold(img, sobel_kernel=sobel_kernel, thresh=dir_thresh)
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) ) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # histogram = np.sum(combined[combined.shape[0]//2:,:], axis=0)
    # plt.plot(histogram)

    return combined

def perspective_transform(img):
    img2 = img
    # plt.imshow(img2)
    # plt.show()
    # img2 = img.copy()
    # cv2.imshow("img2",img2)
    # cv2.waitKey(1)
    img_size2 = (img2.shape[1], img2.shape[0])  # (1116, 552)   (1892, 874) x,y
    print(img_size2)
  # offset =150
    src = np.float32([[(1, 173),
                       (127,98), 
                       (210,98), 
                       (305, 173)]])

    # 3                  #      src = np.float32([[(1, 173),
                      #  (127,98), 
                      #  (210,98), 
                      #  (305, 173)]])
    dst = np.float32([
            [0,img2.shape[0]],            # bottom-left corner
            [0,0],                       # top-left corner
            [img2.shape[1],0 ],           # top-right corner
            [img2.shape[1], img2.shape[0]]  # bottom-right corner
        ])
    M = cv2.getPerspectiveTransform( src,dst)  # , dst
    warped_images = cv2.warpPerspective(img2, M, img_size2)
    cv2.imshow("warped_images",warped_images)

    M_INV = cv2.getPerspectiveTransform( dst,src)  # , dst

    
    return warped_images,M_INV 
    
def sliding_window_polyfit(img):
    print("img.shape" , img.shape)
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((img, img, img))*255)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int_(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 8
    # Set height of windows
    window_height = np.int_(img.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 60
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, out_img

def visualize_polyfit(img, out_img, left_fit, right_fit, left_lane_inds, right_lane_inds):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    color_o =[150,150,100]  # 淡藍色
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]   # 塗滿lane
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]   # 塗滿lane
    # cv2.line(img,ploty, ploty, color_o, 6)#畫出直線
    # cv2.line(img,right_fitx,ploty, color_o,6)#畫出直線

    # plt.figure(figsize = (10,5))
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color=('yellow'))  # x,y
    # plt.plot(right_fitx, ploty, color='yellow')   # x,y
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    cv2.imshow('out_img' , out_img)

def skip_windows_step(binary_warped, left_fit, right_fit):  # skip_window
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return out_img, left_fit, right_fit, left_lane_inds, right_lane_inds
    
def visualize_skip_window_step(binary_warped, out_img, left_fitx, right_fitx, left_lane_inds, right_lane_inds):  # skip_window
    margin = 60    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.uint8(np.dstack((binary_warped, binary_warped, binary_warped))*255)
    window_img = np.zeros_like(out_img)
    
    # Color in left and right line pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    cv2.imshow('window_img' , window_img)
    cv2.waitKey(0)

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    thickness=8
    plt.figure(figsize = (10,5))
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')   # x , y
    plt.plot(right_fitx, ploty, color='yellow')
    # cv2.line(img, left_fitx, ploty, 'yellow',  thickness)#畫出直線
    # cv2.line(img, right_fitx, ploty, 'yellow', thickness )#畫出直線
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    cv2.imshow('result' , result)

def measure_curvature(binary_warped, left_lane_inds, right_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    y_eval = np.max(ploty)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad
def measure_car_pos(binary_warped, left_fit, right_fit):
    xm_per_pix = 3.7/700
    car_position = binary_warped.shape[1]/2
    height = binary_warped.shape[0]
    
    left_fit_x = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    right_fit_x = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    
    lane_center_position = (right_fit_x + left_fit_x) /2
    
    center_dist = (car_position - lane_center_position) * xm_per_pix
    
    return center_dist


def draw_lines(img, binary_img, left_fit, right_fit, Minv):
    ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.imshow("color_warp" , color_warp)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    cv2.imshow("result222" , result)


    return result

def draw_curvature_data(img, curv_rad, car_pos):
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:02.2f}'.format(curv_rad/1000) + 'Km'
    cv2.putText(img, text, (30,70), font, 1.5, (0,255,0), 2, cv2.LINE_AA)
    
    text = 'Car pos. from center: ' + '{:02.3f}'.format(car_pos) + 'm'
    cv2.putText(img, text, (30,120), font, 1.5, (0,255,0), 2, cv2.LINE_AA)
    
    return img

import os

i = 0
array_of_img = []

for filename in os.listdir("./test_image/02/"):

    img = cv2.imread("./test_image/02/" + "/" + filename)

    cv2.imshow('img', img)

    res_img=combined_thresh(img, sobel_kernel=3, abs_thresh=(20,200), _mag_thresh=(20,255), dir_thresh=(0, np.pi/2))  # == combined
    res_img1 , M_INV  = perspective_transform(res_img)
    # plt.imshow(res_img1, cmap='gray')  # figure 1
    # plt.show()
    # histogram = np.sum(res_img1[res_img1.shape[0]//2:,:], axis=0)  # see a histogram to now how the data behaves
    # plt.plot(histogram)
    # plt.show()

    left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = sliding_window_polyfit(res_img1)
    visualize_polyfit(res_img1, out_img, left_fit, right_fit, left_lane_inds, right_lane_inds)

    out_img, left_fit2, right_fit2, left_lane_inds2, right_lane_inds2 = skip_windows_step(res_img1, left_fit, right_fit)
    visualize_skip_window_step(res_img1, out_img, left_fit2, right_fit2, left_lane_inds2, right_lane_inds2)

    left_curverad, right_curverad = measure_curvature(res_img1, left_lane_inds2, right_lane_inds2)
    print(left_curverad, 'm', right_curverad, 'm')
    car_pos = measure_car_pos(res_img1, left_fit2, right_fit2)
    print(car_pos, 'm')
    i =i+1
    array_of_img.append(img)
    new_img = draw_lines(img, res_img, left_fit, right_fit, M_INV)
    cv2.imshow('new_img',new_img)
    Img_Name = "./test_image/lane2/1/2/"+str(i)+".jpg"  # 要存結果再打開
    cv2.imwrite(Img_Name,new_img)
    # final_img = draw_curvature_data(img, (left_curverad + right_curverad)/2, car_pos)
    # plt.imshow(final_img)
    # plt.show()



    cv2.imshow('window_title',res_img1)
    # cv2.imshow("region_of_interest" , region_of_interest(edges) )
    # cv2.imshow("edges" , canny)
    cv2.waitKey(0)
#     break

cv2.destroyAllWindows()