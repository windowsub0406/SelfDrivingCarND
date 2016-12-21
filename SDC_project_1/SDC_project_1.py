# This Python file uses the following encoding: utf-8
#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import random
import os, sys
#%matplotlib inline
       
#cap = cv2.VideoCapture('solidWhiteRight.mp4')
#cap = cv2.VideoCapture('solidYellowLeft.mp4')
cap = cv2.VideoCapture('challenge.mp4')
fit_result, l_fit_result, r_fit_result, L_lane,R_lane = [], [], [], [], []

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, ( 960, 540 ))

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_circle(img,lines, color=[0, 0, 255]):
    for line in lines:
        cv2.circle(img,(line[0],line[1]), 2, color, -1)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_arr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_arr, lines)
    return lines

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def Collect_points(lines):

    # reshape [:4] to [:2]
    interp = lines.reshape(lines.shape[0]*2,2)
    # interpolation & collecting points for RANSAC
    for line in lines:
        if np.abs(line[3]-line[1]) > 5:
            tmp = np.abs(line[3]-line[1])
            a = line[0] ; b = line[1] ; c = line[2] ; d = line[3]
            slope = (line[2]-line[0])/(line[3]-line[1]) 
            for m in range(0,tmp,5):
                if slope>0:
                    new_point = np.array([[int(a+m*slope),int(b+m)]])
                    interp = np.concatenate((interp,new_point),axis = 0)
                elif slope<0:
                    new_point = np.array([[int(a-m*slope),int(b-m)]])
                    interp = np.concatenate((interp,new_point),axis = 0)                
    return interp

def get_random_samples(lines):
    one = random.choice(lines)
    two = random.choice(lines)
    if(two[0]==one[0]): # extract again if values are overlapped
        while two[0]==one[0]:
            two = random.choice(lines)
    one, two = one.reshape(1,2), two.reshape(1,2)
    three = np.concatenate((one,two),axis=1)
    three = three.squeeze()
    return three

def compute_model_parameter(line):
    # y = mx+n
    m = (line[3] - line[1])/(line[2] - line[0])
    n = line[1] - m*line[0]
    # ax+by+c = 0
    a, b, c = m, -1, n
    par = np.array([a,b,c])
    return par

def compute_distance(par, point):
    # distance between line & point
    
    return np.abs(par[0]*point[:,0]+par[1]*point[:,1]+par[2])/np.sqrt(par[0]**2+par[1]**2)

def model_verification(par, lines):
    # calculate distance
    distance = compute_distance(par,lines)
    # total sum of distance between random line and sample points    
    sum_dist = distance.sum(axis=0)
    # average
    avg_dist = sum_dist/len(lines)
    
    return avg_dist

def draw_extrapolate_line(img, par,color=(0,0,255), thickness = 2):

    x1, y1 = int(-par[1]/par[0]*img.shape[0]-par[2]/par[0]), int(img.shape[0])
    x2, y2 = int(-par[1]/par[0]*(img.shape[0]/2+100)-par[2]/par[0]), int(img.shape[0]/2+100)
    cv2.line(img, (x1 , y1), (x2, y2), color, thickness)
    return img

def get_fitline(img, f_lines):

    rows,cols = img.shape[:2]
    output = cv2.fitLine(f_lines,cv2.DIST_L2,0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)
    result = [x1,y1,x2,y2]

    return result

def draw_fitline(img, result_l,result_r, color=(255,0,255), thickness = 10):
    # draw fitting line
    lane = np.zeros_like(img)
    cv2.line(lane, (int(result_l[0]) , int(result_l[1])), (int(result_l[2]), int(result_l[3])), color, thickness)
    cv2.line(lane, (int(result_r[0]) , int(result_r[1])), (int(result_r[2]), int(result_r[3])), color, thickness)
    # add original image & extracted lane lines
    final = weighted_img(lane, img, 1,0.5)  
    return final

def erase_outliers(par, lines):
    # distance between best line and sample points
    distance = compute_distance(par,lines)

    #filtered_dist = distance[distance<15]
    filtered_lines = lines[distance<13,:]
    return filtered_lines

def smoothing(lines, pre_frame):
    # collect frames & print average line
    lines = np.squeeze(lines)
    avg_line = np.array([0,0,0,0])
    
    for ii,line in enumerate(reversed(lines)):
        if ii == pre_frame:
            break
        avg_line += line
    avg_line = avg_line / pre_frame

    return avg_line

def ransac_line_fitting(img, lines, min=100):
    global fit_result, l_fit_result, r_fit_result
    best_line = np.array([0,0,0])
    if(len(lines)!=0):                
        for i in range(30):           
            sample = get_random_samples(lines)
            parameter = compute_model_parameter(sample)
            cost = model_verification(parameter, lines)                        
            if cost < min: # update best_line
                min = cost
                best_line = parameter
            if min < 3: break
        # erase outliers based on best line
        filtered_lines = erase_outliers(best_line, lines)
        fit_result = get_fitline(img, filtered_lines)
    else:
        if (fit_result[3]-fit_result[1])/(fit_result[2]-fit_result[0]) < 0:
            l_fit_result = fit_result
            return l_fit_result
        else:
            r_fit_result = fit_result
            return r_fit_result

    if (fit_result[3]-fit_result[1])/(fit_result[2]-fit_result[0]) < 0:
        l_fit_result = fit_result
        return l_fit_result
    else:
        r_fit_result = fit_result
        return r_fit_result

def detect_lanes_img(img):

    height, width = img.shape[:2]

    # Set ROI
    vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
    ROI_img = region_of_interest(img, vertices)
    
    # Convert to grayimage
    #g_img = grayscale(img)
       
    # Apply gaussian filter
    blur_img = gaussian_blur(ROI_img, 3)
        
    # Apply Canny edge transform
    canny_img = canny(blur_img, 70, 210)
    # to except contours of ROI image
    vertices2 = np.array([[(52,height),(width/2-43, height/2+62), (width/2+43, height/2+62), (width-52,height)]], dtype=np.int32)
    canny_img = region_of_interest(canny_img, vertices2)

    # Perform hough transform
    # Get first candidates for real lane lines  
    line_arr = hough_lines(canny_img, 2, 1 * np.pi/180, 30, 10, 20)
    
    #draw_lines(img, line_arr, thickness=2)

    line_arr = np.squeeze(line_arr)
    # Get slope degree to separate 2 group (+ slope , - slope)
    slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi

    # ignore horizontal slope lines
    line_arr = line_arr[np.abs(slope_degree)<160]
    slope_degree = slope_degree[np.abs(slope_degree)<160]
    # ignore vertical slope lines
    line_arr = line_arr[np.abs(slope_degree)>95]
    slope_degree = slope_degree[np.abs(slope_degree)>95]
    L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    #print(line_arr.shape,'  ',L_lines.shape,'  ',R_lines.shape)
    
    # interpolation & collecting points for RANSAC
    L_interp = Collect_points(L_lines)
    R_interp = Collect_points(R_lines)

    #draw_circle(img,L_interp,(255,255,0))
    #draw_circle(img,R_interp,(0,255,255))

    # erase outliers based on best line
    left_fit_line = ransac_line_fitting(img, L_interp)
    right_fit_line = ransac_line_fitting(img, R_interp)

    # smoothing by using previous frames
    L_lane.append(left_fit_line), R_lane.append(right_fit_line)

    if len(L_lane) > 10:
        left_fit_line = smoothing(L_lane, 10)    
    if len(R_lane) > 10:
        right_fit_line = smoothing(R_lane, 10)
    final = draw_fitline(img, left_fit_line, right_fit_line)

    return final

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame.shape[0] !=540: # resizing for challenge video
        frame = cv2.resize(frame, None, fx=3/4, fy=3/4, interpolation=cv2.INTER_AREA) 
    result = detect_lanes_img(frame)

    cv2.imshow('result',result)

    #out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()