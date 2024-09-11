import cv2
import numpy as np

kernel = np.ones((3, 3), np.uint8)
img = cv2.imread('images/zebra.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian_filtered = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary_img = cv2.threshold(gaussian_filtered, 200, 255, cv2.THRESH_BINARY)
edged = cv2.Canny(gaussian_filtered, 30, 150)
width = img.shape[1]
height = img.shape[0]

def PreProcess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('IPM1', gray)
    gaussian_filtered = cv2.GaussianBlur(gray, (3, 3), 0)
    gaussian_filtered[0:height*5//8, 0:width] = 0
    _, binary_img = cv2.threshold(gaussian_filtered, 220, 255, cv2.THRESH_BINARY)
    cv2.imshow('IPM', binary_img)
    erosion = cv2.erode(binary_img, kernel, iterations=1)
    contour = binary_img - erosion
    return contour

def IPM(img):
    pts_src = np.float32([[width/3, height/2], [2*width/3, height/2], [0, height], [width, height]])  # source point
    pts_dst = np.float32([[width/3, height/2], [2*width/3, height/2], [width/2-100, height], [width/2+100, height]])  # destination point
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    result = cv2.warpPerspective(img, matrix, (width, height))
    result1 = result
    result = PreProcess(result)
    return result, result1

result, result1 = IPM(img)

# 使用霍夫线变换
lines = cv2.HoughLinesP(result, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=15)

# 筛选接近竖直的线段
vertical_lines = []
if len(lines) != 0:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            vertical_lines.append(line)
        else:
            slope = (x2 - x1) / (y2 - y1)
            if abs(slope) < 0.48:
                vertical_lines.append(line)

minx = width
maxx = 0
miny = height
maxy = 0
time = 0

for line in vertical_lines:
    x1, y1, x2, y2 = line[0]
    if x1 < minx:
        minx = x1
        time += 1
    if x1 > maxx:
        maxx = x1
        time += 1
    if y1 < miny:
        miny = y1
        time += 1
    if y1 > maxy:
        maxy = y1
        time += 1
    if x2 < minx:
        minx = x2
        time += 1
    if x2 > maxx:
        maxx = x2
        time += 1
    if y2 < miny:
        miny = y2
        time += 1
    if y2 > maxy:
        maxy = y2
        time += 1
if time > 0 and len(vertical_lines) >= 6:
    cv2.rectangle(result1, (minx, miny), (maxx, maxy), (255, 0, 0), 2)

        # 绘制筛选后的线段
if len(vertical_lines) >= 6:
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result1, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 使用红色绘制接近竖直的线段

#pts_src = np.float32([[width/3, height/2], [2*width/3, height/2], [0, height], [width, height]])  # 源点
#pts_dst = np.float32([[width/3, height/2], [2*width/3, height/2], [width/2-100, height], [width/2+100, height]])  # 目标点
#matrix = cv2.getPerspectiveTransform(pts_dst, pts_src)
#img = cv2.warpPerspective(result1, matrix, (width, height))

cv2.imshow('Result', result)

cv2.imshow('Result1', result1)

cv2.imshow('original', img)

cv2.waitKey(0)
cv2.destroyAllWindows()