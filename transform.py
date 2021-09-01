# 身份证的模版技术
from PIL.Image import radial_gradient
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from math import *
import tqdm

img_path="./data/img/"

range_offset=20
imgs=os.listdir(img_path)
num=0
# for cur_image in imgs:
#     if num==10:
#         break
#     num+=1
# print("正在处理第{}张图片".format(num))

# img=cv2.imread(os.path.join(img_path, cur_image))
img=cv2.imread("./testimg.png")
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()

height=img.shape[0]
width=img.shape[1]
# print(width,height)

left_up=np.array([1, 1])
right_up=np.array([width-1, 1])
right_bottom=np.array([width-1, height-1])
left_bottom=np.array([height-1, 1])

left_up_n=left_up + (np.random.rand(2)-0.5)*range_offset
left_up_n[0]=int(max(left_up[0], left_up_n[0]))
left_up_n[1]=int(max(left_up[1], left_up_n[1]))
left_up_n=left_up_n.astype(np.int32)
right_up_n=right_up + (np.random.rand()-0.5)*range_offset
right_up_n[0]=int(min(right_up[0], right_up_n[0]))
right_up_n[1]=int(max(right_up[1], right_up_n[1]))
right_up_n=right_up_n.astype(np.int32)
right_bottom_n=right_bottom + (np.random.rand()-0.5)*range_offset
right_bottom_n[0]=int(min(right_bottom[0], right_bottom_n[0]))
right_bottom_n[1]=int(min(right_bottom[1], right_bottom_n[1]))
right_bottom_n=right_bottom_n.astype(np.int32)
left_bottom_n=left_bottom + (np.random.rand()-0.5)*range_offset
left_bottom_n[0]=int(max(left_bottom[0], left_bottom_n[0]))
left_bottom_n[1]=int(min(left_bottom[1], left_bottom_n[1]))
left_bottom_n=left_bottom_n.astype(np.int32)

# pts1 = np.float32([left_up, right_up, right_bottom, left_bottom])
# pts2 = np.float32([left_up_n, right_up_n, right_bottom_n, left_bottom_n])
# pts1 = np.float32([left_up, right_up, left_bottom, right_bottom, ])
# pts2 = np.float32([left_up_n, right_up_n, left_bottom_n, right_bottom_n, ])
pts1 = np.float32([left_bottom, right_bottom, left_up, right_up])
pts2 = np.float32([left_bottom_n, right_bottom_n, left_up_n, right_up_n])
# pts1 = np.float32([[1, 100], [100, 100], [1, 1], [100, 1]])
# pts2 = np.float32([[2, 200], [200, 200], [1, 1], [200, 1]])
print(left_up, right_up, right_bottom, left_bottom)
print(left_up_n, right_up_n, right_bottom_n, left_bottom_n)
# 生成透视变换矩阵；进行透视变换
M = cv2.getPerspectiveTransform(pts1, pts2)
# print(M)
W_cols,H_rows=4000,3000
dst = cv2.warpPerspective(img, M, dsize=(W_cols,H_rows))
# dst = cv2.warpPerspective(img, M)

# plt.savefig(img_path+"new"+cur_image)
# plt.savefig("./cur/"+"new"+cur_image)

plt.imshow(dst, cmap='gray', interpolation='bicubic')
plt.savefig("newtestimg400.png")
# plt.show()
    