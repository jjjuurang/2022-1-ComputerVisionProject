import cv2
import numpy as np
from matplotlib import pyplot as plt

#import skimage.io, skimage.color
#import matplotlib.pyplot
#import HOG

image = cv2.imread('1st.jpg')
simage = cv2.resize(image, dsize=(500,500), interpolation=cv2.INTER_AREA)

image2 = cv2.imread('2nd.jpg')
simage2 = cv2.resize(image2, dsize=(500,500), interpolation=cv2.INTER_AREA)

c = []

def onMouse(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN:
        c.append(x)
        c.append(y)

cv2.imshow('1st', simage)
cv2.moveWindow(winname='1st', x=50, y=50)
cv2.setMouseCallback('1st', onMouse)

cv2.imshow('2nd', simage2)
cv2.moveWindow(winname='2nd', x=1000, y=50)
cv2.setMouseCallback('2nd', onMouse)

cv2.waitKey(0)
print(c)

imagePatch1 = simage.copy()
for i in range(4):
    cv2.rectangle(imagePatch1, (c[2 * i] - 4, c[2 * i + 1] - 4), (c[2 * i] + 5, c[2 * i + 1] + 5), (255, 0, 0), thickness=2)
cv2.putText(imagePatch1, "0", (c[0] - 4 - 4, c[1] - 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.putText(imagePatch1, "1", (c[2] - 4 - 4, c[3] - 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.putText(imagePatch1, "2", (c[4] - 4 - 4, c[5] - 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.putText(imagePatch1, "3", (c[6] - 4 - 4, c[7] - 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imshow('patches of 1st image', imagePatch1)

imagePatch2 = simage2.copy()
for i in range(4):
    cv2.rectangle(imagePatch2, (c[2 * i + 8] - 4, c[2 * i + 1 + 8] - 4), (c[2 * i + 8] + 5, c[2 * i + 1 + 8] + 5), (0, 255, 0), thickness=2)
cv2.putText(imagePatch2, "0", (c[8] - 4 - 4, c[9] - 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(imagePatch2, "1", (c[10] - 4 - 4, c[11] - 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(imagePatch2, "2", (c[12] - 4 - 4, c[13] - 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(imagePatch2, "3", (c[14] - 4 - 4, c[15] - 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('patches of 2nd image', imagePatch2)

cv2.waitKey(0)

print(c)
print("1st 이미지에서 클릭한 좌표 4개")
print("0: (",c[0],",",c[1],")")
print("1: (",c[2],",",c[3],")")
print("2: (",c[4],",",c[5],")")
print("3: (",c[6],",",c[7],")")

print("2nd 이미지에서 클릭한 좌표 4개")
print("0: (",c[8],",",c[9],")")
print("1: (",c[10],",",c[11],")")
print("2: (",c[12],",",c[13],")")
print("3: (",c[14],",",c[15],")")

patch_image = []

for i in range(8):
    if i in range(4):
        patch_image.append(simage[c[2 * i] - 4:c[2 * i] + 5, c[2 * i + 1] - 4:c[2 * i + 1] + 5])


    else:
        patch_image.append(simage2[c[2 * i] - 4:c[2 * i] + 5, c[2 * i + 1] - 4:c[2 * i + 1] + 5])


descriptor = cv2.HOGDescriptor((9,9), (2,2), (1,1), (1,1), 9)

hog = [];

for i in range(len(patch_image)):
    hog_1 = descriptor.compute(patch_image[i])
    hog.append(hog_1)

hist =[];

for i in range(len(hog)):
    hist_1, _ = np.histogram(hog[i],100,[0.1,1])
    hist.append(hist_1.astype(np.float32))


d = np.zeros((4,4))

for i in range(4):
    for j in range(4):
        d[i, j] += cv2.compareHist(hist[i],hist[j+4], method = cv2.HISTCMP_BHATTACHARYYA)
        
        
        
d_ex=np.delete(d,np.argmin(d)//d.shape[1],axis=0)
d_ex=np.delete(d_ex,np.argmin(d)%d.shape[1],axis=1)

d_ex_ex=np.delete(d_ex,np.argmin(d_ex)//d_ex.shape[1],axis=0)
d_ex_ex=np.delete(d_ex_ex,np.argmin(d_ex)%d_ex.shape[1],axis=1)

d_ex_ex_ex=np.delete(d_ex_ex,np.argmin(d_ex_ex)//d_ex_ex.shape[1],axis=0)
d_ex_ex_ex=np.delete(d_ex_ex_ex,np.argmin(d_ex_ex)%d_ex_ex.shape[1],axis=1)

# 패치 매칭시키기

answer=np.zeros((4,2))

a,b=np.where(d==np.min(d))

answer[0,0]=a[0]
answer[0,1]=b[0]

a,b=np.where(d==np.min(d_ex))

answer[1,0]=a[0]
answer[1,1]=b[0]

a,b=np.where(d==np.min(d_ex_ex))

answer[2,0]=a[0]
answer[2,1]=b[0]

a,b=np.where(d==np.min(d_ex_ex_ex))

answer[3,0]=a[0]
answer[3,1]=b[0]

#시각화

center_point=np.reshape(c,(-1,2))
center_x=center_point[:,0]
center_y=center_point[:,1]
center_x_left = center_x - 4
center_x_right = center_x + 5
center_y_bottom = center_y - 4
center_y_top = center_y + 5
center_tl = np.array([center_x_left , center_y_top])
center_br = np.array([center_x_right , center_y_bottom])
center_tl = np.array([center_x_left , center_y_top]).T
center_br = np.array([center_x_right , center_y_bottom]).T

imagePatch3=cv2.hconcat([imagePatch1, imagePatch2])

fig1_center_point = center_point[:4,:]
fig2_center_point = center_point[4:,:]

move_center_x=np.array([[500,0],[500,0],[500,0],[500,0]])
fig2_center_point = fig2_center_point + move_center_x

for i in range(len(answer)):
    cv2.line(imagePatch3, fig1_center_point[int(answer[i,:][0]),:], fig2_center_point[int(answer[i,:][1]),:], (0,0,0), 2)
    
cv2.imshow('concatenate',imagePatch3)
cv2.waitKey(0)







