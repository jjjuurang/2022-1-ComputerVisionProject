import cv2
import numpy as np
from matplotlib import pyplot as plt

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

gx = []
gy = []

mag = []
angle = []

for i in range(8):
    if i in range(4):
        patch_image.append(simage[c[2 * i] - 4:c[2 * i] + 5, c[2 * i + 1] - 4:c[2 * i + 1] + 5])
    else:
        patch_image.append(simage2[c[2 * i] - 4:c[2 * i] + 5, c[2 * i + 1] - 4:c[2 * i + 1] + 5])

    patch_image[i] = np.float32(patch_image[i]) / 255.0

    gx.append(cv2.Sobel(patch_image[i], cv2.CV_32F, 1, 0, ksize=1))
    gy.append(cv2.Sobel(patch_image[i], cv2.CV_32F, 0, 1, ksize=1))

    mag.append((cv2.cartToPolar(gx[i], gy[i], angleInDegrees=True))[0])
    angle.append((cv2.cartToPolar(gx[i], gy[i], angleInDegrees=True))[1])

    angle[i][:, :, 0] = angle[i][:, :, 0] % 180
    angle[i][:, :, 1] = angle[i][:, :, 1] % 180
    angle[i][:, :, 2] = angle[i][:, :, 2] % 180

bin = [0, 20, 40, 60, 80, 100, 120, 140, 160]

countb = np.zeros((8,9))
countg = np.zeros((8,9))
countr = np.zeros((8,9))

for m in range(8):
    for i in range(9):
        for j in range(9):
            for k in range(8):
                if angle[m][i, j, 0] == bin[k]:
                    countb[m,k] = countb[m,k] + mag[m][i, j, 0]
                elif bin[k] < angle[m][i, j, 0] < bin[k + 1]:
                    countb[m,k] = countb[m,k] + ((bin[k + 1] - angle[m][i, j, 0]) / 20) * mag[m][i, j, 0]
                    countb[m,k + 1] = countb[m,k + 1] + ((angle[m][i, j, 0] - bin[k]) / 20) * mag[m][i, j, 0]
                elif 160 < angle[m][i, j, 0]:
                    countb[m,0] = countb[m,0] + ((angle[m][i, j, 0] - bin[8]) / 20) * mag[m][i, j, 0]
                    countb[m,8] = countb[m,8] + ((180 - angle[m][i, j, 0]) / 20) * mag[m][i, j, 0]

for m in range(8):
    for i in range(9):
        for j in range(9):
            for k in range(8):
                if angle[m][i, j, 1] == bin[k]:
                    countg[m,k] = countg[m,k] + mag[m][i, j, 1]
                elif bin[k] < angle[m][i, j, 1] < bin[k + 1]:
                    countg[m,k] = countg[m,k] + ((bin[k + 1] - angle[m][i, j, 1]) / 20) * mag[m][i, j, 1]
                    countg[m,k + 1] = countg[m,k + 1] + ((angle[m][i, j, 1] - bin[k]) / 20) * mag[m][i, j, 1]
                elif 160 < angle[m][i, j, 1]:
                    countg[m,0] = countg[m,0] + ((angle[m][i, j, 1] - bin[8]) / 20) * mag[m][i, j, 1]
                    countg[m,8] = countg[m,8] + ((180 - angle[m][i, j, 1]) / 20) * mag[m][i, j, 1]

for m in range(8):
    for i in range(9):
        for j in range(9):
            for k in range(8):
                if angle[m][i, j, 2] == bin[k]:
                    countr[m,k] = countr[m,k] + mag[m][i, j, 2]
                elif bin[k] < angle[m][i, j, 2] < bin[k + 1]:
                    countr[m,k] = countr[m,k] + ((bin[k + 1] - angle[m][i, j, 2]) / 20) * mag[m][i, j, 2]
                    countr[m,k + 1] = countr[m,k + 1] + ((angle[m][i, j, 2] - bin[k]) / 20) * mag[m][i, j, 2]
                elif 160 < angle[m][i, j, 2]:
                    countr[m,0] = countr[m,0] + ((angle[m][i, j, 2] - bin[8]) / 20) * mag[m][i, j, 2]
                    countr[m,8] = countr[m,8] + ((180 - angle[m][i, j, 2]) / 20) * mag[m][i, j, 2]

print("그래디언트 히스토그램의 가로축은 각도 0도, 20도, 40도, 60도, 80도, 100도, 120도, 160도를 의미")
f, axes = plt.subplots(8,3)
f.set_size_inches((20,15))
plt.subplots_adjust(wspace=0.3,hspace=0.3)
for i in range(8):
    axes[i, 0].bar(bin, countb[i, :], width=1)
    axes[i, 1].bar(bin, countg[i, :], width=1)
    axes[i, 2].bar(bin, countr[i, :], width=1)
plt.show()

print("Blue 채널에 대한 1st 이미지의 0,1,2,3 패치와 2nd 이미지의 0,1,2,3 패치의 그래디언트 히스토그램 8x9")
print(countb)
print("____________________________________________________________________________________________________________________________________________________")
print("Green 채널에 대한 1st 이미지의 0,1,2,3 패치와 2nd 이미지의 0,1,2,3 패치의 그래디언트 히스토그램 8x9")
print(countg)
print("____________________________________________________________________________________________________________________________________________________")
print("Red 채널에 대한 1st 이미지의 0,1,2,3 패치와 2nd 이미지의 0,1,2,3 패치의 그래디언트 히스토그램 8x9")
print(countr)
print("____________________________________________________________________________________________________________________________________________________")

d = np.zeros((4,4))

for i in range(4):
    for j in range(4):
        for k in range(9):
            d[i, j] += (countb[i, k] - countb[j + 4, k]) * (countb[i, k] - countb[j + 4, k])
            d[i, j] += (countg[i, k] - countg[j + 4, k]) * (countg[i, k] - countg[j + 4, k])
            d[i, j] += (countr[i, k] - countr[j + 4, k]) * (countr[i, k] - countr[j + 4, k])

print("행: 1st 이미지의 0,1,2,3 패치, 열: 2nd 이미지의 0,1,2,3 패치")
print("각 채널(B,G,R)에 대한 그래디언트 히스토그램 거리의 합 4x4 행렬")
print(d)

d_ex=np.delete(d,np.argmin(d)//d.shape[1],axis=0)
d_ex=np.delete(d_ex,np.argmin(d)%d.shape[1],axis=1)

d_ex_ex=np.delete(d_ex,np.argmin(d_ex)//d_ex.shape[1],axis=0)
d_ex_ex=np.delete(d_ex_ex,np.argmin(d_ex)%d_ex.shape[1],axis=1)

d_ex_ex_ex=np.delete(d_ex_ex,np.argmin(d_ex_ex)//d_ex_ex.shape[1],axis=0)
d_ex_ex_ex=np.delete(d_ex_ex_ex,np.argmin(d_ex_ex)%d_ex_ex.shape[1],axis=1)

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

center_point = np.reshape(c, (-1, 2))
center_x = center_point[:, 0]
center_y = center_point[:, 1]
center_x_left = center_x - 4
center_x_right = center_x + 5
center_y_bottom = center_y - 4
center_y_top = center_y + 5
center_tl = np.array([center_x_left, center_y_top])
center_br = np.array([center_x_right, center_y_bottom])
center_tl = np.array([center_x_left, center_y_top]).T
center_br = np.array([center_x_right, center_y_bottom]).T

imagePatch3 = cv2.hconcat([imagePatch1, imagePatch2])

fig1_center_point = center_point[:4, :]
fig2_center_point = center_point[4:, :]

move_center_x = np.array([[500, 0], [500, 0], [500, 0], [500, 0]])
fig2_center_point = fig2_center_point + move_center_x

for i in range(len(answer)):
    cv2.line(imagePatch3, fig1_center_point[int(answer[i, :][0]), :], fig2_center_point[int(answer[i, :][1]), :],
             (0, 0, 0), 2)

cv2.imshow('concatenate', imagePatch3)
cv2.waitKey(0)

print("____________________________________________________________________________________________________________________________________________________")
print("1st 이미지의 0 패치와 2nd 이미지의 패치들의 그래디언트 히스토그램 거리 오름차순:",d[0,:].argsort())
print("1st 이미지의 1 패치와 2nd 이미지의 패치들의 그래디언트 히스토그램 거리 오름차순:",d[1,:].argsort())
print("1st 이미지의 2 패치와 2nd 이미지의 패치들의 그래디언트 히스토그램 거리 오름차순:",d[2,:].argsort())
print("1st 이미지의 3 패치와 2nd 이미지의 패치들의 그래디언트 히스토그램 거리 오름차순:",d[3,:].argsort())
print("____________________________________________________________________________________________________________________________________________________")
print("2nd 이미지의 0 패치와 1st 이미지의 패치들의 그래디언트 히스토그램 거리 오름차순:",d[:,0].argsort())
print("2nd 이미지의 1 패치와 1st 이미지의 패치들의 그래디언트 히스토그램 거리 오름차순:",d[:,1].argsort())
print("2nd 이미지의 2 패치와 1st 이미지의 패치들의 그래디언트 히스토그램 거리 오름차순:",d[:,2].argsort())
print("2nd 이미지의 3 패치와 1st 이미지의 패치들의 그래디언트 히스토그램 거리 오름차순:",d[:,3].argsort())
print("____________________________________________________________________________________________________________________________________________________")
print("1st 이미지의 0 패치와 최소 거리인 2nd 이미지의 패치:",np.argmin(d[0,:]))
print("1st 이미지의 1 패치와 최소 거리인 2nd 이미지의 패치:",np.argmin(d[1,:]))
print("1st 이미지의 2 패치와 최소 거리인 2nd 이미지의 패치:",np.argmin(d[2,:]))
print("1st 이미지의 3 패치와 최소 거리인 2nd 이미지의 패치:",np.argmin(d[3,:]))
print("____________________________________________________________________________________________________________________________________________________")
print("2nd 이미지의 0 패치와 최소 거리인 1st 이미지의 패치:",np.argmin(d[:,0]))
print("2nd 이미지의 1 패치와 최소 거리인 1st 이미지의 패치:",np.argmin(d[:,1]))
print("2nd 이미지의 2 패치와 최소 거리인 1st 이미지의 패치:",np.argmin(d[:,2]))
print("2nd 이미지의 3 패치와 최소 거리인 1st 이미지의 패치:",np.argmin(d[:,3]))
