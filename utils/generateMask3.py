import numpy as np
import math as m
import cv2


def generateMasks(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    maxVal= np.max(img)
    print("max value "+ str(np.max(img)))
    threshold = 0.85 * maxVal
    ret, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)
    print("mask.shape " + str(mask.shape))
    
    width, height = mask.shape
    mask_bin = mask>200
    count = 0
    ind = [0 for i in range(width*height)]
    for i in range(width):
        for j in range(height):
            if(mask_bin[i][j] == True):
                ind[count] = (j)*width + i
                count = count + 1
    ##end of for
    sal = ind[0:count-1]
    np.random.shuffle(sal)

    mask_1 = mask_bin < 0
    for i in range(int(m.floor((count-1)/3))):
        val = sal[i]
        row = m.floor(val/width)
        col = m.floor(val - (row * width))
        mask_1[int(m.floor(col))][int(m.floor(row))] = True
    mask_1 = mask_1.astype(np.uint8)
    #cv2.imwrite('data/mask/mask_1.jpg', mask_1)
    #plt.savefig('img_{}_mask1.jpg'.format(1))
    mask_2 = mask_bin <0
    for i in range(int(m.floor((count-1)/3+1)), int(m.floor(2*(count-1)/3))):
        val = sal[i]
        row = m.floor(val/width)
        col = m.floor(val - (row * width))
        mask_2[int(m.floor(col))][int(m.floor(row))] = True
    mask_2 = mask_2.astype(np.uint8)
    #cv2.imwrite('data/mask/mask_2.jpg', mask_2)
    #plt.savefig('img_{}_mask2.jpg'.format(1))
    mask_3 = mask_bin <0
    for i in range(int(m.floor(2*(count-1)/3+1)), int(m.floor((count-1)/1))):
        val = sal[i]
        row = m.floor(val/width)
        col = m.floor(val - (row * width))
        mask_3[int(m.floor(col))][int(m.floor(row))] = True
    mask_3 = mask_3.astype(np.uint8)
    #cv2.imwrite('data/mask/mask_3.jpg', mask_3)
    #plt.savefig('img_{}_mask3.png'.format(1))
    mask_4 = 1 - (mask_1 | mask_2 | mask_3)
    cv2.imwrite('data/mask_test_images/temp.jpg', mask_4)
    mask_4 = cv2.imread('data/mask_test_images/temp.jpg', cv2.IMREAD_COLOR)
    #cv2.imwrite('data/mask/mask_4_rgb.jpg', mask_4)
    print("mask_4.max "+ str(mask.max()))
    mask1 = cv2.resize(mask_4, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    return mask1
