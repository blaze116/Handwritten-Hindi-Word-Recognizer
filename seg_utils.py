import numpy as np
import cv2
import math
import os

DEBUG = False


def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop

class Localizer:

    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.resize(self.image, (self.image.shape[1]//2, self.image.shape[0]//2))
        if DEBUG:
            show_img(self.image)
        self.shape = self.image.shape
        self.blur = cv2.GaussianBlur(self.image, (5,5), 0)

        self.blur = cv2.dilate(self.blur, np.ones((3, 3), np.uint8))
        for i in range(2):
            self.blur = cv2.erode(self.blur, np.ones((5, 5), np.uint8))
        self.thresh = cv2.adaptiveThreshold(self.blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 7)
        # _, self.thresh = cv2.threshold(self.blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if DEBUG:
            show_img(self.blur, "LOC_BLUR")
            show_img(self.thresh, "LOC_THRESH")

    def get_crop(self, thresh=5):
        crop = None
        contours, _ = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if DEBUG:
            show_img(self.thresh)
        # print(f"Number of contours : {len(contours)}")
        max_area = 0
        for cont in contours:
            cnt_area = cv2.contourArea(cont)
            if cnt_area > 0.01 * self.shape[0] * self.shape[1]:
                # x, y, w, h = cv2.boundingRect(cont)
                # crop_img = self.image[max(y-thresh, 0):y+h+thresh, max(x-thresh,0):x+w+thresh]
                rect = cv2.minAreaRect(cont)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # crop_img = crop_minAreaRect(self.image, rect)
                W = rect[1][0]
                H = rect[1][1]

                Xs = [i[0] for i in box]
                Ys = [i[1] for i in box]
                x1 = min(Xs)
                x2 = max(Xs)
                y1 = min(Ys)
                y2 = max(Ys)

                angle = rect[2]
                if angle < -45:
                    angle += 90

                # Center of rectangle in source image
                center = ((x1+x2)/2,(y1+y2)/2)
                # Size of the upright rectangle bounding the rotated rectangle
                size = (x2-x1+thresh, y2-y1+thresh)
                M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
                # Cropped upright rectangle
                cropped = cv2.getRectSubPix(self.image, size, center)
                cropped = cv2.warpAffine(cropped, M, size)
                croppedW = H if H > W else W
                croppedH = H if H < W else W
                # Final cropped & rotated rectangle
                crop_img = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
                if DEBUG:
                    show_img(crop_img)
                    print(crop_img.shape)
                if cnt_area > max_area and crop_img.shape[0] < 1.5*crop_img.shape[1]:
                    crop = crop_img
                    max_area = cnt_area
        return crop


class Segmenter:

    def __init__(self, image):

        self.image = image
        self.image = cv2.resize(self.image, (int(110*(image.shape[1]/image.shape[0])), 110))
        # print(self.image.shape)
        self.blur = cv2.GaussianBlur(self.image, (5, 5), 0)

        self.blur = cv2.dilate(self.blur, np.ones((3, 3), np.uint8))
        self.blur = cv2.erode(self.blur, np.ones((5, 5), np.uint8))
        self.blur = cv2.dilate(self.blur, np.ones((3, 3), np.uint8))
        for i in range(2):
            self.blur = cv2.erode(self.blur, np.ones((5, 5), np.uint8))

        # _, self.thresh = cv2.threshold(self.blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.thresh = cv2.adaptiveThreshold(self.blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 7)
        
        self.top_sum = self.get_topsum()
        self.bottom_sum = self.get_bottomsum()
        for _ in range(math.ceil(self.image.shape[0] / 40)):
            self.top_sum = cv2.erode(self.top_sum, np.ones((5, 5), np.uint8))

        self.tb_sum = cv2.bitwise_and(self.top_sum, self.bottom_sum)

        for _ in range(2):
            self.tb_sum = cv2.dilate(self.tb_sum, np.ones((3, 3), np.uint8))
        for _ in range(8):
            self.tb_sum = cv2.erode(self.tb_sum, np.ones((3, 3), np.uint8))
        
        self.tb_sum = np.array(cv2.threshold(self.tb_sum, 0, 255, cv2.THRESH_BINARY)[1], dtype=np.uint8) 
        if DEBUG:
            show_img(self.top_sum, "SEG_TOPSUM")
            show_img(self.bottom_sum, "SEG_BOTTOMSUM")
            show_img(self.tb_sum, "SEG_TBSUM")
            show_img(self.thresh, "SEG_THRESH")


    def get_topsum(self):
        ans = []
        temp_sum = self.thresh[0] * 0
        
        for i in range(self.thresh.shape[0]):
            temp = np.sum(self.thresh[:i, :], axis=0)/(i+1)
            ans.append(temp)

        ans = np.round(ans)
        ans = np.clip(ans, 0, 255)
        ret, ans = cv2.threshold(ans, 0, 255, cv2.THRESH_BINARY)
        return ans

    def get_bottomsum(self):
        ans = []
        temp_sum = np.sum(self.thresh, axis=0)
        
        for i in range(self.thresh.shape[0]):
            temp_sum -= self.thresh[i]
            ans.append(temp_sum/(i+1))

        ans = np.round(ans)
        ans = np.clip(ans, 0, 255)
        ret, ans = cv2.threshold(ans, 0, 127, cv2.THRESH_BINARY)
        return ans

    def segment(self, thresh=5):
        contours, _ = cv2.findContours(self.tb_sum, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # print(f"Number of contours : {len(contours)}")
        crops = []
        for cont in contours:
            if cv2.contourArea(cont) > 0.005 * self.image.shape[0] * self.image.shape[1]:
                x, y, w, h = cv2.boundingRect(cont)
                x = max(0, int(x - self.image.shape[0]/15))
                y = max(0, int(y - self.image.shape[0]/15))
                w += int(self.image.shape[0]/15)
                h += int(self.image.shape[0]/15)
                crop_img = self.image[max(y-thresh, 0):y+h+thresh, max(x-thresh,0):x+w+thresh]
                crops.append([crop_img, x])
        crops.sort(key=lambda x:x[1])
        return [el for el, _ in crops]
        
def show_img(image, win_name="IMG"):
    cv2.imshow(win_name, image)
    if cv2.waitKey(0) == ord('s'):
        cv2.imwrite(f"res/img_{len(os.listdir('res/'))+1}.jpg", image)
    cv2.destroyAllWindows()

def pad_image(image):
    wh = max(image.shape[0], image.shape[1])
    temp_img = np.zeros((wh, wh))
    temp_img[:image.shape[0], :image.shape[1]] = image
    kernel = np.ones((3, 3), np.uint8)
    temp_img = cv2.erode(temp_img, kernel)
    temp_img = cv2.dilate(temp_img, kernel)
    ker2= np.ones((3, 3), np.uint8)
    temp_img=cv2.morphologyEx(temp_img,cv2.MORPH_OPEN, ker2)
    return temp_img


mapping={
    1:"क", 2:"ख", 3:"घ",
    4:"च", 5:"छ", 6:"ज", 7:"ञ",
    8:"ट", 9:"ठ", 10:"ड", 11:"ढ",
    12:"त", 13:"थ", 14:"न",
    15:"प", 16:"फ", 17:"म",
    18:"र", 19:"ल", 20:"व", 21:"ष",
    22:"स", 23:"ह", 24:"क्ष", 25:"त्र", 26:"ज्ञ",
    27:"अ", 28:"इ", 29:"ई", 30:"उ", 31:"ऊ",
    32:"अं"}
