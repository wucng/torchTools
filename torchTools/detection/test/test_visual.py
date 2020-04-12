try:
    from .. import visual
except:
    import sys
    sys.path.append("..")
    import visual
import numpy as np
import cv2,os
from PIL import Image
import matplotlib.pyplot as plt

im_name = "./2.jpg"

def test_matplot():
    """
    # [[x1,y1,w,h],]
    visual.matplot.draw_img(im_name, [[73, 35, 326 - 73, 467 - 35], [333, 188, 560 - 333, 469 - 188]],
                            ["dog", "cat"],False,0.3)
    """

    """
    # [[x1,y1,x2,y2],]
    # im = plt.imread(im_name)
    # visual.matplot.draw_bbox(im, [[73, 35, 326, 467], [333, 188, 560, 469]], ["dog", "cat"],useMask=False,alpha=0.3)
    """

    """
    im = plt.imread(im_name)
    bbox = [73, 35, 326, 467]
    mask = np.zeros(im.shape[:2], np.uint8)
    mask[35:467, 73:326] = 1
    visual.matplot.draw_mask(im, [mask], [bbox],alpha=0.2) # 画mask有问题
    # """

    # """
    im = plt.imread(im_name)
    mask = np.zeros(im.shape[:2], np.uint8)
    mask[35:467, 73:326] = 1
    color_mask = [255, 0, 0]  # RGB
    color_mask = np.asarray(color_mask, np.uint8).reshape([1, 3])
    im = visual.matplot.vis_mask(im, mask, color_mask, 0.5, True)
    plt.imshow(im) # ,cmap="gray"
    plt.show()
    # """

def test_pillow():
    # 显示中文需要字体 "msyh.ttc" 可把 C:\Windows\Fonts\Microsoft YaHei UI(简体) 复制到当前目录即可
    font_path = "./msyh.ttc" if os.path.exists("./msyh.ttc") else "./FiraMono-Medium.otf"
    img = cv2.imread(im_name)
    bboxes = [[73, 35, 326, 467], [333, 188, 560, 469]]
    labels = ["狗狗:0.98", "cat"]
    # img = visual.pillow.vis_class_cn(img, bboxes[0], labels[0], 15, False,font_path=font_path)
    # img = visual.pillow.vis_class_cn1(img, bboxes[0], labels[0], 0.8, False,font_path=font_path)
    # img = visual.pillow.vis_class_cn2(img, bboxes[0], labels[0], 0.8,font_path=font_path)

    # 多行显示
    img = visual.pillow.putTextListOnCVImage(img,['中文','英文','法文'],(10,10),(255,255,255),font_path=font_path)


    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_opencv():
    img = cv2.imread(im_name)
    bboxes = [[73, 35, 326, 467], [333, 188, 560, 469]]
    labels = ["dog:0.98", "cat"]

    # img = visual.opencv.vis_rect(img, bboxes[0], labels[0], 0.5)
    # img = visual.opencv.vis_rect(img, bboxes[0], labels[0], 0.5,inside=False,useMask=False)

    # img = visual.opencv.vis_class(img, bboxes[0], labels[0], 0.5)

    mask = np.zeros(img.shape[:2], np.uint8)
    mask[35:467, 73:326] = 1
    color_mask = [255, 0, 0]  # RGB
    color_mask = np.asarray(color_mask, np.uint8).reshape([1, 3])
    img = visual.opencv.vis_mask(img,mask,color_mask,0.3,True)

    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    # test_matplot()
    # test_pillow()
    test_opencv()

