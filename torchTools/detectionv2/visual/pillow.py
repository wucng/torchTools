'''
可以显示中文
显示中文需要字体 "msyh.ttc" 可把 C:\Windows\Fonts\Microsoft YaHei UI(简体) 复制到当前目录即可
FiraMono-Medium.otf 只能显示英文
'''
import PIL.Image
from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
import os

# 使用PIL画文本画底色和画框(字在框内,字在框上方)
def cv2ImgAddText(img, text,pos, textColor=(255, 255, 255), textSize=20,word_inside=False,font_path="./"):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    left, top=pos[0],pos[1]
    list_icon_size = 3
    padding = 3
    # draw = ImageDraw.Draw(img)
    image = Image.new('RGBA', img.size)
    draw = ImageDraw.Draw(image)

    draw.rectangle(pos, outline="red", width=2, fill=(99, 184, 255, 120))

    fontText = ImageFont.truetype(
        font_path, textSize, encoding="utf-8") # "msyh.ttf"
    # draw.rectangle((left, top-textSize-7,left+textSize,top),fill="black")
    font_w, font_h = fontText.getsize(text)
    # draw background
    if word_inside:
        # 字在框内
        draw.rectangle((left, top, left + font_w + list_icon_size + padding, top + font_h + padding), fill=(0, 0, 0, 255))
        draw.text((left+4, top), text, textColor, font=fontText)
    else:
        # 字在框上方
        offset=int(textSize*1.5)
        draw.rectangle((left, top-offset, left + font_w + list_icon_size + padding, top + font_h + padding-offset), fill=(0, 0, 0, 255))
        draw.text((left+4, top-offset), text, textColor, font=fontText,align="center")

    # draw.rectangle(pos,outline="red")
    # draw.rectangle(list(map(lambda x:x-1,pos)),outline="red") # fill="gray"
    # draw.rectangle(pos, outline="red",width=2,fill=(99 ,184 ,255, 120))

    img.paste(image, mask=image)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def vis_class_cn(img, pos, class_str="person", textSize = 20,word_inside=False,font_path="./"):
    """Visualizes the class."""
    txt = class_str
    img=cv2ImgAddText(img,txt,pos,textSize=textSize,word_inside=word_inside,font_path=font_path) # 15
    return img

# ---------------------------------------------------------------------------
# 使用PIL画文本画底色,画框都使用opencv(字在框内,字在框上方)
def cv2ImgAddText1(img, text,left, top, textColor=(255, 255, 255), textSize=20,word_inside=False,font_path="./"):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # left, top=pos[0],pos[1]
    list_icon_size = 3
    padding = 3
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        font_path, textSize, encoding="utf-8") # "msyh.ttf"
    # draw.rectangle((left, top-textSize-7,left+textSize,top),fill="black")
    font_w, font_h = fontText.getsize(text)
    # draw background
    if word_inside:
        # 字在框内
        draw.rectangle((left, top, left + font_w + list_icon_size + padding, top + font_h + padding), fill=(0, 0, 0, 215))
        draw.text((left+4, top), text, textColor, font=fontText)
    else:
        # 字在框上方
        offset=int(textSize*1.5)
        draw.rectangle((left, top-offset, left + font_w + list_icon_size + padding, top + font_h + padding-offset), fill=(0, 0, 0, 215))
        draw.text((left+4, top-offset), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def vis_class_cn1(img, pos, class_str="person", font_scale=0.35,word_inside=False,font_path="./"):
    """Visualizes the class."""
    # temp_GREEN=np.clip(np.asarray(_GREEN),0,255).astype(np.uint8).tolist()

    # x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    # back_tl = x0, y0 - int(1.2 * txt_h)-8
    # back_br = x0 + txt_w-48, y0
    # cv2.rectangle(img, back_tl, back_br, [0,0,0], -1) # _GREEN
    img=cv2ImgAddText1(img,txt,pos[0]+2,pos[1],textSize=txt_h,word_inside=word_inside,font_path=font_path) # 15
    cv2.rectangle(img,(pos[0],pos[1]),(pos[2],pos[3]),[0,0,255],2) # _GREEN
    return img

# ---------------------------------------------------------------------------
# 只使用PIL画文本,画框画底色都使用opencv(字在框上方)
def cv2ImgAddText2(img, text, left, top, textColor=(255, 255, 255), textSize=20,font_path="./"):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        font_path, textSize, encoding="utf-8") # "msyh.ttf"
    draw.text((left+5, top-int(textSize*1.5)), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def vis_class_cn2(img, pos, class_str="person", font_scale=0.35,font_path="./"):
    """Visualizes the class."""
    # temp_GREEN=np.clip(np.asarray(_GREEN),0,255).astype(np.uint8).tolist()

    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.2 * txt_h)-8
    back_br = x0 + txt_w-60, y0
    cv2.rectangle(img, back_tl, back_br, [0,0,0], -1) # _GREEN
    img=cv2ImgAddText2(img,txt,pos[0],pos[1],textSize=txt_h,font_path=font_path) # 15
    cv2.rectangle(img,(pos[0],pos[1]),(pos[2],pos[3]),[0,0,255],2) # _GREEN
    return img


# --------------------画多行文本---------------------------------------------
def cvImage2PIL(cvImage):
    cv2_im = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im


def pilImage2CV(pilImage):
    open_cv_image = np.array(pilImage)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def putTextListOnPILImage(pilImage, textInfoList, offset=(0,0), defaultColor=(255,255,255), fontSize=12,font_path="./"):
    image = Image.new('RGBA', pilImage.size)
    draw = ImageDraw.Draw(image)
    #draw = ImageDraw.Draw(pilImage)

    list_icon_size = 3
    padding = 3
    offset_y = 0

    for textInfo in textInfoList:
        if type(textInfo) == list:
            text, x, y, color = textInfo
        else:
            text = textInfo
            x,y = offset
            color = defaultColor

        y +=offset_y
        if color is None:
            color = defaultColor

        font = ImageFont.truetype(
            font_path, fontSize, encoding="utf-8")
        # if not type(text) == unicode:
        #     text = unicode(text, "utf-8")
        font_w, font_h = font.getsize(text)
        # draw background
        draw.rectangle((x, y, x + font_w + list_icon_size +padding, y + font_h+padding), fill=(0, 0, 0, 215))
        # draw list icon
        draw.rectangle((x, y+(font_h-list_icon_size)/2, x+list_icon_size, y + (font_h - list_icon_size) / 2+list_icon_size), fill=(0,255,0, 215))
        # draw text
        draw.text((x + list_icon_size +padding, y), text, color, font=font)
        offset_y += font_h + padding + padding

    pilImage.paste(image, mask=image)

    return pilImage


def putTextListOnCVImage(cvImage, textInfoList, offset=(0,0), defaultColor=(255,255,255), fontSize=12,font_path="./"):
    pilImage = cvImage2PIL(cvImage)
    pilImage = putTextListOnPILImage(pilImage, textInfoList, offset,  defaultColor, fontSize,font_path)
    cvImage = pilImage2CV(pilImage)
    return cvImage