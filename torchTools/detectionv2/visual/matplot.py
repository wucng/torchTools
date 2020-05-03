# -*- coding:utf-8 -*-
"""
参考：https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
plt 线型，颜色 参考：https://www.cnblogs.com/darkknightzh/p/6117528.html

不能显示中文
"""
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

from PIL import ImageDraw, ImageFont,Image
from matplotlib import patches, patheffects
import numpy as np


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

# matplotlib 画框
def show_img(im, figsize=None, ax=None):
    if not ax:
        fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b,useMask=False,alpha=0.5):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:],
                                           fill=True if useMask else False,
                                           edgecolor='red', lw=2,
                                           alpha=alpha if useMask else 1.0))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold',
                   bbox=dict(
                       facecolor='black', alpha=1.0, pad=0, edgecolor='none'),
                   )
    draw_outline(text, 1)

def draw_img(im,boxes=[],labels=[],useMask=False,alpha=0.3,output_path="./test.jpg",saveImg=False):
    """
    boxes格式:
    不是 [[x1,y1,x2,y2],]
    也不是 [[x0,y0,w,h],]
    而是 [[x1,y1,w,h],]
    （x1,y1）左上角坐标
    （x0,y0）中心点坐标
    （x2,y2）右下角坐标
    （w,h）框的宽度与高度
    """
    if type(im) == str: # 传入图片路径
        im = Image.open(im).convert("RGB")
        # im = np.asarray(Image.open(im).convert("RGB"),np.uint8)
    if labels:
        assert len(boxes)==len(labels),"boxes {} and labels {} must be same".format(len(boxes),len(labels))
    ax = show_img(im)
    for i,box in enumerate(boxes):
        draw_rect(ax, box,useMask,alpha)
        if labels:
            draw_text(ax, box[:2], labels[i])

    if saveImg:plt.savefig(output_path)
    # fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    # for i, ax in enumerate(axes.flat):
    #     ...
    # plt.tight_layout() # 紧凑

    plt.show()
    plt.close('all')

# ---------------------------------------------------
def draw_bbox(im,bboxs=[],labels=[],dpi=200,useMask=False,alpha=0.3,output_path="./test.jpg",saveImg=False):
    """boxes格式:
    [[x1,y1,x2,y2],]"""

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    for i,bbox in enumerate(bboxs):
        # 画框
        ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1],
                                  fill=True if useMask else False, edgecolor='r',facecolor='r', #color='r'
                                  linewidth=2, alpha=alpha if useMask else 1.0,linestyle='--'))
        if labels:
            # 画文本
            text = labels[i]
            ax.text(
                bbox[0], bbox[1],
                text,
                verticalalignment='top',
                fontsize=14,
                family='serif',
                bbox=dict(
                    facecolor='black', alpha=1.0, pad=0, edgecolor='none'),
                color='white')

    if saveImg:fig.savefig(output_path, dpi=dpi)
    # plt.tight_layout()
    plt.show()
    plt.close('all')

def draw_mask(im,mask=[],bboxs=[],labels=[],dpi=200,alpha=0.3,output_path="./test.jpg",saveImg=False):
    """boxes格式:
        [[x1,y1,x2,y2],]"""
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    for i in range(len(mask)):
        # 画mask
        e=mask[i]
        # color_list = colormap(rgb=True) / 255
        # color_mask = color_list[mask_color_id % len(color_list), 0:3]
        color_mask=(0,255/255.,0/255.) # RGB

        _, contour, hier = cv2.findContours(
            e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for c in contour:
            polygon = Polygon(
                c.reshape((-1, 2)),
                fill=True, facecolor=color_mask,
                edgecolor='w', linewidth=1.2,
                alpha=alpha)
            ax.add_patch(polygon)

        if bboxs:
            # 画框
            bbox = bboxs[i]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              # fill=True,
                              edgecolor='r',  # color='r',
                              linewidth=2,
                              # alpha=box_alpha,
                              linestyle='--'))

        if labels:
            # 画文本
            text = labels[i]
            ax.text(
                bbox[0], bbox[1],
                text,
                verticalalignment='top',
                fontsize=14,
                family='serif',
                bbox=dict(
                    facecolor='black', alpha=box_alpha, pad=0, edgecolor='none'),
                color='white')


    if saveImg:fig.savefig(output_path, dpi=dpi)
    plt.show()
    plt.close('all')

def draw_keypoints():
    pass

# opencv 画
def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)