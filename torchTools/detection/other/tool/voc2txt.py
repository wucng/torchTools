# -*- coding:utf-8 -*-
"""preprocess pascal_voc data
将voc2007数据转成txt格式，具体格式如下：

xxx/VOC2007/JPEGImages/007519.jpg 426,193,500,375,15,238,164,309,187,15,99,161,192,185,15
# image_path x1,y1,x2,y2,label,x1,y1,x2,y2,label
"""
import os
import xml.etree.ElementTree as ET
import struct
import numpy as np
import sys
from tqdm import tqdm
#
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#
# classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
#                'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
#                'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
#                'sofa': 17, 'train': 18, 'tvmonitor': 19}

sets=[('2007', 'train'), ('2007', 'val')]
classes=["head_shoulder"]

# YOLO_ROOT = os.path.abspath('/data')
# DATA_PATH = os.path.join(YOLO_ROOT, 'VOCdevkit2007')
# OUTPUT_PATH = os.path.join(YOLO_ROOT, 'train.txt')

assert len(sys.argv)>=3,"python3 voc2txt.py `pwd`/VOCdevkit ./train.txt"

DATA_PATH=sys.argv[1]
OUTPUT_PATH=sys.argv[2]

# 推荐
def parse_xml(xml_file):
    """parse xml_file

    Args:
      xml_file: the input xml file path

    Returns:
      image_path: string
      labels: list of [xmin, ymin, xmax, ymax, class]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # image_path = ''
    labels = []
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)

    # filename=root.find('filename').text
    filename=os.path.basename(xml_file).replace(".xml",".jpg")
    image_path = os.path.join(DATA_PATH, 'VOC2007/JPEGImages', filename)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes:  # or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        # bb = convert((w, h), b)
        labels.append([*b, cls_id])

    return image_path, labels

# 这种要求格式完全一致 不推荐
def parse_xml2(xml_file):
    """parse xml_file

    Args:
      xml_file: the input xml file path

    Returns:
      image_path: string
      labels: list of [xmin, ymin, xmax, ymax, class]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_path = ''
    labels = []

    for item in root:
        if item.tag == 'filename':
            image_path = os.path.join(DATA_PATH, 'VOC2007/JPEGImages', item.text)
        elif item.tag == 'object':
            obj_name = item[0].text
            obj_num = classes_num[obj_name]
            for i in range(len(item)):
                if item[i].tag=="bndbox":
                    xmin = int(item[i][0].text)
                    ymin = int(item[i][1].text)
                    xmax = int(item[i][2].text)
                    ymax = int(item[i][3].text)
                    labels.append([xmin, ymin, xmax, ymax, obj_num])
                    # break

    return image_path, labels

def convert_to_string2(image_path, labels):
    """convert image_path, lables to string
    Returns:
      string
    """
    out_string = ''
    out_string += image_path
    for label in labels:
        for index, i in enumerate(label):
            if index == 0:
                out_string += ' ' + str(i)
            else:
                out_string += ',' + str(i)

    out_string += '\n'

    return out_string

def convert_to_string(image_path, labels):
    """convert image_path, lables to string
    Returns:
      string
    """
    out_string = ''
    out_string += image_path
    for j,label in enumerate(labels):
        for index, i in enumerate(label):
            if j+index==0:
                out_string += ' ' + str(i)
            else:
                out_string += ',' + str(i)

    out_string += '\n'

    return out_string


def main():
    out_file = open(OUTPUT_PATH, 'w')

    xml_dir = DATA_PATH + '/VOC2007/Annotations/'

    xml_list = os.listdir(xml_dir)
    xml_list = [xml_dir + temp for temp in xml_list]

    for xml in tqdm(xml_list):
        try:
            image_path, labels = parse_xml(xml)
            record = convert_to_string(image_path, labels)
            out_file.write(record)
        except Exception as e:
            print(e)

    out_file.close()

if __name__ == '__main__':
    main()