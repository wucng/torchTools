from lxml import etree
import csv
import cv2
import os
from tqdm import tqdm
import glob

colors={'face':(0,255,0),'non_motor_vehicle':(84,134,181),'vehicle':(146,61,146)}

def parsing_xml(xml_path):
    content = []
    xml_file = etree.parse(xml_path)
    root_node = xml_file.getroot()
    for sub_node in root_node:
        # if sub_node.tag == 'filename':
        #     content.append(sub_node.text)
        if sub_node.tag == 'object':
            # content.append(sub_node[0].text)
            # bboxs4个坐标
            bbox = []
            bbox.append(sub_node[0].text)
            bbox.append(int(sub_node[1][0].text))
            bbox.append(int(sub_node[1][2].text))
            bbox.append(int(sub_node[1][1].text))
            bbox.append(int(sub_node[1][3].text))
            content.append(bbox)

    return content

def draw(path):
    xml_path=path
    img_path=path.replace('Annotations','JPEGImages').replace('.xml','.jpg')

    img=cv2.imread(img_path)
    content=parsing_xml(xml_path)
    for d in content:
        cv2.rectangle(img,(d[1],d[2]),(d[3],d[4]),colors[d[0]],2,16)
        # cv2.putText(img,'%s'%(d[0][:3]),(d[1],d[2]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

    # cv2.putText(img, d[0], (60, 50),
    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

    cv2.imwrite(img_path,img,[int( cv2.IMWRITE_JPEG_QUALITY), 100])

def glob_format(path,fmt_list = ('.xml',),base_name = False):
    fs = []
    if not os.path.exists(path):return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                os.remove(item) # 删除掉不满足的文件（也许是非图片）
                continue
            if base_name:fs.append(os.path.splitext(file)[0])
            else:fs.append(item)
    return fs

xml_paths=glob_format('./Annotations')

for path in tqdm(xml_paths):
    draw(path)

