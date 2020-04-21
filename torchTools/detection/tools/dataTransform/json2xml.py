from lxml.etree import Element, SubElement, tostring
# import pprint
from xml.dom.minidom import parseString
import os
import sys
from tqdm import tqdm
import json
import cv2

def glob_format(path,base_name = False):
    print('--------pid:%d start--------------' % (os.getpid()))
    fmt_list = (".json",) # '.jpg', '.jpeg', '.png',
    fs = []
    if not os.path.exists(path):return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:fs.append(item)
    print('--------pid:%d end--------------' % (os.getpid()))
    return fs

def save_xml(jdata,save_path):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = jdata.get("folder")

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = jdata.get("filename")

    node_filename = SubElement(node_root, 'source')
    node_filename.text = "ILSVRC_2013" #jdata.get("filename")

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(jdata.get("width"))

    node_height = SubElement(node_size, 'height')
    node_height.text = str(jdata.get("height"))

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(jdata.get("depth"))

    # <segmented>
    # node_segmented = SubElement(node_root, 'segmented')
    # node_segmented.text = '0'

    # <object>
    for object in jdata["object"]:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = object[0]
        # node_pose = SubElement(node_object, 'pose')
        # node_pose.text = 'Unspecified'
        # node_truncated = SubElement(node_object, 'truncated')
        # node_truncated.text = '0'
        # node_difficult = SubElement(node_object, 'difficult')
        # node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(object[1])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(object[2])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(object[3])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(object[4])

    xml = tostring(node_root, pretty_print=True)  #格式化显示，该换行的换行
    dom = parseString(xml)

    # print (xml)
    with open(os.path.join(save_path,jdata.get("filename").replace(".jpg",".xml",1)),'w') as fp: # 名字要与图片名一样
        dom.writexml(fp,encoding="utf-8")

if __name__=="__main__":
    # jdata = {"folder": "VOC2007", "filename": "000001.jpg", "width": 500, "height": 375, "depth": 3,
    #          "object": [["dog", 99, 358, 135, 375], ["cat", 99, 358, 135, 375]]}
    #
    # save_xml(jdata,"./")
    assert len(sys.argv)>1,"python3 json2xml.py ./zhaohang/01"
    json_paths=glob_format(sys.argv[1]) # "./zhaohang/01"
    for path in tqdm(json_paths):
        try:
            jdata_org=json.load(open(path))
            jdata={}
            jdata.update({"folder":os.path.basename(os.path.dirname(os.path.dirname(path)))})
            jdata.update({"filename": os.path.basename(path).replace(".json",".jpg")})
            img_path=path.replace(".json",".jpg").replace("bbox","images")
            h,w,c=cv2.imread(img_path,cv2.IMREAD_COLOR).shape
            jdata.update({"width": w,"height":h,"depth":c})
            _object=[]
            for bbox in jdata_org:
                temp=[]
                temp.append("face")
                temp.extend(list(map(int,bbox["bbox"])))
                _object.append(temp)

            jdata.update({"object":_object})
            save_path=os.path.dirname(path).replace("bbox","xml")
            if not os.path.exists(save_path):os.makedirs(save_path)
            save_xml(jdata, save_path)
        except Exception as e:
            print(e)
            print(path)