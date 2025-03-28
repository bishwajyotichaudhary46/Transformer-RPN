import torch
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET

class LoadData:
    def __init__(self, img_dir, xml_dir, label2idx):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.label2idx = label2idx
        self.img_infos = []

    def load(self):
        xml_files = [os.path.join(self.xml_dir, dir, file) for dir in os.listdir(self.xml_dir) for file in os.listdir(os.path.join(self.xml_dir, dir))]
        for file in tqdm(xml_files, desc='Processing XML files'):
            img_info = {}
            img_info['id'] = os.path.basename(file).split('.xml')[0]
            xml_info = ET.parse(file)
            root = xml_info.getroot()
            size = root.find('size')
            folder = file.split('/')[3]
            img_info['image'] = os.path.join(self.img_dir, folder,'{}.jpg'.format(img_info['id']))
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_info['height'] = height
            img_info['width'] = width
            detections = []
        

            for obj in xml_info.findall('object'):
                det = {}
                label = self.label2idx[obj.find('name').text]
                if obj.find('name').text == 'leaf_blight':
                    print(img_info['id'])
                bbox_info = obj.find('bndbox')
                bbox = [
                    int(float(bbox_info.find('xmin').text))-1,
                    int(float(bbox_info.find('ymin').text))-1,
                    int(float(bbox_info.find('xmax').text))-1,
                    int(float(bbox_info.find('ymax').text))-1
                ]
                det['label'] = label
                det['bbox'] = bbox
                detections.append(det)
            
            img_info['detections'] = detections
            self.img_infos.append(img_info)
        return self.img_infos