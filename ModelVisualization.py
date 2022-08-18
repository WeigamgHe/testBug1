import os.path
import sys
import torch
import json
import createModel as cm
import argparse
import hiddenlayer as hl
import base_net
import torchvision.models as md
from pdf2image import convert_from_path
from torchviz import make_dot
from grad_cam import __model__
import torchvision.models
from base_net import *

import grad_cam as gc
sys.path.append(r'C:\Users\axin\Desktop\DL\cam_draw\deep-learning-for-image-processing\Test\grad_cam')
sys.path.append(r'C:\Users\axin\Desktop\DL\cam_draw\deep-learning-for-image-processing\Test\grad_cam'
                r'\ModelVisualizationImageRes')
sys.path.append(r'C:\Users\axin\anaconda3\Lib\site-packages\torchviz')

dir = r'C:\Users\axin\Desktop\DL\cam_draw\deep-learning-for-image-processing\Test\grad_cam' \
      r'\ModelVisualizationImageRes'

lib__bin = r'C:\Users\axin\anaconda3\Lib\site-packages\poppler-0.67.0_x86\poppler-0.67.0\bin'

outputfile_path = r'C:\Users\axin\Desktop\DL\cam_draw\deep-learning-for-image-processing\Test' \
                  r'\grad_cam\ModelVisualizationImageRes'

parser = argparse.ArgumentParser(description='create Model Train.')

parser.add_argument('--config',
                    default=r'C:\Users\\axin\Desktop\DL\cam_draw\deep-learning-for-image-processing\pytorch_classification\grad_cam\config.json')

class load_json():
    def __init__(self,json):
        """
        json_path 是config.json的路径！！
        """
        self.outputPath = None
        self.KernalVisualizationConfig = None
        self.model = None
        self.datasetConfig = None
        self.dataset = None
        self.json_path = json
        self.parse_json(json)

    def parse_json(self, json_path):
        """
        读取config.json来初始化类的json信息
        """
        print('\n\n json_path = ', json_path)
        with open(json_path, "r", encoding="utf-8") as fr:
            params = json.load(fr)

        self.initMember(params)

    def initMember(self, params):
        """
        初始化类成员
        """
        self.dataset = params.get("dataset")
        self.datasetConfig = params.get("datasetConfig")
        self.model = params.get("model")
        self.KernalVisualizationConfig = params.get("KernalVisualizationConfig")
        self.outputPath = params.get("outputPath")

    def load(self):
        """
        name 是模型的名字
        返回一个json指定的模型model
        """
        pth_files = self.model.get("modelStructurePath")
        model = __model__.get(self.model.get("name"))(pretrained=True)
        model.load_state_dict(torch.load(pth_files))
        return model

if __name__ == '__main__':
    args = parser.parse_args()
    obj = load_json(args.config)
    model = obj.load()
    hl_graph = hl.build_graph(model, torch.zeros([3, 3, 224, 224]))
    hl_graph.theme = hl.graph.THEMES['blue'].copy()
    hl_graph.save('./ModelVisualizationImageRes/Resnet18.png', format='png')
    print('Have done!')


