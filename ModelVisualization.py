import os.path
import sys
import torch
# import poppler
import base_net
from pdf2image import convert_from_path
from torchviz import make_dot
from grad_cam import __model__

sys.path.append(r'C:\Users\axin\Desktop\DL\cam_draw\deep-learning-for-image-processing\Test\grad_cam')
sys.path.append(r'C:\Users\axin\Desktop\DL\cam_draw\deep-learning-for-image-processing\Test\grad_cam'
                r'\ModelVisualizationImageRes')
sys.path.append(r'C:\Users\axin\anaconda3\Lib\site-packages\torchviz')


class ConNet(torch.nn.Module):
    def __init__(self):
        super(ConNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, 2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        output = self.out(x)
        return output


def getParameter():
    global dir, lib__bin, outputfile_path
    dir = r'C:\Users\axin\Desktop\DL\cam_draw\deep-learning-for-image-processing\Test\grad_cam' \
          r'\ModelVisualizationImageRes'
    lib__bin = r'C:\Users\axin\anaconda3\Lib\site-packages\poppler-0.67.0_x86\poppler-0.67.0\bin'
    outputfile_path = r'C:\Users\axin\Desktop\DL\cam_draw\deep-learning-for-image-processing\Test' \
                      r'\grad_cam\ModelVisualizationImageRes'


if __name__ == '__main__':
    # convnet = ConNet()
    # x = torch.randn(size=(1, 1, 28, 28))
    # y = convnet(x)

    # conv_viz = make_dot(y, params=dict(list(convnet.named_parameters()) + [("x", x)]))
    #
    # conv_viz.format = "png"
    #
    # conv_viz_directory = "data"
    # conv_viz.view()
    getParameter()

    model_name = "AlexNet"
    x = torch.rand(8, 3, 256, 512)
    model = base_net.AlexNet()
    # model = __model__.get(model_name)

    y = model(x)
    g = make_dot(y)

    book_path = os.path.join(str(dir), str(model_name + "_ok"))
    g.render(book_path, view=False)
    book = os.path.join(dir, str(model_name + "_ok.pdf"))

    pages = convert_from_path(book, dpi=600,
                              poppler_path=lib__bin,
                              output_file=outputfile_path)

    img_path = os.path.join(dir ,'res.jpeg')
    for page in pages:
        page.save(img_path)