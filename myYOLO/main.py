import torch
from model import Yolov1, MyYolov1


if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 448, 448)

    model = MyYolov1()
    dummy_output1 = model(dummy_input)
    print(dummy_output1.shape)

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    dummy_output2 = model(dummy_input)
    print(dummy_output2.shape)

