from tools import infer_det
from tools.infer import predict_system, predict_det, utility
from tools.infer import predict_system
from tools.infer.predict_system import TextSystem
from tools.infer.utility import parse_args
import numpy as np
import os

if __name__ == '__main__':

    args = utility.parse_args()
    args.draw_img_save_dir = './output/savedResult'
    # 更改字典路径: 使用自己训练的识别模型时需要添加下面一条代码
    args.rec_char_dict_path = 'ppocr/utils/dict/chinese_cht_dict.txt'
    args.det_model_dir = 'inference/det_model'
    args.rec_model_dir = 'output/v3_chinese_cht_mobile/inference'
    args.image_dir = "C:/Users/Gatsby/datasets/det/1"
    text_sys = TextSystem(args)
    pass
