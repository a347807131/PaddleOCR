from tools.infer import predict_system
from tools.infer.utility import parse_args
import numpy as np
import os


if __name__ == '__main__':
    det_dir = 'inference/det_model'
    rec_dir = './pretrain_models/chinese_cht_PP-OCRv3_rec_train/inference'
    drop_score = 0.5
    img_dir = "D:/datasets/chengdu/1"

    args = parse_args()
    args.drop_score = drop_score
    args.draw_img_save_dir = "output/savedResult"
    args.image_dir = img_dir
    # 更改字典路径: 使用自己训练的识别模型时需要添加下面一条代码
    args.rec_char_dict_path = 'ppocr/utils/dict/chinese_cht_dict.txt'
    args.det_model_dir = det_dir
    args.rec_model_dir = rec_dir
    predict_system.main(args)

    # python
    # tools / train.py
    # - c configs / det / ch_PP - OCRv3 / ch_PP - OCRv3_det_student.yml
    # - o Global.pretrained_model =./ pretrain_models / ch_PP - OCRv3_det_distill_train / best_accuracy
    # Global.use_amp = True
    # Global.scale_loss = 1024.0
    # Global.use_dynamic_loss_scaling = True

    # 导出模型
    # 加载配置文件`det_mv3_db.yml`，从`output/det_db`目录下加载`best_accuracy`模型，inference模型保存在`./output/det_db_inference`目录下
    # python3
    # tools / export_model.py - c
    # configs / det / det_mv3_db.yml - o
    # Global.pretrained_model = "./output/det_db/best_accuracy"
    # Global.save_inference_dir = "./output/det_db_inference/"


from tools import infer_det
from tools.infer import predict_system, predict_det, utility
from tools.infer import predict_system
from tools.infer.predict_system import TextSystem
from tools.infer.utility import parse_args
import numpy as np
import os

# if __name__ == '__main__':
#
#     args = utility.parse_args()
#     args.draw_img_save_dir = './output/savedResult'
#     # 更改字典路径: 使用自己训练的识别模型时需要添加下面一条代码
#     args.rec_char_dict_path = 'ppocr/utils/dict/chinese_cht_dict.txt'
#     args.det_model_dir = 'inference/det_model'
#     args.rec_model_dir = 'output/v3_chinese_cht_mobile/inference'
#     args.image_dir = "C:/Users/Gatsby/datasets/det/1"
#     text_sys = TextSystem(args)
#     pass
