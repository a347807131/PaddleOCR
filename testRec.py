from tools import infer_det
from tools.infer import predict_system,predict_det
from tools.infer import predict_system
from tools.infer.utility import parse_args
import numpy as np
import os


# 识别文字
def img_rec(book_id, img_dir, drop_score, det_dir, rec_dir):
    args = parse_args()
    args.drop_score = drop_score
    args.book_id = book_id
    args.draw_img_save_dir = './output/savedResult'
    args.image_dir = img_dir
    # 更改字典路径: 使用自己训练的识别模型时需要添加下面一条代码
    args.rec_char_dict_path = 'ppocr/utils/dict/chinese_cht_dict.txt'
    args.det_model_dir = det_dir
    args.rec_model_dir = rec_dir
    predict_system.main(args)

def img_det(book_id, img_dir, drop_score, det_dir, rec_dir):
    args = parse_args()
    args.drop_score = drop_score
    args.book_id = book_id
    args.draw_img_save_dir = './output/savedResult'
    args.image_dir = img_dir
    # 更改字典路径: 使用自己训练的识别模型时需要添加下面一条代码
    args.rec_char_dict_path = 'ppocr/utils/dict/chinese_cht_dict.txt'
    args.det_model_dir = det_dir
    args.rec_model_dir = rec_dir
    infer_det.main(args)

if __name__ == '__main__':
    # det_dir = './inference/ch_mobile_det_infer/'
    # det_dir = './inference/ch_ppocr_server_v2.0_det_infer/'
    det_dir = './inference/det_model/'
    # det_dir = './inference/ch_PP-OCRv3_det_slim_infer/'
    # rec_dir = './inference/my_ppocrv3_rec/'
    # rec_dir = './inference/ch_mobile_rec_infer/'
    # rec_dir = './inference/rec_model/'

    # det_dir = './output/det_r50_vd/inference'
    rec_dir = 'output/v3_chinese_cht_mobile/inference'
    thresh = 0.5
    book_id = '0001'
    img_det(book_id, "C:/Users/Gatsby/datasets/1948-02", thresh, det_dir, rec_dir)
