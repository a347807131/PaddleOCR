from tools.infer import predict_rec, utility

if __name__ == '__main__':
    args = utility.parse_args()

    rec_dir = './pretrain_models/chinese_cht_PP-OCRv3_rec_train/inference'
    drop_score = 0.5
    img_dir = "D:/tar/tagged_vertical_3.6w/test"
    args.drop_score = drop_score
    # args.draw_img_save_dir = "output/savedResult"
    args.image_dir = img_dir
    # 更改字典路径: 使用自己训练的识别模型时需要添加下面一条代码
    args.rec_char_dict_path = './ppocr/utils/dict/chinese_cht_dict.txt'
    args.rec_model_dir = rec_dir
    predict_rec.main(args)