# recommended paddle.__version__ == 2.0.0
python  tools/train.py -c output/v3_chinese_cht_mobile/config.yml
python  tools/train.py -c output/det_r50_vd/config.yml
python tools/export_model.py -c output/ch_PP-OCR_V3_det/config.yml


python tools/infer_rec.py -c output/v3_chinese_cht_mobile/config.yml -o Global.infer_img="./doc/imgs_en/" Global.pretrained_model="output/v3_chinese_cht_mobile/best_accuracy"

python  tools/train.py -c output/ch_PP-OCR_V3_det/config.yml