# recommended paddle.__version__ == 2.0.0
python  tools/train.py -c output/v3_chinese_cht_mobile/config.yml
python  tools/train.py -c output/det_r50_vd/config.yml
python tools/export_model.py -c output/det_r50_vd/config.yml
python tools/infer/predict_system.py -c output/det_r50_vd/config.yml
