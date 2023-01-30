

import os
import cv2
import numpy as np
import time
import sys

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
import json


from tools.infer.predict_det import TextDetector

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'


logger = get_logger()


if __name__ == '__main__':

    args = utility.parse_args()
    # args.image_dir = "C:/Users/Gatsby/datasets/det/1"
    args.image_dir = "C:/Users/Gatsby/datasets/det/1"

    # args.det_model_dir = 'output/ch_PP-OCR_V3_det'
    args.det_model_dir = 'inference/det_model'

    args.draw_img_save_dir = './output/savedResult'
    # args.pretrained_model = "pretrain_models/ch_PP-OCRv3_det_distill_train/best_accuracy"


    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)
    total_time = 0
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)

    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(2):
            res = text_detector(img)

    save_results = []
    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        for index, img in enumerate(imgs):
            st = time.time()
            dt_boxes, _ = text_detector(img)
            elapse = time.time() - st
            total_time += elapse
            if len(imgs) > 1:
                save_pred = os.path.basename(image_file) + '_' + str(
                    index) + "\t" + str(
                    json.dumps([x.tolist() for x in dt_boxes])) + "\n"
            else:
                save_pred = os.path.basename(image_file) + "\t" + str(
                    json.dumps([x.tolist() for x in dt_boxes])) + "\n"
            save_results.append(save_pred)
            logger.info(save_pred)
            if len(imgs) > 1:
                logger.info("{}_{} The predict time of {}: {}".format(
                    idx, index, image_file, elapse))
            else:
                logger.info("{} The predict time of {}: {}".format(
                    idx, image_file, elapse))

            src_im = utility.draw_text_det_res(dt_boxes, img)

            if flag_gif:
                save_file = image_file[:-3] + "png"
            elif flag_pdf:
                save_file = image_file.replace('.pdf',
                                               '_' + str(index) + '.png')
            else:
                save_file = image_file
            img_path = os.path.join(
                draw_img_save_dir,
                "det_res_{}".format(os.path.basename(save_file)))
            cv2.imwrite(img_path, src_im)
            logger.info("The visualized image saved in {}".format(img_path))

    with open(os.path.join(draw_img_save_dir, "det_results.txt"), 'w') as f:
        f.writelines(save_results)
        f.close()
    if args.benchmark:
        text_detector.autolog.report()
