from tools.infer.predict_system import main
from tools.infer.utility import parse_args
import numpy as np
import os


# 识别文字
def img_rec(book_id, img_dir, drop_score, det_dir, rec_dir):
    args = parse_args()
    args.drop_score = drop_score
    args.book_id = book_id
    args.draw_img_save_dir = img_dir+'/ocr_data'
    args.image_dir = img_dir
    # 更改字典路径: 使用自己训练的识别模型时需要添加下面一条代码
    args.rec_char_dict_path = './ppocr/utils/my_dict.txt'
    args.det_model_dir = det_dir
    args.rec_model_dir = rec_dir
    main(args)


# 排序识别后的文本行内容，方便计算准确率
def sortRecResult(root, filename):
    with open(root+'/ocr_data/'+filename, 'r', encoding='utf-8') as fp:
        data = fp.readlines()
    names = []
    texts = []
    indexs = []
    for line in data:
        img_info = line.split('\t')
        names.append(img_info[0])
        text = ''
        for text_info in eval(img_info[1]):
            text += text_info['transcription']

        texts.append(text)
        ind = int(img_info[0].split('.')[0])
        indexs.append(ind)
    sort_inds = np.argsort(indexs)
    with open(root+'test_lines.txt', 'w', encoding='utf-8')as fp:
        for idx in sort_inds:
            fp.write(names[idx]+'\t'+texts[idx]+'\n')


def cal_accuracy(y_path, pred_path):
    with open(y_path, 'r', encoding='utf-8') as fp:
        y = fp.readlines()
    with open(pred_path, 'r', encoding='utf-8') as fp:
        pred_y = fp.readlines()
    total = 0
    correct = 0
    for i in range(len(y)):

        _, y_text = y[i].split('\t')
        _, pred_text = pred_y[i].split('\t')
        y_text, pred_text = y_text.strip(), pred_text.strip()
        total += len(y_text)
        j = 0
        while j < len(y_text) and j < len(pred_text):
            if y_text[j] == pred_text[j]:
                correct += 1
            j += 1
    print(total, correct)
    print('Accuracy:', correct/total)
    return correct/total


if __name__=='__main__':
    det_dir = 'inference/det_model'
    rec_dir = 'inference/rec_model'
    thresh = 0.5
    book_id = '0001'
    img_dir="C:/Users/Gatsby/datasets/det/1"
    img_rec(book_id, img_dir, thresh, det_dir, rec_dir)
    sortRecResult(img_dir, book_id+'.txt')
    acc = cal_accuracy(img_dir+'/testimg.txt', img_dir+'/test_lines.txt')
    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    with open('./result/result.txt', 'a', encoding='utf-8') as fp:
        fp.write('box_thresh='+str(thresh)+'\tAccuracy='+str(acc)+'\tdet_dir='+det_dir+'\trec_dir='+rec_dir+'\n')


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
