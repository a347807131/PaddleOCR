# import json
# import pymysql
# import ocr_parser
# from tools.infer.predict_system import main
# from tools.infer.utility import parse_args
#
#
# args = ocr_parser.parse_args()
#
#
# def load_json(path):
#     with open(path, 'r', encoding='utf-8') as fp:
#         return json.load(fp)
#
#
# def save_json(path, data):
#     with open(path, 'w', encoding='utf-8') as fp:
#         json.dump(data, fp)
#
#
# def get_book_path(book_id, type_id):
#     conn = pymysql.connect(host=args.sql_host, user=args.sql_user, password=args.sql_password, port=args.sql_port,
#                            database=args.sql_database, charset='utf8')
#     cur = conn.cursor()  # 生成游标对象
#     attribute_id = type_id * 1000 + 4  # int
#     sql = "select attribute_value from gallery_book where book_id=%s and attribute_id=%s"
#     values = [book_id, attribute_id]
#     cur.execute(sql, values)
#     path = cur.fetchall()  # 获取数据
#     cur.close()
#     conn.close()
#     return path
#
#
# def img_rec(book_id, img_dir, det_dir, rec_dir):
#     '''
#         book_id: 图书编号，用于创建相应的识别文件名
#         img_dir: 图片路径， 可以是文件夹或者单张图片
#         det_dir: 检测模型路径， 例：'./inference/my_ppocrv3_det/'
#     '''
#     args = parse_args()
#     # 测试最优后再更改
#     # args.drop_score = 0.5
#     args.book_id = book_id
#     args.draw_img_save_dir = img_dir+'/ocr_data'
#     args.image_dir = img_dir
#     # 更改字典路径: 使用自己训练的识别模型时需要添加下面一条代码
#     # args.rec_char_dict_path = './ppocr/utils/my_oppocr_key.txt'
#     args.det_model_dir = det_dir
#     args.rec_model_dir = rec_dir
#     main(args)
#
#
# # 按x轴排序文本
# def sort_text(text_info):
#     texts = []
#     locs = []
#     for info in text_info:
#         text = info['transcription']
#         x1 = info['points'][0][0]
#         texts.append(text)
#         locs.append(x1)
#     # 按索引排序
#     sorted_idx = sorted(range(len(locs)), key=lambda k: locs[k], reverse=True)
#     target_str = ''
#     i = 0
#     for i in range(len(sorted_idx)):
#         if i < len(locs) - 1:
#             target_str += texts[sorted_idx[i]] + '\n'
#         else:
#             # 最后一行不加换行
#             target_str += texts[sorted_idx[i]]
#     return target_str
#
#
# # 存储单张图片内容
# def get_img_content(dir, book_id):
#     '''
#         dir: ocr_dir,例：get_img_content('imgs/ocr_data/', '0001')
#         book_id: 如名
#     '''
#     with open(dir+book_id+'.txt', 'r', encoding='utf-8')as fp:
#         lines = fp.readlines()
#     for line in lines:
#         img_name, text_info = line.split('\t')
#         filename = img_name.split('.')[0]
#         text_info = eval(text_info)
#         sort_text(text_info)
#         print(img_name, 'is processing...')
#         target_txt = sort_text(text_info)
#         with open(dir+filename+'.txt', 'w', encoding='utf-8')as fp:
#             fp.write(target_txt)
#
#
