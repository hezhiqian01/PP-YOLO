import os
import time
import cv2
import numpy as np
import paddle.fluid as fluid
from PIL import Image, ImageFont, ImageDraw
import os
from infer import predict_image


input_size = [608, 608]
# image mean
img_mean = [0.485, 0.456, 0.406]
# image std.
img_std = [0.229, 0.224, 0.225]
# data label/
label_file = 'dataset/label_list.txt'
# threshold value
score_threshold = 0.5
# infer model path
infer_model_path = 'output/'
# Whether use GPU to train.
use_gpu = False

if not os.path.exists(infer_model_path):
    raise ValueError("The model path [%s] does not exist." % infer_model_path)

place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

[infer_program,
 feeded_var_names,
 target_var] = fluid.io.load_inference_model(dirname=infer_model_path,
                                             executor=exe,
                                             model_filename='__model__',
                                             params_filename='__params__')

with open(label_file, 'r', encoding='utf-8') as f:
    names = f.readlines()


# 图像预处理
def load_image(image_path):
    with open(image_path, 'rb') as f:
        im_read = f.read()
    data = np.frombuffer(im_read, dtype='uint8')
    img = cv2.imdecode(data, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 获取原图片的大小
    im_size = np.array([[img.shape[0], img.shape[1]]], dtype=np.int32)
    img = cv2.resize(img, (input_size[0], input_size[1]))
    img = (img - img_mean) * img_std
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, im_size


# 预测图像
def infer(image_path):
    img, im_size = load_image(image_path)
    start = time.time()
    nmsed_out_v = exe.run(infer_program,
                          feed={feeded_var_names[0]: img, feeded_var_names[1]: im_size},
                          fetch_list=target_var,
                          return_numpy=False)
    nmsed_out_v = np.array(nmsed_out_v[0])
    end = time.time()
    results = []
    for dt in nmsed_out_v:
        if dt[1] < score_threshold:
            continue
        results.append(dt)
        print("预测时间：%d, 预测结果：%s" % (round((end - start) * 1000), results))
    return results


# 对图像进行画框
def draw_image(image_path, results):
    img = cv2.imread(image_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for result in results:
        xmin, ymin, xmax, ymax = result[2:]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 0, 255), width=4)
        # 字体的格式
        font_style = ImageFont.truetype("font/simfang.ttf", 18, encoding="utf-8")
        # 绘制文本
        draw.text((xmin, ymin), '%s, %0.2f' % (names[int(result[0])], result[1]), (0, 255, 0), font=font_style)
    # 显示图像
    cv2.imshow('result image', cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def get_annotation(img_path):
    img_name = img_path.split('/')[-1].split('.')[0]
    # ann_path = img_path.replace("images", "annotations").replace('jpg', "txt")
    ann_path = "dataset/annotations/{}.txt".format(img_name)
    results = []
    with open(ann_path) as f:
        for line in f:
            content = line.split(' ')
            cls, trust, xmin, ymin, xmax, ymax = 0, 1, content[2], content[3], content[4], content[5]
            result = (cls, trust, int(xmin), int(ymin), int(xmax), int(ymax.rstrip()))
            results.append(result)
    return results


def eval():
    count = 0
    total = 0
    for img_path in get_images():
        result = infer(img_path)
        if result:
            print(result)
            print(img_path)
            count += 1
            # break
        total += 1
        print("count/total: {}/{}".format(count, total))


def get_images():
    base = 'dataset/Fires/test/images/'
    for filename in os.listdir(base):
        if "(1)" in filename:
            continue
        filename = base + filename
        yield filename

def predict():
    detector = Detector(args.model_dir, use_gpu=args.use_gpu, run_mode=args.run_mode)



def predict_test_data():
    import datetime
    result_file = "./result_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d%H"))
    if os.path.exists(result_file):
        os.remove(result_file)
    with open(result_file, "a") as f:
        for img_path in get_images():
            result = predict_image(img_path)
            for res in result["boxes"]:
                img_name = img_path.split("/")[-1].split(".")[0]
                confidence = res[1]
                xmin = res[2]
                ymin = res[3]
                xmax = res[4]
                ymax = res[5]
                line = "{} {} {} {} {} {}\n".format(img_name, confidence, xmin, ymin, xmax, ymax)
                # print(line)
                f.write(line)


def clean():
    result_file = "./result.txt"
    new_lines = []
    with open(result_file, "r") as f:
        for line in f:
            line = line.split(" ")
            img_name = line[0]
            confid = line[1]
            xmin, ymin, xmax, ymax = line[2], line[3], line[4], line[5]
            new_lines.append("{} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(img_name, float(confid),
                                                          round(float(xmin)),
                                                          round(float(ymin)),
                                                          round(float(xmax)),
                                                          round(float(ymax))
                                                        # xmin.split(".")[0],
                                                        # ymin.split(".")[0],
                                                        # xmax.split(".")[0],
                                                        # xmax.split(".")[0]
                                                     ))
    # print(len(new_lines))
    # print(new_lines[0])
    os.remove(result_file)
    with open(result_file, "a") as f:
        for line in new_lines:
            f.write(line)


if __name__ == '__main__':
    predict_test_data()
    # clean()

    # 单张图像预测
    # img_path = 'dataset/images/000416901017954.jpg'
    # result = infer(img_path)
    # # result = get_annotation(img_path)
    # draw_image(img_path, result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
