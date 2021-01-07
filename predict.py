import os
import infer

threshold = 0.01


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


def get_images():
    base = 'dataset/Fires/test/images/'
    for filename in os.listdir(base):
        if "(1)" in filename:
            continue
        filename = base + filename
        yield filename


def predict(img_path):
    model_dir = 'output'
    detector = infer.Detector(model_dir, use_gpu=False, run_mode="fluid")
    result = detector.predict(img_path, threshold=threshold)
    return result


def predict_test_data():
    import datetime
    result_file = "./result_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d%H"))
    if os.path.exists(result_file):
        os.remove(result_file)
    with open(result_file, "a") as f:
        for img_path in get_images():
            result = predict(img_path, )
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
    result_file = "./result_2021010617.txt"
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
    # predict_test_data()
    clean()

    # 单张图像预测
    # img_path = 'dataset/images/000416901017954.jpg'
    # result = infer(img_path)
    # # result = get_annotation(img_path)
    # draw_image(img_path, result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
