import os
import xml.etree.ElementTree as ET
import matplotlib.image as im


def preprocess():
    file_dir = "./dataset/Fires/train/annotations/"
    for filename in os.listdir(file_dir):
        if not filename.endswith("txt"):
            continue
        full_filename = file_dir + filename
        with open(full_filename) as f:
            txt = f.readlines()[0]
            img, label, x1, y1, x2, y2 = tuple(txt.split(" "))
            print(img, label, x1, y1, x2, y2)
            root = ET.Element("annotation")
            tree = ET.ElementTree(root)

            fn = ET.Element('filename')
            fn.text = img

            size = ET.Element("size")
            width = ET.Element("width")
            width.text = "486"
            height = ET.Element('height')
            height.text = "500"
            depth = ET.Element("depth")
            depth.text = "3"
            size.extend([width, height, depth])

            object = ET.Element("object")
            name = ET.Element("name")
            name.text = label
            bndbox = ET.Element("bndbox")
            xmin = ET.Element("xmin")
            xmin.text = x1
            ymin = ET.Element("ymin")
            ymin.text = y1
            xmax = ET.Element("xmax")
            xmax.text = x2
            ymax = ET.Element("ymax")
            ymax.text = y2

            root.extend([fn, size, object, bndbox])
            target_file = "./dataset/annotations/" + filename.replace("txt", "xml")
            print(target_file)
            tree.write(target_file, encoding='utf-8', xml_declaration=True)
        # break


def preprocess2():
    file_dir = "./dataset/Fires/train/annotations/"
    image_dir = "./dataset/Fires/train/images/"
    for filename in os.listdir(file_dir):
        if not filename.endswith("txt") or "(1)" in filename:
            continue
        full_filename = file_dir + filename
        with open(full_filename) as f:
            content = f.readlines()
            txt = content[0]
            img, label, x1, y1, x2, y2 = tuple(txt.split(" "))
            print(img, label, x1, y1, x2, y2)
            root = ET.Element("annotation")

            fn = ET.SubElement(root, 'filename')
            fn.text = img

            image_file = image_dir+filename.replace("txt", "jpg")
            if not os.path.exists(image_file):
                continue
            img = im.imread(image_file)
            size = ET.SubElement(root, "size")
            width = ET.SubElement(size, "width")
            # width.text = "486"
            width.text = str(img.shape[1])
            height = ET.SubElement(size, 'height')
            # height.text = "500"
            height.text = str(img.shape[0])
            depth = ET.SubElement(size, "depth")
            depth.text = str(img.shape[2])

            for line in content:
                img, label, x1, y1, x2, y2 = tuple(line.split(" "))
                object = ET.SubElement(root, "object")
                name = ET.SubElement(object, "name")
                name.text = label
                difficult = ET.SubElement(object, "difficult")
                difficult.text = "0"
                bndbox = ET.SubElement(object, "bndbox")
                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = x1
                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = y1
                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = x2
                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = y2

            target_file = "./dataset/annotations/" + filename.replace("txt", "xml")
            print(target_file)

            tree = ET.ElementTree(root)
            ET.dump(root)
            tree.write(target_file, encoding='utf-8', xml_declaration=True)
        # break


def read_annotations():
    dir = "dataset/trainval.txt"
    with open(dir) as f:
        for line in f:
            line = line.strip().split(" ")
            anno = line[1]
            tree = ET.parse("dataset/" + anno)
            root = tree.getroot()
            ET.dump(root)
            break


def run():
    preprocess2()
    # read_annotations()


if __name__ == '__main__':
    run()

