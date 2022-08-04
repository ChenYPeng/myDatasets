import collections
import datetime
import glob
import json
import uuid
import os
import os.path as osp
import sys
import numpy as np
import imgviz
import labelme
import shutil

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)


def main():
    noviz = False
    root = os.getcwd()
    dataset = 'hanfengseg'
    sets = ['train2017', 'val2017', 'test2017']
    output_dir = osp.join(root, dataset)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print('Creating dataset:', output_dir)
    else:
        shutil.rmtree(output_dir)
        print('Output directory already exists:', output_dir)

    if not osp.exists(osp.join(output_dir, 'annotations')):
        os.makedirs(osp.join(output_dir, 'annotations'))
    if not osp.exists(osp.join(output_dir, 'JPEGImages')):
        os.makedirs(osp.join(output_dir, 'JPEGImages'))
    if not osp.exists(osp.join(output_dir, 'Visualization')):
        os.makedirs(osp.join(output_dir, 'Visualization'))

    for set in sets:
        input_dir = osp.join(root, set)
        filename = 'instances_%s' % set  # instances_train2017
        now = datetime.datetime.now()

        data = dict(
            info=dict(
                description='HanFeng',
                url=None,
                version="5.0.1",
                year="2022",
                contributor=None,
                date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
            ),
            licenses=[dict(url=None, id=0, name=None, )],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            type="instances",
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

        class_name_to_id = {}
        for i, line in enumerate(open(osp.join(root, 'labels.txt')).readlines()):
            class_id = i - 1  # starts with -1
            class_name = line.strip()
            if class_id == -1:
                assert class_name == "__ignore__"
                continue
            class_name_to_id[class_name] = class_id
            data["categories"].append(
                dict(supercategory=None, id=class_id, name=class_name, )
            )
        out_ann_file = osp.join(output_dir, 'annotations', filename + '.json')  # ./annotations\instances_test2017.json
        label_files = glob.glob(osp.join(input_dir, '*.json'))  # [./train2017\\0.json, ...]

        for image_id, filename in enumerate(label_files):
            print("Generating dataset from:", filename)

            label_file = labelme.LabelFile(filename=filename)

            base = osp.splitext(osp.basename(filename))[0]
            img_base_name = base + ".jpg"
            out_img_file = osp.join(output_dir, "JPEGImages", set, base + ".jpg")
            img = labelme.utils.img_data_to_arr(label_file.imageData)
            imgviz.io.imsave(out_img_file, img)

            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    file_name=img_base_name,
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                )
            )

            masks = {}  # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            for shape in label_file.shapes:
                points = shape["points"]
                label = shape["label"]
                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")
                mask = labelme.utils.shape_to_mask(
                    img.shape[:2], points, shape_type
                )

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                if shape_type == "circle":
                    (x1, y1), (x2, y2) = points
                    r = np.linalg.norm([x2 - x1, y2 - y1])
                    # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                    # x: tolerance of the gap between the arc and the line segment
                    n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                    i = np.arange(n_points_circle)
                    x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                    y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                    points = np.stack((x, y), axis=1).flatten().tolist()
                else:
                    points = np.asarray(points).flatten().tolist()

                segmentations[instance].append(points)
            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )

            if not noviz:
                viz = img
                if masks:
                    labels, captions, masks = zip(
                        *[
                            (class_name_to_id[cnm], cnm, msk)
                            for (cnm, gid), msk in masks.items()
                            if cnm in class_name_to_id
                        ]
                    )
                    viz = imgviz.instances2rgb(
                        image=img,
                        labels=labels,
                        masks=masks,
                        captions=captions,
                        font_size=15,
                        line_width=2,
                    )
                out_viz_file = osp.join(
                    output_dir, "Visualization", set, base + ".jpg"
                )
                imgviz.io.imsave(out_viz_file, viz)

        with open(out_ann_file, "w") as f:
            json.dump(data, f)


if __name__ == '__main__':
    main()
