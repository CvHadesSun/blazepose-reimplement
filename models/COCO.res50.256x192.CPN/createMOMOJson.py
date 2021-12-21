
from pycocotools.coco import COCO
import os.path as osp
import numpy as np
import json

def get_encode_pts(joints, scale=[1.3, 1.3]):
    # TODO: find the mid hip points and under head points
    # version 1.
    # center p
    assert len(scale) == 2
    if joints[11,-1] <=0 or joints[12,-1]<=0:
        return np.zeros([2,3]),False


    center_x = (joints[11, 0] + joints[12, 0])/2  # mid of hip [11,12]
    center_y = (joints[11, 1] + joints[12, 1])/2  # mid of hip [11,12]
    # head p
    # mid_shoulder_x = joints[5, 0] + joints[6, 0]  # mid of shoulder [5,6]
    # mid_shoulder_y = joints[5, 1] + joints[6, 1]  # mid of shoulder [5,6]

    if joints[0,-1]<=0:
        return np.zeros([2,3]),False

    nose_x = joints[0, 0]
    nose_y = joints[0, 1]

    
    head_x = 0.
    head_y = 0.

    dist_x = nose_x - center_x
    dist_y = nose_y - center_y

    if dist_x == 0:
        head_x = center_x
    else:
        head_x = center_x + scale[0] * dist_x

    if dist_y == 0:
        head_y = center_y
    else:
        head_y = center_y + scale[1] * dist_y

    return np.array([[center_x, center_y, 1.],
                     [head_x, head_y, 1.]],dtype=np.float32),True

# def load_train_data(val_annot_path):
#     with open(val_annot_path,'r',encoding='utf8')as fp:
#         json_data = json.load(fp)

#     print("train dataset number:",len(json_data['annotations']))
#     print(json_data['annotations'][0])
#     for ann in json_data['annotations']:
#         # ann = coco.anns[aid]
#         joints = ann['keypoints']
#         joints = np.array(joints).reshape(-1,3)
#         add_joints, _ = get_encode_pts(joints)
#         joints = np.vstack([joints, add_joints])
#         ann['keypoints'] = joints.reshape(-1).tolist()

#     with open('momo_train.json', 'w') as f:
#             json.dump(json_data, f)

def load_train_data(val_annot_path,output_path):
    with open(val_annot_path,'r',encoding='utf8')as fp:
        json_data = json.load(fp)

    print("train dataset number:",len(json_data['annotations']))
    # print(json_data['annotations'][0])
    new_ann = []
    for ann in json_data['annotations']:
        # ann = coco.anns[aid]
        joints = ann['keypoints']
        joints = np.array(joints).reshape(-1,3)
        add_joints, valid = get_encode_pts(joints)
        joints = np.vstack([joints, add_joints])
        # ann['keypoints'] = joints.reshape(-1).tolist()
        if valid:
            new_ann.append(ann)
    json_data['annotations'] = new_ann
    with open(output_path, 'w') as f:
            json.dump(json_data, f)
            


# dataset_root = '/home/zml/workspace_pose/human-pose-estimation.pytorch-master/data/coco'
dataset_annotation = '/workdir/swh/dataset/coco/annotations'
output_path=osp.join(dataset_annotation,'person_keypoints_val2017_filtered.json')
val_annot_path = osp.join(dataset_annotation, 'person_keypoints_val2017.json')
load_train_data(val_annot_path,output_path)
