
# coding=utf-8

import os
import os.path as osp
import numpy as np
import cv2
import math

import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_dir, 'MSCOCO', 'PythonAPI'))

from pycocotools.coco import COCO
from utils.joints import coco_to_balzepose

class MOMOJoints(object):
    def __init__(self):
        self.kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',
        'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
        'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
        self.max_num_joints = 17
        self.color = np.random.randint(0, 256, (self.max_num_joints, 3))

        self.mpi = []
        self.test_mpi = []
        for mpi, stage in zip([self.mpi, self.test_mpi], ['train', 'val']):
            if stage == 'train':
                # self._train_gt_path=os.path.join(cur_dir, 'MSCOCO', 'annotations', 'person_keypoints_trainvalminusminival2014.json')
                self._train_gt_path=os.path.join('/workdir/swh/dataset/coco/annotations','person_keypoints_train2017.json')
                coco = COCO(self._train_gt_path)
            else:
                # self._val_gt_path=os.path.join(cur_dir, 'MSCOCO', 'annotations', 'person_keypoints_minival2014.json')
                self._val_gt_path=os.path.join('/workdir/swh/dataset/coco/annotations','person_keypoints_val2017.json')
                coco = COCO(self._val_gt_path)
            if stage == 'train':
                for aid in coco.anns.keys():
                    ann = coco.anns[aid]
                    if ann['image_id'] not in coco.imgs or ann['image_id'] == '366379':
                        continue
                    imgname = coco.imgs[ann['image_id']]['file_name']
                    prefix = 'val' if 'val' in imgname else 'train'
                    rect = np.array([0, 0, 1, 1], np.int32)
                    if ann['iscrowd']:
                        continue
                    joints = ann['keypoints']

                    # to filter invalid instance.
                    np_joints = np.array(joints).reshape(-1, 3)
                    pose_3d, score, valid, flag = coco_to_balzepose(np_joints, scale=[1.3,1.3])
                    if not flag: # invalid instance.
                        continue
                    # bbox = ann['bbox']
                    bbox=self._compute_bbox(pose_3d)

                    pose_3d = pose_3d.reshape(-1).tolist()
                    valid = valid.reshape(-1).tolist()

                    # >>to square the bbox
                    # x1,y1,w,h=bbox
                    # center_x=x1+w/2
                    # center_y=y1+h/2
                    # w= w if w>h else h
                    # h=w
                    # x1=center_x-w/2
                    # y1=center_y-h/2
                    # bbox=[x1,y1,w,h]
                    # <<<
                    if np.sum(joints[2::3]) == 0 or ann['num_keypoints'] == 0 :
                        continue
                    # imgname = prefix + '2014/' + 'COCO_' + prefix + '2014' + '_' + str(ann['image_id']).zfill(12) + '.jpg'
                    imgname=prefix + '2017/' +coco.imgs[ann['image_id']]['file_name']
                    humanData = dict(aid = aid,joints=pose_3d, imgpath=imgname, headRect=rect, bbox=bbox, imgid = ann['image_id'], segmentation = ann['segmentation'],valid=valid)
                    mpi.append(humanData)
            elif stage == 'val':
                # files = [(img_id,coco.imgs[img_id]) for img_id in coco.imgs]
                # for img_id,img_info in files:
                #     imgname = stage + '2017/' + img_info['file_name']
                #     # imgname=stage + '2017/' +coco.imgs[ann['image_id']]['file_name']
                #     humanData = dict(imgid = img_id,imgpath = imgname)
                #     mpi.append(humanData)
                
                for aid in coco.anns.keys():
                    ann = coco.anns[aid]
                    # print(ann['image_id'] not in coco.imgs) 
                    # if ann['image_id'] not in coco.imgs or ann['image_id'] == '366379':
                    #     continue
                    imgname = coco.imgs[ann['image_id']]['file_name']
                    prefix = 'val' 
                    rect = np.array([0, 0, 1, 1], np.int32)
                    if ann['iscrowd']:
                        continue
                    joints = ann['keypoints']

                    # to filter invalid instance.
                    np_joints = np.array(joints).reshape(-1, 3)
                    pose_3d, score, valid, flag = coco_to_balzepose(np_joints, scale=[1.3,1.3])
                    if not flag:
                        continue
                    bbox = ann['bbox']
                    pose_3d = pose_3d.reshape(-1).tolist()
                    # if np.sum(joints[2::3]) == 0 or ann['num_keypoints'] == 0 :
                    #     continue
                    imgname=prefix + '2017/' +coco.imgs[ann['image_id']]['file_name']   
                    humanData = dict(imgid = ann['image_id'],imgpath = imgname,bbox=bbox,joints=pose_3d)
                    mpi.append(humanData)                  
            else:
                print('COCO data error, please check')
                embed()

    def load_data(self, min_kps=1):
        mpi = [i for i in self.mpi if np.sum(np.array(i['joints'], copy=False)[2::3] > 0) >= min_kps]
        return mpi, self.test_mpi

    def _compute_bbox(self, joints):
        center = joints[-2]
        top_head = joints[-1]
        radius = math.sqrt((center[0] - top_head[0]) ** 2 + (center[1] - top_head[1]) ** 2)
        w = radius * 2
        h = radius * 2
        xmin = center[0] - w / 2
        ymin = center[1] - h / 2
        return [xmin, ymin, w, h]

if __name__ == '__main__':
    coco_joints = MOMOJoints()
    train, test = coco_joints.load_data(min_kps=1)
    from IPython import embed; embed()

