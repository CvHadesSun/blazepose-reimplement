
'''
author:cvhadessun
date:2021-12-22 /3:47
filter:
    * to filter the image which has three or more person 
    * to filter the instance which has not both left-shoulder,right-shoulder and left-hip ,right-hip
'''

from posixpath import join, relpath
from pycocotools.coco import COCO
import json
import os.path as osp
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


# COCO api 

# cats = coco.loadCats(coco.getCatIds())
# catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
# imgIds = coco.getImgIds(catIds=catIds )
# imgIds = coco.getImgIds(imgIds = [324158])


def load_coco_json(file_path,output_path,max_instance_num=2,filter_joint=False):
    # 
    coco=COCO(file_path)

    catIds = coco.getCatIds(catNms=['person']) # only human

    imgIds = coco.getImgIds(catIds=catIds ) # all img ids

    with open(file_path,'r',encoding='utf8')as fp:
        json_data = json.load(fp)

    
    num_instance=0
    filtered_imgs=0
    keeped_imgs=0

    new_imgs=[]
    new_anns=[]
    for img_id in tqdm(imgIds):
        # per img
        annIds = coco.getAnnIds(imgIds=[img_id], catIds=catIds, iscrowd=None)

        if len(annIds)>max_instance_num:
            filtered_imgs +=1
            continue

        tmp_ann=None
        for ann_id in annIds:
            ann = coco.anns[ann_id]
            # 9,3,12,6
            joints = ann['keypoints']
            joints=np.array(joints).reshape(-1,3)
            # filter invalid annotation
            if filter_joint:
                if joints[9,-1]==0 or joints[3,-1]==0 or joints[12,-1]==0 or joints[6,-1]==0 :
                    # print(joints)
                    continue
            new_anns.append(ann)
            tmp_ann=ann
            num_instance +=1
        if not isinstance(tmp_ann,type(None)):
            new_imgs.append(coco.imgs[tmp_ann['image_id']])
            keeped_imgs +=1

    # overwrite images and annotations 
    json_data['images'] = new_imgs
    json_data['annotations'] = new_anns


    # pprint

    table = PrettyTable(['state','num-image','num-instance'])
    table.add_row(['BEFORE',len(imgIds),len(coco.anns.keys())])
    table.add_row(['AFTER',keeped_imgs,num_instance])
    print(table)


    with open(output_path, 'w') as f:
        
        json.dump(json_data, f)
        

    
def vis_coco_json(file_path,img_folder,output_path):

    coco=COCO(file_path)

    catIds = coco.getCatIds(catNms=['person']) # only human

    imgIds = coco.getImgIds(catIds=catIds ) # all img ids

    for img_id in tqdm(imgIds):

        imgname=coco.imgs[img_id]['file_name']

        img=io.imread(osp.join(img_folder,imgname))

        annIds = coco.getAnnIds(imgIds=[img_id], catIds=catIds, iscrowd=None)

        anns = coco.loadAnns(annIds)

        plt.imshow(img)

        plt.axis('off')

        coco.showAnns(anns)

        plt.savefig(osp.join(output_path,imgname.replace('/','_')))

        plt.clf()

    


        
# filtered

dataset_annotation = '/workdir/swh/dataset/coco/annotations'
json_file=osp.join(dataset_annotation, 'person_keypoints_val2017.json')
load_coco_json(json_file,'./test_output.json',filter_joint=True)


# vis

# data_folder='/workdir/swh/dataset/coco/images/val2017'

# vis_coco_json('./test_output.json',data_folder,'debug')