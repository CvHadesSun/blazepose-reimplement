import math

import numpy as np

'''to define the blaze pose 2d joints format and some transform tools.pose 2D :[x,y,visbility,confidence]'''


def get_encode_pts(joints, scale=[1.25, 1.25]):
    # TODO: find the mid hip points and under head points
    # version 1.
    # center p
    assert len(scale) == 2
    if joints[11,-1] <=0 or joints[12,-1]<=0:
        return np.zeros([2,4]),False


    center_x = (joints[11, 0] + joints[12, 0])/2  # mid of hip [11,12]
    center_y = (joints[11, 1] + joints[12, 1])/2  # mid of hip [11,12]
    # head p
    # mid_shoulder_x = joints[5, 0] + joints[6, 0]  # mid of shoulder [5,6]
    # mid_shoulder_y = joints[5, 1] + joints[6, 1]  # mid of shoulder [5,6]

    if joints[0,-1]<=0:
        # print("nose:",joints[0])
        return np.zeros([2,4]),False

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

    return np.array([[center_x, center_y, 1., 1.],
                     [head_x, head_y, 1., 1.]],dtype=np.float32),True


def compute_pose_flag(joints):
    # TODO: using joints
    # design the number of visible joints to pose confidence.
    assert joints.shape[-1] == 4
    vis_joints = joints[:, -2]
    num_vis = np.sum(vis_joints)

    # scale into [0,1]
    max_value = float(joints.shape[0])
    assert max_value > 0
    return num_vis / max_value


def get_visiblity_confidence(joints):
    # TODO: get the visibility an prediction confidence info from data labels.

    # coco: 0:no label; 1:label but no visibility, 2: label and can be vis.
    # design: valid-->[vis,conf] : only visible joint is confidence.
    #           0-->[0,0]
    #           1-->[0,0]
    #           2-->[1,1]
    # return : [x,y,vis,conf]

    # joints [N,3]
    joints_vis = np.zeros([joints.shape[0], ], dtype=np.float32)  # [N,1]
    joints_conf = np.zeros([joints.shape[0], ], dtype=np.float32)  # [N,1]
    label_info = joints[:, -1]
    vis_indexes = np.where(label_info == 2)
    joints_vis[vis_indexes] = 1.
    joints_conf[vis_indexes] = 1.

    new_joints = np.zeros([joints.shape[0], 4], dtype=np.float32)
    new_joints[:, :2] = joints[:, :2]
    new_joints[:, 2] = joints_vis[:]
    new_joints[:, -1] = joints_conf[:]
    return new_joints


def coco_to_balzepose(joints, scale):
    # TODO: convert the coco format joints to defined blaze pose joints format.
    # 1. get the encode points
    aux_pts,flag = get_encode_pts(joints, scale)  # [2, 4]
    if flag:
        # 2.compute the vis and conf label info.
        new_joints = get_visiblity_confidence(joints)  # [N,4]
        valid = joints[:, -1]  # [N,1]
        # 3.stack the all train joints.
        with_aux_pts = np.vstack([new_joints, aux_pts])  # [N+2,4]
        # 4. compute the pose flag
        pose_flag = compute_pose_flag(new_joints)
        # add one dim into joints.
        pose_3d = np.zeros([with_aux_pts.shape[0], 5])
        pose_3d[:, :2] = with_aux_pts[:, :2]
        pose_3d[:, 2] = with_aux_pts[:, 0]  # assign the x into z
        pose_3d[:, 3:] = with_aux_pts[:, 2:]  # [N+2,5]
        valid_stack = np.hstack([valid, np.array([1, 1])])
        # print("valid_stack",valid_stack)
        return pose_3d, pose_flag, valid_stack,True
    else:
        return 0,0,0,False
