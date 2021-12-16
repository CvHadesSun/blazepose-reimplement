
import cv2
import numpy as np
import matplotlib.pyplot as plt

def vis_keypoints( img, kps, force=False, kp_thresh=0.2, alpha=1):
    # print("kps",kps)
    
    kps = kps.T
    # print(kps)
    # kps[0,:] *=256/64
    # kps[1,:] *=256/64
    
    kps_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    kps_symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    kps_lines = [(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
                    (13, 15), (5, 6), (11, 12)]
    num_kps = 17
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
                            kps[:2, 5] +
                            kps[:2, 6]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, 5],
        kps[2, 6])
    mid_hip = (
                        kps[:2, 11] +
                        kps[:2, 12]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, 11],
        kps[2, 12])
    nose_idx = 0
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(kps[:2, nose_idx].astype(np.int32)),
            color=colors[len(kps_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(mid_hip.astype(np.int32)),
            color=colors[len(kps_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh or force:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    # mean_score = np.mean(kps[2,:]) #/np.std(kps[2,:])
    # cv2.putText(kp_mask,  "%.2f"%(mean_score), (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
