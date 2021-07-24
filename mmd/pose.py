# -*- coding: utf-8 -*-
import os
import glob
import copy
import json

import numpy as np
from tqdm import tqdm
import cv2
import pathlib
import mediapipe as mp
import datetime
import shutil
import itertools
import mediapipe as mp

from mmd.utils.MLogger import MLogger

logger = MLogger(__name__, level=1)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def execute(args):
    try:
        logger.info('人物(身体)推定開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        # 入力ファイル
        process_input_path = os.path.join(args.img_dir, "input_30fps.mp4")
        video = cv2.VideoCapture(process_input_path)

        # 既存は削除
        if os.path.exists(os.path.join(args.img_dir, "joints")):
            shutil.rmtree(os.path.join(args.img_dir, "joints"))
        if os.path.exists(os.path.join(args.img_dir, "smooth")):
            shutil.rmtree(os.path.join(args.img_dir, "smooth"))

        # 出力ディレクトリ
        os.makedirs(os.path.join(args.img_dir, "joints"), exist_ok=True)

        # 幅
        W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 高さ
        H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 総フレーム数
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        process_pose_path = os.path.join(args.img_dir, "pose.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(process_pose_path, fourcc, 30, (W, H))

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            for n in tqdm(range(int(count))):
                # 動画から1枚キャプチャして読み込む
                flag, img = video.read()  # Capture frame-by-frame

                # 動画が終わっていたら終了
                if flag == False:
                    break

                params_json_path = os.path.join(args.img_dir, "joints", f'{n:012}_joints.json')
                image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
                
                # 一旦書き込み不可
                image.flags.writeable = False
                pose_results = pose.process(image)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image.flags.writeable = True

                # json出力
                joint_dict = {}
                joint_dict["image"] = {"width": W, "height": H}
                joint_dict["joints"] = {}

                if pose_results.pose_landmarks and pose_results.pose_landmarks.landmark and pose_results.pose_world_landmarks and pose_results.pose_world_landmarks.landmark:
                    for landmark, world_landmark, output_name in zip(pose_results.pose_landmarks.landmark, pose_results.pose_world_landmarks.landmark, POSE_LANDMARKS):
                        joint_dict["joints"][output_name] = {'x': float(landmark.x), 'y': float(landmark.y), 'z': float(landmark.z), 
                                                            'wx': float(world_landmark.x), 'wy': float(world_landmark.y), 'wz': float(world_landmark.z)}
                    
                    joint_dict["joints"]['spine2'] = mean_joint(joint_dict, 'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder')
                    joint_dict["joints"]['spine3'] = mean_joint(joint_dict, 'left_shoulder', 'right_shoulder')
                    joint_dict["joints"]['spine1'] = mean_joint(joint_dict, 'left_hip', 'right_hip', 'spine2')
                    joint_dict["joints"]['pelvis'] = mean_joint(joint_dict, 'left_hip', 'right_hip', 'spine2')
                    joint_dict["joints"]['pelvis2'] = mean_joint(joint_dict, 'left_hip', 'right_hip')
                    joint_dict["joints"]['neck'] = mean_joint(joint_dict, 'mouth_left', 'mouth_right', 'nose')
                    joint_dict["joints"]['head'] = mean_joint(joint_dict, 'left_eye', 'right_eye')
                    joint_dict["joints"]['left_collar'] = mean_joint(joint_dict, 'left_shoulder', 'right_shoulder')
                    joint_dict["joints"]['right_collar'] = mean_joint(joint_dict, 'left_shoulder', 'right_shoulder')
                    joint_dict["joints"]['left_f_base'] = mean_joint(joint_dict, 'left_index', 'left_pinky')
                    joint_dict["joints"]['right_f_base'] = mean_joint(joint_dict, 'right_index', 'right_pinky')

                    # 認識映像の出力
                    mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, POSE_CONNECTIONS)

                out.write(image)

                with open(params_json_path, 'w') as f:
                    json.dump(joint_dict, f, indent=4)

        out.release()
        video.release()

        cv2.destroyAllWindows()
            
        return True, args.img_dir
    except Exception as e:
        logger.critical("人物(身体)推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False, None

def mean_joint(joint_dict: dict, *joint_names):
    xs = []
    ys = []
    zs = []
    wxs = []
    wys = []
    wzs = []

    for joint_name in joint_names:
        xs.append(joint_dict["joints"][joint_name]['x'])
        ys.append(joint_dict["joints"][joint_name]['y'])
        zs.append(joint_dict["joints"][joint_name]['z'])
        wxs.append(joint_dict["joints"][joint_name]['wx'])
        wys.append(joint_dict["joints"][joint_name]['wy'])
        wzs.append(joint_dict["joints"][joint_name]['wz'])

    return {'x': np.mean(xs, axis=0), 'y': np.mean(ys, axis=0), 'z': np.mean(zs, axis=0), 'wx': np.mean(wxs, axis=0), 'wy': np.mean(wys, axis=0), 'wz': np.mean(wzs, axis=0)}


from mediapipe.python.solutions.pose import POSE_CONNECTIONS, PoseLandmark

UPPER_POSE_CONNECTIONS = frozenset([
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER),
    (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER),
    (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER),
    (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER),
    (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR),
    (PoseLandmark.MOUTH_RIGHT, PoseLandmark.MOUTH_LEFT),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB),
    (PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB),
    (PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_HIP),
])

POSE_LANDMARKS = [
    'nose',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_ear',
    'right_ear',
    'mouth_left',
    'mouth_right',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot_index',
    'right_foot_index',
]

HAND_LANDMARKS = [
    "wrist", 
    'thumb1', 
    'thumb2', 
    'thumb3', 
    'thumb4', 
    'index1', 
    'index2', 
    'index3', 
    'index4', 
    'middle1', 
    'middle2', 
    'middle3', 
    'middle4', 
    'ring1', 
    'ring2', 
    'ring3', 
    'ring4', 
    'pinky1', 
    'pinky2',
    'pinky3',  
    'pinky4', 
]


