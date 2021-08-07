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


from mediapipe.python.solutions.pose import POSE_CONNECTIONS, PoseLandmark

# 左右逆で出力
POSE_LANDMARKS = [
    'nose',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_ear',
    'left_ear',
    'mouth_right',
    'mouth_left',
    'right_shoulder',
    'left_shoulder',
    'right_elbow',
    'left_elbow',
    'right_wrist',
    'left_wrist',
    'right_pinky',
    'left_pinky',
    'right_index',
    'left_index',
    'right_thumb',
    'left_thumb',
    'right_hip',
    'left_hip',
    'right_knee',
    'left_knee',
    'right_ankle',
    'left_ankle',
    'right_heel',
    'left_heel',
    'right_foot_index',
    'left_foot_index',
]
