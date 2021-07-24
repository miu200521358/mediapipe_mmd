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

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric

logger = MLogger(__name__, level=1)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def execute(args):
    try:
        logger.info('人物指推定開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        process_hand_path = os.path.join(args.img_dir, "detector.mp4")
        process_img_pathes = sorted(glob.glob(os.path.join(args.img_dir, "frames", "**", "frame_*.png")), key=sort_by_numeric)

        hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, upper_body_only=True)

        img = cv2.imread(process_img_pathes[0])
        W = img.shape[1]
        H = img.shape[0]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(process_hand_path, fourcc, 30, (W, H))

        for iidx, process_img_path in enumerate(tqdm(process_img_pathes)):
            fname = f'{iidx:012}'
            params_json_path = os.path.join(os.path.dirname(process_img_path), f'{fname}_joints.json')

            img = cv2.imread(process_img_path)

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            hand_results = hands.process(image)
            pose_results = pose.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # json出力
            joint_dict = {}
            joint_dict["image"] = {"width": W, "height": H}
            joint_dict["joints"] = {}

            if pose_results.pose_landmarks is not None:
                for landmark, output_name in zip(pose_results.pose_landmarks.landmark, POSE_LANDMARKS):
                    joint_dict["joints"][output_name] = {'x': float(landmark.x), 'y': -float(landmark.y), 'z': float(landmark.z)}

                # 認識映像の出力
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, UPPER_POSE_CONNECTIONS)

            if hand_results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks,
                                                    hand_results.multi_handedness):
                    # 手の方向
                    direction = handedness.classification[0].label.lower()
                    # 手は左右反対に判定される
                    direction = 'right' if direction == 'left' else 'left'

                    for landmark, output_name in zip(hand_landmarks.landmark, HAND_LANDMARKS):
                        logger.debug("direction: {0}, landmark: x={1}, y={2}, z={3}, output={4}", direction, landmark.x, landmark.y, landmark.z, output_name)

                        if output_name:
                            joint_dict["joints"][f'{direction}_{output_name}'] = {'x': float(landmark.x), 'y': -float(landmark.y), 'z': float(landmark.z)}

                    # 認識映像の出力
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for direction in ['right', 'left']:
                if f'body_{direction}_wrist' in joint_dict['joints'] and f'{direction}_wrist' in joint_dict['joints']:
                    # 身体判定でも手判定でも手首が見つかった場合
                    body_wrist_val = np.array([joint_dict['joints'][f'body_{direction}_wrist']['x'], joint_dict['joints'][f'body_{direction}_wrist']['y'], joint_dict['joints'][f'body_{direction}_wrist']['z']])
                    hand_wrist_val = np.array([joint_dict['joints'][f'{direction}_wrist']['x'], joint_dict['joints'][f'{direction}_wrist']['y'], joint_dict['joints'][f'{direction}_wrist']['z']])
                    diff_val = body_wrist_val - hand_wrist_val

                    for output_name in HAND_LANDMARKS:
                        # 身体判定の手首位置と手判定の手首位置を合わせる
                        output = joint_dict['joints'][f'{direction}_{output_name}']
                        joint_val = np.array([output['x'], output['y'], output['z']])
                        joint_val += diff_val
                        joint_dict['joints'][f'{direction}_{output_name}'] = {'x': float(joint_val[0]), 'y': float(joint_val[1]), 'z': float(joint_val[2])}
            
            # 頭の位置として、鼻のXYと腕開始のZ平均を保持
            joint_dict["joints"]['head'] = {'x': joint_dict['joints']['nose']['x'], 'y': joint_dict['joints']['nose']['y'], 'z': np.mean([joint_dict['joints']['left_arm']['z'], joint_dict['joints']['right_arm']['z']])}

            with open(params_json_path, 'w') as f:
                json.dump(joint_dict, f, indent=4)

            # トラッキングmp4合成
            out.write(image)

        pose.close()
        hands.close()
        out.release()
        cv2.destroyAllWindows()

        logger.info('人物指推定終了: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)
        
        return True, args.img_dir
    except Exception as e:
        logger.critical("指推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False, None

from mediapipe.python.solutions.pose import PoseLandmark

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
    'left_arm',
    'right_arm',
    'left_elbow',
    'right_elbow',
    'body_left_wrist',
    'body_right_wrist',
    'body_left_pinky',
    'body_right_pinky',
    'body_left_index',
    'body_right_index',
    'body_left_thumb',
    'body_right_thumb',
    'left_hip',
    'right_hip',
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


