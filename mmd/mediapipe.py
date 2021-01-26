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

logger = MLogger(__name__, level=1)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def execute(args):
    try:
        logger.info('人物指推定開始: %s', args.video_file, decoration=MLogger.DECORATION_BOX)

        # 親パス(指定がなければ動画のある場所。Colabはローカルで作成するので指定あり想定)
        base_path = str(pathlib.Path(args.video_file).parent) if not args.parent_dir else args.parent_dir

        if len(args.parent_dir) > 0:
            process_img_dir = base_path
        else:
            process_img_dir = os.path.join(base_path, "{0}_{1:%Y%m%d_%H%M%S}".format(os.path.basename(args.video_file).replace('.', '_'), datetime.datetime.now()))

        process_hand_path = os.path.join(process_img_dir, "detector.mp4")

        # 既存は削除
        if os.path.exists(process_img_dir):
            shutil.rmtree(process_img_dir)

        # フォルダ生成
        os.makedirs(process_img_dir)

        hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, upper_body_only=True)
        cap = cv2.VideoCapture(args.video_file)

        # 幅
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 高さ
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 総フレーム数
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(process_hand_path, fourcc, fps, (W, H))

        for n in tqdm(range(int(count))):
            # 動画から1枚キャプチャして読み込む
            flag, img = cap.read()  # Capture frame-by-frame

            # 動画が終わっていたら終了
            if flag == False:
                break

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

            fname = f'{n:012}'
            fname_path = os.path.join(process_img_dir, 'frames')

            # フォルダ生成
            os.makedirs(fname_path, exist_ok=True)

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

            params_json_path = os.path.join(fname_path, f'joints_{fname}.json')
            with open(params_json_path, 'w') as f:
                json.dump(joint_dict, f, indent=4)

            # トラッキングmp4合成
            out.write(image)

        pose.close()
        hands.close()
        cap.release()            
        out.release()
        cv2.destroyAllWindows()

        logger.info('人物指推定終了: %s', process_img_dir, decoration=MLogger.DECORATION_BOX)
        
        return True, process_img_dir
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


