# -*- coding: utf-8 -*-
import os
import json

import numpy as np
from tqdm import tqdm
import cv2
import mediapipe as mp
import shutil
import mediapipe as mp

from mmd.prepare import INPUT_VIDEO_NAME
from mmd.utils.MLogger import MLogger

logger = MLogger(__name__, level=1)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def execute(args):
    try:
        logger.info('人物推定開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        # 入力ファイル
        process_input_path = os.path.join(args.img_dir, INPUT_VIDEO_NAME)

        # 既存は削除
        if os.path.exists(os.path.join(args.img_dir, "joints")):
            shutil.rmtree(os.path.join(args.img_dir, "joints"))
        if os.path.exists(os.path.join(args.img_dir, "smooth")):
            shutil.rmtree(os.path.join(args.img_dir, "smooth"))

        # 出力ディレクトリ
        os.makedirs(os.path.join(args.img_dir, "joints"), exist_ok=True)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

            video = cv2.VideoCapture(process_input_path)
            # 幅
            W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # 高さ
            H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 総フレーム数
            count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            process_pose_path = os.path.join(args.img_dir, "pose.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(process_pose_path, fourcc, 30, (W, H))

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
                results = holistic.process(image)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image.flags.writeable = True

                # json出力
                joint_dict = {}
                joint_dict["image"] = {"width": W, "height": H}
                joint_dict["joints"] = {}
                joint_dict["right_hand_joints"] = {}
                joint_dict["left_hand_joints"] = {}

                if results.pose_landmarks and results.pose_landmarks.landmark and results.pose_world_landmarks and results.pose_world_landmarks.landmark:
                    for landmark, world_landmark, output_name in zip(results.pose_landmarks.landmark, results.pose_world_landmarks.landmark, POSE_LANDMARKS):

                        joint_dict["joints"][output_name] = {'x': -float(landmark.x), 'y': -float(landmark.y), 'z': float(landmark.z), 
                                                             'wx': -float(world_landmark.x), 'wy': -float(world_landmark.y), 'wz': float(world_landmark.z), 
                                                             'visibility': float(landmark.visibility)}
                    
                    # 認識映像の出力
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                if results.right_hand_landmarks:
                    for landmark, output_name in zip(results.right_hand_landmarks.landmark, HAND_LANDMARKS):
                        joint_dict["left_hand_joints"][f'left_{output_name}'] = {'x': -float(landmark.x), 'y': -float(landmark.y), 'z': float(landmark.z)}

                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                if results.left_hand_landmarks:
                    for landmark, output_name in zip(results.left_hand_landmarks.landmark, HAND_LANDMARKS):
                        joint_dict["right_hand_joints"][f'right_{output_name}'] = {'x': -float(landmark.x), 'y': -float(landmark.y), 'z': float(landmark.z)}

                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # if results.face_landmarks:
                #     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

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
                
                out.write(image)

                with open(params_json_path, 'w') as f:
                    json.dump(joint_dict, f, indent=4)

            out.release()
            video.release()

        cv2.destroyAllWindows()
            
        return True, args.img_dir
    except Exception as e:
        logger.critical("人物推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False, None

# 左右逆
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
    'body_right_wrist',
    'body_left_wrist',
    'body_right_pinky',
    'body_left_pinky',
    'body_right_index',
    'body_left_index',
    'body_right_thumb',
    'body_left_thumb',
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

HAND_LANDMARKS = [
    "wrist", 
    'thumb1', 
    'thumb2', 
    'thumb3', 
    'thumb', 
    'index1', 
    'index2', 
    'index3', 
    'index', 
    'middle1', 
    'middle2', 
    'middle3', 
    'middle', 
    'ring1', 
    'ring2', 
    'ring3', 
    'ring', 
    'pinky1', 
    'pinky2',
    'pinky3',  
    'pinky', 
]
