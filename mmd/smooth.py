# -*- coding: utf-8 -*-
import os
import argparse
import glob
import re
import json
import csv
import cv2
import shutil
import sys

# import vision essentials
import numpy as np
from tqdm import tqdm

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric
from mmd.mmd.VmdData import OneEuroFilter


logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(args):
    try:
        logger.info('関節スムージング処理開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        logger.info("関節スムージング開始", decoration=MLogger.DECORATION_LINE)

        frame_json_pathes = sorted(glob.glob(os.path.join(args.img_dir, "joints", "*.json")), key=sort_by_numeric)

        frame_pattern = re.compile(r'^(\d+)_joints\.')

        all_joints = {}

        for frame_json_path in tqdm(frame_json_pathes, desc=f"Read ... "):
            m = frame_pattern.match(os.path.basename(frame_json_path))
            if m:
                # キーフレの場所を確定（間が空く場合もある）
                fno = int(m.groups()[0])

                frame_joints = {}
                with open(frame_json_path, 'r') as f:
                    frame_joints = json.load(f)
                
                # ジョイントグローバル座標を保持
                for group_name in ["joints", "left_hand_joints", "right_hand_joints"]:
                    for jname, joint in frame_joints[group_name].items():
                        if (group_name, jname, 'x') not in all_joints:
                            all_joints[(group_name, jname, 'x')] = {}

                        if (group_name, jname, 'y') not in all_joints:
                            all_joints[(group_name, jname, 'y')] = {}

                        if (group_name, jname, 'z') not in all_joints:
                            all_joints[(group_name, jname, 'z')] = {}
                        
                        if (group_name, jname, 'wx') not in all_joints:
                            all_joints[(group_name, jname, 'wx')] = {}

                        if (group_name, jname, 'wy') not in all_joints:
                            all_joints[(group_name, jname, 'wy')] = {}

                        if (group_name, jname, 'wz') not in all_joints:
                            all_joints[(group_name, jname, 'wz')] = {}
                        
                        all_joints[(group_name, jname, 'x')][fno] = joint["x"]
                        all_joints[(group_name, jname, 'y')][fno] = joint["y"]
                        all_joints[(group_name, jname, 'z')][fno] = joint["z"]

                        if "wx" in joint and "wy" in joint and "wz" in joint:
                            all_joints[(group_name, jname, 'wx')][fno] = joint["wx"]
                            all_joints[(group_name, jname, 'wy')][fno] = joint["wy"]
                            all_joints[(group_name, jname, 'wz')][fno] = joint["wz"]
                    
        # スムージング
        for (group_name, jname, axis), joints in tqdm(all_joints.items(), desc=f"Filter ... "):
            filter = OneEuroFilter(freq=30, mincutoff=1, beta=0.00000000001, dcutoff=1)
            for fno, joint in joints.items():
                all_joints[(group_name, jname, axis)][fno] = filter(joint, fno)

        # 出力先ソート済みフォルダ
        smoothed_dir_path = os.path.join(args.img_dir, "smooth")
        os.makedirs(smoothed_dir_path, exist_ok=True)

        # 出力
        for frame_json_path in tqdm(frame_json_pathes, desc=f"Save ... "):
            m = frame_pattern.match(os.path.basename(frame_json_path))
            if m:
                # キーフレの場所を確定（間が空く場合もある）
                fno = int(m.groups()[0])

                frame_joints = {}
                with open(frame_json_path, 'r', encoding='utf-8') as f:
                    frame_joints = json.load(f)
                
                # ジョイントグローバル座標を保存
                for group_name in ["joints", "left_hand_joints", "right_hand_joints"]:
                    for jname, joint in frame_joints[group_name].items():
                        frame_joints[group_name][jname]["x"] = all_joints[(group_name, jname, 'x')][fno]
                        frame_joints[group_name][jname]["y"] = all_joints[(group_name, jname, 'y')][fno]
                        frame_joints[group_name][jname]["z"] = all_joints[(group_name, jname, 'z')][fno]

                        if "wx" in joint and "wy" in joint and "wz" in joint:
                            frame_joints[group_name][jname]["wx"] = all_joints[(group_name, jname, 'wx')][fno]
                            frame_joints[group_name][jname]["wy"] = all_joints[(group_name, jname, 'wy')][fno]
                            frame_joints[group_name][jname]["wz"] = all_joints[(group_name, jname, 'wz')][fno]

                smooth_json_path = os.path.join(smoothed_dir_path, f"smooth_{fno:012}.json")
                
                with open(smooth_json_path, 'w', encoding='utf-8') as f:
                    json.dump(frame_joints, f, indent=4)

        logger.info('関節スムージング処理終了: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("関節スムージングで予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False
