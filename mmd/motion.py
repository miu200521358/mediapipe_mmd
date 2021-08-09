# -*- coding: utf-8 -*-
from numpy.lib.function_base import flip
from mmd.module.MParams import BoneLinks
import os
import argparse
import glob
import re
import json
import csv
import datetime
import numpy as np
from tqdm import tqdm
import math
from mmd.utils import MServiceUtils

from mmd.utils.MLogger import MLogger
from mmd.utils.MServiceUtils import sort_by_numeric

from mmd.module.MMath import MQuaternion, MVector3D, MVector2D, MMatrix4x4, MRect, MVector4D, fromEulerAngles
from mmd.mmd.VmdData import VmdBoneFrame, VmdMorphFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk, OneEuroFilter
from mmd.mmd.VmdWriter import VmdWriter
from mmd.mmd.PmxData import Material, PmxModel, Bone, Vertex, Bdef1, Ik, IkLink, DisplaySlot
from mmd.mmd.PmxWriter import PmxWriter
from mmd.holistic import HAND_LANDMARKS, POSE_LANDMARKS
from mmd.utils.MServiceUtils import get_file_encoding, calc_global_pos, separate_local_qq

logger = MLogger(__name__, level=1)

MIKU_METER = 12.5

def execute(args):
    try:
        logger.info('モーション生成処理開始: {0}', args.img_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.img_dir):
            logger.error("指定された処理用ディレクトリが存在しません。: {0}", args.img_dir, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(os.path.join(args.img_dir, "smooth")):
            logger.error("指定されたスムージングディレクトリが存在しません。\nスムージングが完了していない可能性があります。: {0}", \
                         os.path.join(args.img_dir, "smooth"), decoration=MLogger.DECORATION_BOX)
            return False

        # モデルをCSVから読み込む
        miku_model = read_bone_csv(args.bone_config)
        motion_dir_path = os.path.join(args.img_dir, "motion")
        os.makedirs(motion_dir_path, exist_ok=True)
        
        process_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        smooth_pattern = re.compile(r'^smooth_(\d+)\.')
        start_heel = np.array([9999999, 9999999, 9999999])

        pmx_writer = PmxWriter()
        vmd_writer = VmdWriter()

        smooth_json_pathes = sorted(glob.glob(os.path.join(args.img_dir, "smooth", "*.json")), key=sort_by_numeric)

        trace_mov_motion = VmdMotion()
        trace_rot_motion = VmdMotion()
        trace_miku_motion = VmdMotion()

        # KEY: 処理対象ボーン名, VALUE: vecリスト
        target_bone_global_vecs = {}
        fnos = []
        # 画像サイズ
        image_size = MVector3D()

        for mmname in ["全ての親", "センター", "グルーブ", "pelvis", "pelvis2", "spine1", "spine2", "neck", "head", "head_tail", "left_collar", "right_collar", "body_left_wrist_tail", "body_right_wrist_tail"]:
            target_bone_global_vecs[mmname] = {}
        
        logger.info("モーション結果位置計算開始", decoration=MLogger.DECORATION_LINE)

        with tqdm(total=(len(smooth_json_pathes) * (len(POSE_LANDMARKS) + len(HAND_LANDMARKS) + 12))) as pchar:
            for sidx, smooth_json_path in enumerate(smooth_json_pathes):
                m = smooth_pattern.match(os.path.basename(smooth_json_path))
                if m:
                    # キーフレの場所を確定（間が空く場合もある）
                    fno = int(m.groups()[0])

                    frame_joints = {}
                    with open(smooth_json_path, 'r', encoding='utf-8') as f:
                        frame_joints = json.load(f)
                    
                    if "joints" in frame_joints:
                        fnos.append(fno)
                        
                        if len(frame_joints["joints"].keys()) > 0:
                            for jname, joint in frame_joints["joints"].items():
                                if jname not in target_bone_global_vecs:
                                    target_bone_global_vecs[jname] = {}
                                target_bone_global_vecs[jname][fno] = (np.array([joint["wx"], joint["wy"], joint["wz"]]) * MIKU_METER)
                                pchar.update(1)
                        else:
                            for jname in POSE_LANDMARKS:
                                if jname not in target_bone_global_vecs:
                                    target_bone_global_vecs[jname] = {}

                                if fno == 0:
                                    target_bone_global_vecs[jname][fno] = np.array([0, 0, 0])
                                else:
                                    target_bone_global_vecs[jname][fno] = np.array([target_bone_global_vecs[jname][fno - 1][0], target_bone_global_vecs[jname][fno - 1][1], target_bone_global_vecs[jname][fno - 1][2]])

                                pchar.update(1)
                        
                        for mbname in ["全ての親", "グルーブ"]:
                            target_bone_global_vecs[mbname][fno] = np.array([0, 0, 0])

                        if 'left_hip' in target_bone_global_vecs and 'right_hip' in target_bone_global_vecs and fno in target_bone_global_vecs['left_hip'] and fno in target_bone_global_vecs['right_hip']:
                            # 下半身
                            target_bone_global_vecs['pelvis'][fno] = np.mean([target_bone_global_vecs['left_hip'][fno], target_bone_global_vecs['right_hip'][fno]], axis=0)

                            # 上半身
                            target_bone_global_vecs['spine1'][fno] = np.mean([target_bone_global_vecs['left_hip'][fno], target_bone_global_vecs['right_hip'][fno]], axis=0)
                        else:
                            if fno == 0:
                                target_bone_global_vecs["pelvis"][fno] = np.array([0, 0, 0])
                                target_bone_global_vecs["spine1"][fno] = np.array([0, 0, 0])
                            else:
                                target_bone_global_vecs["pelvis"][fno] = np.array([target_bone_global_vecs["pelvis"][fno - 1][0], target_bone_global_vecs["pelvis"][fno - 1][1], target_bone_global_vecs["pelvis"][fno - 1][2]])
                                target_bone_global_vecs["spine1"][fno] = np.array([target_bone_global_vecs["spine1"][fno - 1][0], target_bone_global_vecs["spine1"][fno - 1][1], target_bone_global_vecs["spine1"][fno - 1][2]])

                        pchar.update(2)

                        # 上半身2
                        if 'left_hip' in target_bone_global_vecs and 'right_hip' in target_bone_global_vecs and 'left_shoulder' in target_bone_global_vecs and 'right_shoulder' in target_bone_global_vecs \
                            and fno in target_bone_global_vecs['left_hip'] and fno in target_bone_global_vecs['right_hip'] and fno in target_bone_global_vecs['left_shoulder'] and fno in target_bone_global_vecs['right_shoulder']:
                            target_bone_global_vecs['spine2'][fno] = np.mean([target_bone_global_vecs['left_hip'][fno], target_bone_global_vecs['right_hip'][fno], 
                                                                            target_bone_global_vecs['left_shoulder'][fno], target_bone_global_vecs['right_shoulder'][fno]], axis=0)
                        else:
                            if fno == 0:
                                target_bone_global_vecs["spine2"][fno] = np.array([0, 0, 0])
                            else:
                                target_bone_global_vecs["spine2"][fno] = np.array([target_bone_global_vecs["spine2"][fno - 1][0], target_bone_global_vecs["spine2"][fno - 1][1], target_bone_global_vecs["spine2"][fno - 1][2]])
                        pchar.update(1)

                        # 下半身先
                        if 'pelvis' in target_bone_global_vecs and 'spine1' in target_bone_global_vecs and 'spine2' in target_bone_global_vecs \
                            and fno in target_bone_global_vecs['pelvis'] and fno in target_bone_global_vecs['spine1'] and fno in target_bone_global_vecs['spine2']:
                            target_bone_global_vecs['pelvis2'][fno] = target_bone_global_vecs['pelvis'][fno] + (target_bone_global_vecs['spine1'][fno] - target_bone_global_vecs['spine2'][fno])
                        else:
                            if fno == 0:
                                target_bone_global_vecs["pelvis2"][fno] = np.array([0, 0, 0])
                            else:
                                target_bone_global_vecs["pelvis2"][fno] = np.array([target_bone_global_vecs["pelvis2"][fno - 1][0], target_bone_global_vecs["pelvis2"][fno - 1][1], target_bone_global_vecs["pelvis2"][fno - 1][2]])
                        pchar.update(1)

                        if 'left_shoulder' in target_bone_global_vecs and 'right_shoulder' in target_bone_global_vecs \
                            and fno in target_bone_global_vecs['left_shoulder'] and fno in target_bone_global_vecs['right_shoulder']:
                            # 首
                            target_bone_global_vecs['neck'][fno] = np.mean([target_bone_global_vecs['left_shoulder'][fno], target_bone_global_vecs['right_shoulder'][fno]], axis=0)
                        
                            # 左肩
                            target_bone_global_vecs['left_collar'][fno] = np.mean([target_bone_global_vecs['left_shoulder'][fno], target_bone_global_vecs['right_shoulder'][fno]], axis=0)
                        
                            # 右肩
                            target_bone_global_vecs['right_collar'][fno] = np.mean([target_bone_global_vecs['left_shoulder'][fno], target_bone_global_vecs['right_shoulder'][fno]], axis=0)
                        else:
                            if fno == 0:
                                target_bone_global_vecs["neck"][fno] = np.array([0, 0, 0])
                                target_bone_global_vecs["left_collar"][fno] = np.array([0, 0, 0])
                                target_bone_global_vecs["right_collar"][fno] = np.array([0, 0, 0])
                            else:
                                target_bone_global_vecs["neck"][fno] = np.array([target_bone_global_vecs["neck"][fno - 1][0], target_bone_global_vecs["neck"][fno - 1][1], target_bone_global_vecs["neck"][fno - 1][2]])
                                target_bone_global_vecs["left_collar"][fno] = np.array([target_bone_global_vecs["left_collar"][fno - 1][0], target_bone_global_vecs["left_collar"][fno - 1][1], target_bone_global_vecs["left_collar"][fno - 1][2]])
                                target_bone_global_vecs["right_collar"][fno] = np.array([target_bone_global_vecs["right_collar"][fno - 1][0], target_bone_global_vecs["right_collar"][fno - 1][1], target_bone_global_vecs["right_collar"][fno - 1][2]])
                        pchar.update(3)

                        # 頭
                        if 'left_ear' in target_bone_global_vecs and 'right_ear' in target_bone_global_vecs \
                            and fno in target_bone_global_vecs['left_ear'] and fno in target_bone_global_vecs['right_ear']:
                            target_bone_global_vecs['head'][fno] = np.mean([target_bone_global_vecs['left_ear'][fno], target_bone_global_vecs['right_ear'][fno]], axis=0)
                        else:
                            if fno == 0:
                                target_bone_global_vecs["head"][fno] = np.array([0, 0, 0])
                            else:
                                target_bone_global_vecs["head"][fno] = np.array([target_bone_global_vecs["head"][fno - 1][0], target_bone_global_vecs["head"][fno - 1][1], target_bone_global_vecs["head"][fno - 1][2]])
                        pchar.update(1)

                        # 頭先
                        if 'head' in target_bone_global_vecs and 'neck' in target_bone_global_vecs \
                            and fno in target_bone_global_vecs['head'] and fno in target_bone_global_vecs['neck']:
                            target_bone_global_vecs['head_tail'][fno] = target_bone_global_vecs['head'][fno] + (target_bone_global_vecs['head'][fno] - target_bone_global_vecs['neck'][fno])
                        else:
                            if fno == 0:
                                target_bone_global_vecs["head_tail"][fno] = np.array([0, 0, 0])
                            else:
                                target_bone_global_vecs["head_tail"][fno] = np.array([target_bone_global_vecs["head_tail"][fno - 1][0], target_bone_global_vecs["head_tail"][fno - 1][1], target_bone_global_vecs["head_tail"][fno - 1][2]])
                        pchar.update(1)

                        # 左手首先
                        if 'body_left_index' in target_bone_global_vecs and 'body_left_pinky' in target_bone_global_vecs \
                            and fno in target_bone_global_vecs['body_left_index'] and fno in target_bone_global_vecs['body_left_pinky']:
                            target_bone_global_vecs['body_left_wrist_tail'][fno] = np.mean([target_bone_global_vecs['body_left_index'][fno], target_bone_global_vecs['body_left_pinky'][fno]], axis=0)
                        else:
                            if fno == 0:
                                target_bone_global_vecs["body_left_wrist_tail"][fno] = np.array([0, 0, 0])
                            else:
                                target_bone_global_vecs["body_left_wrist_tail"][fno] = np.array([target_bone_global_vecs["body_left_wrist_tail"][fno - 1][0], target_bone_global_vecs["body_left_wrist_tail"][fno - 1][1], target_bone_global_vecs["body_left_wrist_tail"][fno - 1][2]])
                        pchar.update(1)

                        # 右手首先
                        if 'body_right_index' in target_bone_global_vecs and 'body_right_pinky' in target_bone_global_vecs \
                            and fno in target_bone_global_vecs['body_right_index'] and fno in target_bone_global_vecs['body_right_pinky']:
                            target_bone_global_vecs['body_right_wrist_tail'][fno] = np.mean([target_bone_global_vecs['body_right_index'][fno], target_bone_global_vecs['body_right_pinky'][fno]], axis=0)
                        else:
                            if fno == 0:
                                target_bone_global_vecs["body_right_wrist_tail"][fno] = np.array([0, 0, 0])
                            else:
                                target_bone_global_vecs["body_right_wrist_tail"][fno] = np.array([target_bone_global_vecs["body_right_wrist_tail"][fno - 1][0], target_bone_global_vecs["body_right_wrist_tail"][fno - 1][1], target_bone_global_vecs["body_right_wrist_tail"][fno - 1][2]])
                        pchar.update(1)

                        if 'left_hip' in frame_joints["joints"] and 'right_hip' in frame_joints["joints"]:
                            relative_pelvis_pos = np.mean([[frame_joints["joints"]["left_hip"]["x"], frame_joints["joints"]["left_hip"]["y"]], [frame_joints["joints"]["right_hip"]["x"], frame_joints["joints"]["right_hip"]["y"]]], axis=0)
                            target_bone_global_vecs["センター"][fno] = MVector3D((image_size.x() / 2 - (relative_pelvis_pos[0] * image_size.x())), relative_pelvis_pos[1] * image_size.y(), 0)
                        else:
                            if fno == 0:
                                target_bone_global_vecs["センター"][fno] = MVector3D()
                            else:
                                target_bone_global_vecs["センター"][fno] = target_bone_global_vecs["センター"][fno - 1].copy()
                    else:
                        pchar.update(len(POSE_LANDMARKS))
                    
                    for direction in ["left", "right"]:
                        if len(frame_joints[f"{direction}_hand_joints"].keys()) > 0:
                            for jname, joint in frame_joints[f"{direction}_hand_joints"].items():
                                if jname not in target_bone_global_vecs:
                                    target_bone_global_vecs[jname] = {}
                                target_bone_global_vecs[jname][fno] = (np.array([joint["x"], joint["y"], joint["z"]]) * MIKU_METER * 3)
                                pchar.update(1)
                        else:
                            for lname in HAND_LANDMARKS:
                                jname = f'{direction}_{lname}'

                                if jname not in target_bone_global_vecs:
                                    target_bone_global_vecs[jname] = {}

                                if fno == 0:
                                    target_bone_global_vecs[jname][fno] = np.array([0, 0, 0])
                                else:
                                    target_bone_global_vecs[jname][fno] = np.array([target_bone_global_vecs[jname][fno - 1][0], target_bone_global_vecs[jname][fno - 1][1], target_bone_global_vecs[jname][fno - 1][2]])

                                pchar.update(1)

            trace_model = PmxModel()
            trace_model.vertices["vertices"] = []
            # 空テクスチャを登録
            trace_model.textures.append("")

            # 全ての親 ------------------------
            trace_model.bones["全ての親"] = Bone("全ての親", "Root", MVector3D(), -1, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
            trace_model.bones["全ての親"].index = 0
            trace_model.bone_indexes[0] = "全ての親"
            trace_model.display_slots["Root"] = DisplaySlot("Root", "Root", 1, 0)
            trace_model.display_slots["Root"].references.append(trace_model.bones["全ての親"].index)

            # モーフの表示枠
            trace_model.display_slots["表情"] = DisplaySlot("表情", "Exp", 1, 1)

            # センター ------------------------
            trace_model.bones["センター"] = Bone("センター", "Center", MVector3D(0, 9, 0), trace_model.bones["全ての親"].index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
            trace_model.bones["センター"].index = len(trace_model.bones) - 1
            trace_model.bone_indexes[trace_model.bones["センター"].index] = "センター"
            trace_model.display_slots["センター"] = DisplaySlot("センター", "Center", 0, 0)
            trace_model.display_slots["センター"].references.append(trace_model.bones["センター"].index)

            # グルーブ ------------------------
            trace_model.bones["グルーブ"] = Bone("グルーブ", "Groove", MVector3D(0, 9.5, 0), trace_model.bones["センター"].index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
            trace_model.bones["グルーブ"].index = len(trace_model.bones) - 1
            trace_model.bone_indexes[trace_model.bones["グルーブ"].index] = "グルーブ"
            trace_model.display_slots["センター"].references.append(trace_model.bones["グルーブ"].index)

            # その他
            for display_name in ["体幹", "左足", "右足", "左手", "右手", "左指", "右指", "顔", "眉", "鼻", "目", "口", "輪郭"]:
                trace_model.display_slots[display_name] = DisplaySlot(display_name, display_name, 0, 0)
            
            # # 右手首元 ------------------------
            # trace_model.bones["右手首元"] = Bone("右手首元", "Right Wirst", MVector3D(-5, 13, 0), trace_model.bones["センター"].index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
            # trace_model.bones["右手首元"].index = len(trace_model.bones) - 1
            # trace_model.bone_indexes[trace_model.bones["右手首元"].index] = "右手首元"
            # trace_model.display_slots["右指"].references.append(trace_model.bones["右手首元"].index)

            # # 左手首元 ------------------------
            # trace_model.bones["左手首元"] = Bone("左手首元", "Left Wirst", MVector3D(5, 13, 0), trace_model.bones["センター"].index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
            # trace_model.bones["左手首元"].index = len(trace_model.bones) - 1
            # trace_model.bone_indexes[trace_model.bones["左手首元"].index] = "左手首元"
            # trace_model.display_slots["左指"].references.append(trace_model.bones["左手首元"].index)

            logger.info("モーション(移動)計算開始", decoration=MLogger.DECORATION_LINE)

            with tqdm(total=(len(PMX_CONNECTIONS.keys()) * (len(fnos) + 1))) as pchar:
                for fno in fnos:
                    joint = target_bone_global_vecs["センター"][fno]                    
                    center_bf = trace_mov_motion.calc_bf("センター", fno)
                    groove_bf = trace_mov_motion.calc_bf("グルーブ", fno)
                    # groove_bf.position.setY(joint.y() + trace_model.bones["グルーブ"].position.y())
                    groove_bf.position.setY(joint.y())
                    groove_bf.key = True
                    trace_mov_motion.bones[groove_bf.name][fno] = groove_bf

                    center_bf.position.setX(joint.x())
                    center_bf.key = True
                    trace_mov_motion.bones[center_bf.name][fno] = center_bf
                    pchar.update(1)

                for jidx, (jname, pconn) in enumerate(PMX_CONNECTIONS.items()):
                    if jname not in target_bone_global_vecs:
                        pchar.update(len(fnos))
                        continue

                    mname = pconn['mmd']
                        
                    if jname in ["left_wrist", "right_wrist"]:
                        trace_model.bones[mname] = trace_model.bones[mname[:3]].copy()
                        trace_model.bones[mname].parent_index = trace_model.bones[mname[:3]].index
                        trace_model.bones[mname].name = mname
                        trace_model.bones[mname].english_name = mname
                        trace_model.bones[mname].index = len(list(trace_model.bones.keys())) - 1
                        trace_model.bone_indexes[trace_model.bones[mname].index] = mname
                        trace_model.display_slots[pconn["display"]].references.append(trace_model.bones[mname].index)

                        # from_links = trace_model.create_link_2_top_one(mname[:3], is_defined=False)

                        # for fno in fnos:
                        #     now_wrist_vec = calc_global_pos_from_mov(trace_model, from_links, trace_mov_motion, fno)

                        #     bf = trace_mov_motion.calc_bf(mname, fno)
                        #     parent_bf = trace_mov_motion.calc_bf(mname[:3], fno)
                        #     bf.position = parent_bf.position.copy()
                        #     bf.key = True
                        #     trace_mov_motion.bones[mname][fno] = bf
                    else:
                        # ボーン登録
                        create_bone(trace_model, jname, pconn, target_bone_global_vecs, miku_model)

                        pmname = PMX_CONNECTIONS[pconn['parent']]['mmd'] if pconn['parent'] in PMX_CONNECTIONS else pconn['parent']

                        # モーションも登録
                        for fno in fnos:
                            if fno in target_bone_global_vecs[jname]:
                                joint = target_bone_global_vecs[jname][fno]
                                parent_joint = target_bone_global_vecs[pconn['parent']][fno]
                                
                                trace_bf = trace_mov_motion.calc_bf(mname, fno)
                                # # parentが全ての親の場合
                                # trace_bf.position = MVector3D(joint) - trace_model.bones[mname].position
                                # parentが親ボーンの場合
                                trace_bf.position = MVector3D(joint) - MVector3D(parent_joint) \
                                                    - (trace_model.bones[mname].position - trace_model.bones[pmname].position)
                                if mname in ["センター", "グルーブ"]:
                                    pos = trace_bf.position * (trace_model.bones["頭先"].position.y() / image_size.y())
                                    if mname == "センター":
                                        trace_bf.position.setX(pos.x())
                                        trace_bf.position.setY(0)
                                        trace_bf.position.setZ(pos.z())
                                    if mname == "グルーブ":
                                        trace_bf.position.setX(0)
                                        trace_bf.position.setY(pos.y() + trace_model.bones["グルーブ"].position.y())
                                        trace_bf.position.setZ(0)
                                trace_bf.key = True
                                trace_mov_motion.bones[mname][fno] = trace_bf
                            pchar.update(1)

            for bidx, bone in trace_model.bones.items():
                # 表示先の設定
                if bone.tail_index and bone.tail_index in trace_model.bones:
                    bone.flag |= 0x0001
                    bone.tail_index = trace_model.bones[bone.tail_index].index
                else:
                    bone.tail_index = -1
            
            # if "右手首元" in trace_mov_motion.bones:
            #     del trace_mov_motion.bones["右手首元"]

            # if "左手首元" in trace_mov_motion.bones:
            #     del trace_mov_motion.bones["左手首元"]

            trace_model_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_mov_model.pmx")
            logger.info("トレース(移動)モデル生成開始【{0}】", os.path.basename(trace_model_path), decoration=MLogger.DECORATION_LINE)
            trace_model.name = f"Trace結果 (移動用)"
            trace_model.english_name = f"TraceModel (Move)"
            trace_model.comment = f"Trace結果 表示用モデル (移動用)\n足ＩＫがありません"
            pmx_writer.write(trace_model, trace_model_path)

            trace_mov_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_mov.vmd")
            logger.info("モーション(移動)生成開始【{0}】", os.path.basename(trace_mov_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(trace_model, trace_mov_motion, trace_mov_motion_path)

            # 足ＩＫ ------------------------
            ik_show = VmdShowIkFrame()
            ik_show.fno = 0
            ik_show.show = 1

            for direction in ["左", "右"]:
                create_bone_leg_ik(trace_model, direction)
                
                ik_show.ik.append(VmdInfoIk(f'{direction}足ＩＫ', 0))
                ik_show.ik.append(VmdInfoIk(f'{direction}つま先ＩＫ', 0))

            trace_rot_motion.showiks.append(ik_show)
            trace_miku_motion.showiks.append(ik_show)

            trace_model_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_rot_model.pmx")
            logger.info("トレース(移動)モデル生成開始【{0}】", os.path.basename(trace_model_path), decoration=MLogger.DECORATION_LINE)
            trace_model.name = f"Trace結果 (回転用)"
            trace_model.english_name = f"TraceModel (Rot)"
            trace_model.comment = f"Trace結果 表示用モデル (回転用)\n足ＩＫはモーション側でOFFにしています"
            pmx_writer.write(trace_model, trace_model_path)
            
            logger.info("モーション(回転)計算開始", decoration=MLogger.DECORATION_LINE)
            
            with tqdm(total=((len(VMD_CONNECTIONS.keys()) + 2) * (len(fnos)))) as pchar:
                for mname in ["センター", "グルーブ"]:
                    for fno in fnos:
                        mov_bf = trace_mov_motion.calc_bf(mname, fno)
                        now_bf = trace_rot_motion.calc_bf(mname, fno)
                        now_bf.position = mov_bf.position
                        now_bf.key = True
                        trace_rot_motion.bones[mname][fno] = now_bf

                        pchar.update(1)

                for jname, jconn in VMD_CONNECTIONS.items():
                    mname = PMX_CONNECTIONS[jname]['mmd']
                    direction_from_mname = PMX_CONNECTIONS[jconn["direction"][0]]['mmd']
                    direction_to_mname = PMX_CONNECTIONS[jconn["direction"][1]]['mmd']
                    up_from_mname = PMX_CONNECTIONS[jconn["up"][0]]['mmd']
                    up_to_mname = PMX_CONNECTIONS[jconn["up"][1]]['mmd']
                    cross_from_mname = PMX_CONNECTIONS[jconn["cross"][0]]['mmd'] if 'cross' in jconn else direction_from_mname
                    cross_to_mname = PMX_CONNECTIONS[jconn["cross"][1]]['mmd'] if 'cross' in jconn else direction_to_mname

                    # トレースモデルの初期姿勢
                    trace_direction_from_vec = trace_model.bones[direction_from_mname].position
                    trace_direction_to_vec = trace_model.bones[direction_to_mname].position
                    trace_direction = (trace_direction_to_vec - trace_direction_from_vec).normalized()

                    trace_up_from_vec = trace_model.bones[up_from_mname].position
                    trace_up_to_vec = trace_model.bones[up_to_mname].position
                    trace_up = (trace_up_to_vec - trace_up_from_vec).normalized()

                    trace_cross_from_vec = trace_model.bones[cross_from_mname].position
                    trace_cross_to_vec = trace_model.bones[cross_to_mname].position
                    trace_cross = (trace_cross_to_vec - trace_cross_from_vec).normalized()

                    trace_up_cross = MVector3D.crossProduct(trace_up, trace_cross).normalized()
                    trace_stance_qq = MQuaternion.fromDirection(trace_direction, trace_up_cross)

                    direction_from_links = trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                    direction_to_links = trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                    up_from_links = trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
                    up_to_links = trace_model.create_link_2_top_one(up_to_mname, is_defined=False)
                    cross_from_links = trace_model.create_link_2_top_one(cross_from_mname, is_defined=False)
                    cross_to_links = trace_model.create_link_2_top_one(cross_to_mname, is_defined=False)

                    for fno in fnos:
                        now_direction_from_vec = calc_global_pos_from_mov(trace_model, direction_from_links, trace_mov_motion, fno)
                        now_direction_to_vec = calc_global_pos_from_mov(trace_model, direction_to_links, trace_mov_motion, fno)
                        now_up_from_vec = calc_global_pos_from_mov(trace_model, up_from_links, trace_mov_motion, fno)
                        now_up_to_vec = calc_global_pos_from_mov(trace_model, up_to_links, trace_mov_motion, fno)
                        now_cross_from_vec = calc_global_pos_from_mov(trace_model, cross_from_links, trace_mov_motion, fno)
                        now_cross_to_vec = calc_global_pos_from_mov(trace_model, cross_to_links, trace_mov_motion, fno)

                        # トレースモデルの回転量 ------------
                        now_direction = (now_direction_to_vec - now_direction_from_vec).normalized()
                        now_up = (now_up_to_vec - now_up_from_vec).normalized()
                        now_cross = (now_cross_to_vec - now_cross_from_vec).normalized()

                        now_up_cross = MVector3D.crossProduct(now_up, now_cross).normalized()
                        now_stance_qq = MQuaternion.fromDirection(now_direction, now_up_cross)

                        cancel_qq = MQuaternion()
                        for cancel_jname in jconn["cancel"]:
                            cancel_qq *= trace_rot_motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                        now_qq = cancel_qq.inverted() * now_stance_qq * trace_stance_qq.inverted()

                        now_bf = trace_rot_motion.calc_bf(mname, fno)
                        now_bf.rotation = now_qq
                        now_bf.key = True
                        trace_rot_motion.bones[mname][fno] = now_bf

                        pchar.update(1)

            trace_rot_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_rot.vmd")
            logger.info("トレースモーション(回転)生成開始【{0}】", os.path.basename(trace_rot_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(trace_model, trace_rot_motion, trace_rot_motion_path)

            logger.info("モーション(あにまさ式ミク)計算開始", decoration=MLogger.DECORATION_LINE)

            with tqdm(total=(len(trace_rot_motion.bones.keys()) * len(fnos))) as pchar:
                for mname in ["センター", "グルーブ"]:
                    for fno in fnos:
                        rot_bf = trace_rot_motion.calc_bf(mname, fno)
                        miku_bf = trace_miku_motion.calc_bf(mname, fno)
                        miku_bf.position = rot_bf.position.copy()
                        miku_bf.rotation = rot_bf.rotation.copy()
                        miku_bf.key = True

                        trace_miku_motion.bones[mname][fno] = miku_bf
                        pchar.update(1)
                        continue
                    
                for jname, jconn in VMD_CONNECTIONS.items():
                    mname = PMX_CONNECTIONS[jname]['mmd']
                    parent_jnmae = PMX_CONNECTIONS[jname]['parent']

                    parent_name = "センター"
                    if mname not in ["全ての親", "センター", "グルーブ"] and parent_jnmae in PMX_CONNECTIONS:
                        parent_name = PMX_CONNECTIONS[parent_jnmae]['mmd']
                    
                    base_axis = PMX_CONNECTIONS[jname]["axis"] if jname in PMX_CONNECTIONS else None
                    parent_axis = PMX_CONNECTIONS[jname]["parent_axis"] if jname in PMX_CONNECTIONS and "parent_axis" in PMX_CONNECTIONS[jname] else PMX_CONNECTIONS[parent_jnmae]["axis"] if parent_jnmae in PMX_CONNECTIONS else None
                    trace_parent_local_x_qq = trace_model.get_local_x_qq(parent_name, parent_axis)
                    trace_target_local_x_qq = trace_model.get_local_x_qq(mname, base_axis)
                    miku_parent_local_x_qq = miku_model.get_local_x_qq(parent_name, parent_axis)
                    miku_target_local_x_qq = miku_model.get_local_x_qq(mname, base_axis)

                    parent_local_x_qq = miku_parent_local_x_qq.inverted() * trace_parent_local_x_qq
                    target_local_x_qq = miku_target_local_x_qq.inverted() * trace_target_local_x_qq

                    miku_local_x_axis = miku_model.get_local_x_axis(mname)
                    miku_local_y_axis = MVector3D.crossProduct(miku_local_x_axis, MVector3D(0, 0, 1))

                    for fno in fnos:
                        rot_bf = trace_rot_motion.calc_bf(mname, fno)
                        miku_bf = trace_miku_motion.calc_bf(mname, fno)

                        miku_bf.position = rot_bf.position.copy()
                        new_miku_qq = rot_bf.rotation.copy()

                        if (len(mname) > 2 and mname[2] == "指") or mname[1:] in ["ひざ"]:
                            # 指・ひざは念のためX捩り除去
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, new_miku_qq, miku_local_x_axis)
                            new_miku_qq = now_yz_qq
                        
                        if (len(mname) > 2 and mname[2] == "指"):
                            pass
                        elif mname[1:] in ["手首"]:
                            new_miku_qq = parent_local_x_qq.inverted() * new_miku_qq
                        elif mname[1:] in ["肩", "足"]:
                            new_miku_qq = new_miku_qq * target_local_x_qq
                        else:
                            new_miku_qq = parent_local_x_qq.inverted() * new_miku_qq * target_local_x_qq

                        if len(mname) > 2 and mname[2] == "指":
                            # 指は正方向にしか曲がらない
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, new_miku_qq, miku_local_x_axis)
                            new_miku_qq = MQuaternion.fromAxisAndAngle((MVector3D(0, 0, -1) if mname[0] == "左" else MVector3D(0, 0, 1)), now_yz_qq.toDegree())
                        elif mname[1:] in ["ひざ"]:
                            # ひざは足IK用にYのみ
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, new_miku_qq, miku_local_x_axis)
                            new_miku_qq = MQuaternion.fromAxisAndAngle(miku_local_y_axis, now_yz_qq.toDegree())

                        miku_bf.rotation = new_miku_qq
                        miku_bf.key = True

                        trace_miku_motion.bones[mname][fno] = miku_bf
                        pchar.update(1)

            trace_miku_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_miku.vmd")
            logger.info("トレースモーション(あにまさ式ミク)生成開始【{0}】", os.path.basename(trace_miku_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(miku_model, trace_miku_motion, trace_miku_motion_path)

            logger.info("モーション(あにまさ式ミク)外れ値削除開始", decoration=MLogger.DECORATION_LINE)

            loop_cnt = 2
            with tqdm(total=(len(trace_miku_motion.bones.keys()) * len(fnos) * loop_cnt)) as pchar:
                for n in range(loop_cnt):
                    bone_names = list(trace_miku_motion.bones.keys())
                    for mname in bone_names:
                        if mname in ["センター", "グルーブ"]:
                            for prev_fno, fno in zip(fnos[:-1:2], fnos[1::2]):
                                prev_bf = trace_miku_motion.calc_bf(mname, prev_fno)
                                now_bf = trace_miku_motion.calc_bf(mname, fno)

                                distance = prev_bf.position.distanceToPoint(now_bf.position)
                                if 0.005 + (n * 0.001) > distance:
                                    # 離れすぎてるのは削除(回数を重ねるほどに変化量の検知を鈍くする)
                                    if fno in trace_miku_motion.bones[mname]:
                                        del trace_miku_motion.bones[mname][fno]
                                pchar.update(2)
                            continue
                        
                        for prev_fno, fno in zip(fnos[:-1:2], fnos[1::2]):
                            prev_bf = trace_miku_motion.calc_bf(mname, prev_fno)
                            now_bf = trace_miku_motion.calc_bf(mname, fno)

                            dot = MQuaternion.dotProduct(prev_bf.rotation, now_bf.rotation)
                            if 0.8 - (n * 0.1) > dot:
                                # 離れすぎてるのは削除(回数を重ねるほどに変化量の検知を鈍くする)
                                if fno in trace_miku_motion.bones[mname]:
                                    del trace_miku_motion.bones[mname][fno]
                            pchar.update(2)

            logger.info("モーション(あにまさ式ミク)スムージング開始", decoration=MLogger.DECORATION_LINE)

            loop_cnt = 1
            with tqdm(total=(len(trace_miku_motion.bones.keys()) * len(fnos) * loop_cnt)) as pchar:
                for n in range(loop_cnt):
                    bone_names = list(trace_miku_motion.bones.keys())
                    for mname in bone_names:
                        if mname in ["センター", "グルーブ"]:
                            mxfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.1, dcutoff=1)
                            myfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.1, dcutoff=1)
                            mzfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.1, dcutoff=1)
                            for fidx, fno in enumerate(fnos):
                                bf = trace_miku_motion.calc_bf(mname, fno)
                                smooth_pos = MVector3D()
                                smooth_pos.setX(mxfilter(bf.position.x(), fno))
                                smooth_pos.setY(myfilter(bf.position.y(), fno))
                                smooth_pos.setZ(mzfilter(bf.position.z(), fno))

                                bf.position = smooth_pos
                                bf.key = True
                                trace_miku_motion.bones[mname][fno] = bf
                                pchar.update(1)
                            continue

                        for prev_fno, next_fno in zip(fnos[n:], fnos[3+n:]):
                            prev_bf = trace_miku_motion.calc_bf(mname, prev_fno)
                            next_bf = trace_miku_motion.calc_bf(mname, next_fno)
                            for fno in range(prev_fno + 1, next_fno):
                                bf = trace_miku_motion.calc_bf(mname, fno)
                                # まず前後の中間をそのまま求める
                                filterd_qq = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, (fno - prev_fno) / (next_fno - prev_fno))
                                # 現在の回転にも少し近づける
                                bf.rotation = MQuaternion.slerp(filterd_qq, bf.rotation, 0.7)
                                bf.key = True
                                trace_miku_motion.bones[mname][fno] = bf

                            pchar.update(1)

            trace_miku_motion_path = os.path.join(motion_dir_path, f"trace_{process_datetime}_miku_smooth.vmd")
            logger.info("スムージングトレースモーション(あにまさ式ミク)生成開始【{0}】", os.path.basename(trace_miku_motion_path), decoration=MLogger.DECORATION_LINE)
            vmd_writer.write(miku_model, trace_miku_motion, trace_miku_motion_path)

        return True
    except Exception as e:
        logger.critical("モーション生成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False
    
def calc_global_pos_from_mov(model: PmxModel, links: BoneLinks, motion: VmdMotion, fno: int):
    trans_vs = MServiceUtils.calc_relative_position(model, links, motion, fno)
    
    result_pos = MVector3D()
    for v in trans_vs:
        result_pos += v

    return result_pos

def create_bone(trace_model: PmxModel, jname: str, jconn: dict, target_bone_global_vecs: dict, miku_model: PmxModel):
    # MMDボーン名
    mname = jconn["mmd"]
    if mname in trace_model.bones:
        return

    joints = list(target_bone_global_vecs[jname].values())
    parent_joints = list(target_bone_global_vecs[jconn["parent"]].values())
    # 親ボーン
    parent_bone = trace_model.bones[PMX_CONNECTIONS[jconn["parent"]]['mmd']] if jconn["parent"] in PMX_CONNECTIONS else trace_model.bones[jconn["parent"]]

    bone_length = np.median(np.linalg.norm(np.array(joints) - np.array(parent_joints), ord=2, axis=1))

    # 親からの相対位置
    if "指" in jconn["display"]:
        # 指は完全にミクに合わせる
        parent_name = parent_bone.name if "手首元" not in parent_bone.name else parent_bone.name[:3]
        bone_relative_pos = miku_model.bones[mname].position - miku_model.bones[parent_name].position
    else:
        # トレース元から採取
        bone_axis = MVector3D(np.median(np.array(joints), axis=0) - np.median(np.array(parent_joints), axis=0)).normalized()
        bone_relative_pos = MVector3D(bone_axis * bone_length)

    bone_pos = parent_bone.position + bone_relative_pos
    bone = Bone(mname, mname, bone_pos, parent_bone.index, 0, 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010)
    bone.index = len(list(trace_model.bones.keys()))
    if len(jconn['tail']) > 0:
        bone.tail_index = PMX_CONNECTIONS[jconn["tail"]]['mmd']

    # ボーンINDEX
    trace_model.bone_indexes[bone.index] = bone.name
    # ボーン
    trace_model.bones[bone.name] = bone
    # 表示枠
    trace_model.display_slots[jconn["display"]].references.append(trace_model.bones[bone.name].index)
    if jconn["parent"] not in ["全ての親", "センター", "グルーブ"]:
        # 材質
        trace_model.materials[bone.name] = Material(bone.name, bone.name, MVector3D(0, 0, 1), 1, 0, MVector3D(), MVector3D(0.5, 0.5, 0.5), 0x02 | 0x08, MVector4D(0, 0, 0, 1), 1, 0, 0, 0, 0, 0)
        trace_model.materials[bone.name].vertex_count = 12 * 3
        start_vidx = len(trace_model.vertices["vertices"])

        for vidx, (b, v) in enumerate([(bone, bone.position + MVector3D(-0.05, 0, -0.05)), (bone, bone.position + MVector3D(-0.05, 0, 0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(-0.05, 0, 0.05)), (parent_bone, parent_bone.position + MVector3D(-0.05, 0, -0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(-0.05, 0, -0.05)), (parent_bone, parent_bone.position + MVector3D(-0.05, 0, 0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(0.05, 0, 0.05)), (parent_bone, parent_bone.position + MVector3D(0.05, 0, -0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(0.05, 0, -0.05)), (parent_bone, parent_bone.position + MVector3D(0.05, 0, 0.05)), \
                                       (bone, bone.position + MVector3D(0.05, 0, 0.05)), (bone, bone.position + MVector3D(0.05, 0, -0.05)), \
                                       (bone, bone.position + MVector3D(-0.05, 0, 0.05)), (bone, bone.position + MVector3D(-0.05, 0, -0.05)), \
                                       (bone, bone.position + MVector3D(0.05, 0, -0.05)), (bone, bone.position + MVector3D(0.05, 0, 0.05)), \
                                       (bone, bone.position + MVector3D(-0.05, 0, 0.05)), (bone, bone.position + MVector3D(0.05, 0, 0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(0.05, 0, 0.05)), (parent_bone, parent_bone.position + MVector3D(-0.05, 0, 0.05)), \
                                       (bone, bone.position + MVector3D(-0.05, 0, -0.05)), (parent_bone, parent_bone.position + MVector3D(-0.05, 0, -0.05)), \
                                       (parent_bone, parent_bone.position + MVector3D(0.05, 0, -0.05)), (bone, bone.position + MVector3D(0.05, 0, -0.05)), \
                                       ]):
            v1 = Vertex(start_vidx + vidx, v, MVector3D(0, 0, -1), MVector3D(), [], Bdef1(b.index), 1)
            trace_model.vertices["vertices"].append(v1)
        # 面1（上下辺）
        trace_model.indices.append(start_vidx + 0)
        trace_model.indices.append(start_vidx + 1)
        trace_model.indices.append(start_vidx + 2)
        # 面2（上下辺）
        trace_model.indices.append(start_vidx + 2)
        trace_model.indices.append(start_vidx + 3)
        trace_model.indices.append(start_vidx + 0)
        # 面3(縦前)
        trace_model.indices.append(start_vidx + 4)
        trace_model.indices.append(start_vidx + 5)
        trace_model.indices.append(start_vidx + 6)
        # 面4(縦前)
        trace_model.indices.append(start_vidx + 6)
        trace_model.indices.append(start_vidx + 7)
        trace_model.indices.append(start_vidx + 4)
        # 面3(縦左)
        trace_model.indices.append(start_vidx + 8)
        trace_model.indices.append(start_vidx + 9)
        trace_model.indices.append(start_vidx + 10)
        # 面4(縦左)
        trace_model.indices.append(start_vidx + 10)
        trace_model.indices.append(start_vidx + 11)
        trace_model.indices.append(start_vidx + 8)
        # 面5(縦右)
        trace_model.indices.append(start_vidx + 12)
        trace_model.indices.append(start_vidx + 13)
        trace_model.indices.append(start_vidx + 14)
        # 面6(縦右)
        trace_model.indices.append(start_vidx + 14)
        trace_model.indices.append(start_vidx + 15)
        trace_model.indices.append(start_vidx + 12)
        # 面7(縦後)
        trace_model.indices.append(start_vidx + 16)
        trace_model.indices.append(start_vidx + 17)
        trace_model.indices.append(start_vidx + 18)
        # 面8(縦後)
        trace_model.indices.append(start_vidx + 18)
        trace_model.indices.append(start_vidx + 19)
        trace_model.indices.append(start_vidx + 16)
        # 面9(縦後)
        trace_model.indices.append(start_vidx + 20)
        trace_model.indices.append(start_vidx + 21)
        trace_model.indices.append(start_vidx + 22)
        # 面10(縦後)
        trace_model.indices.append(start_vidx + 22)
        trace_model.indices.append(start_vidx + 23)
        trace_model.indices.append(start_vidx + 20)

def create_bone_leg_ik(pmx: PmxModel, direction: str):
    leg_name = f'{direction}足'
    knee_name = f'{direction}ひざ'
    ankle_name = f'{direction}足首'
    toe_name = f'{direction}つま先'
    leg_ik_name = f'{direction}足ＩＫ'
    toe_ik_name = f'{direction}つま先ＩＫ'

    if leg_name in pmx.bones and knee_name in pmx.bones and ankle_name in pmx.bones:
        # 足ＩＫ
        flag = 0x0000 | 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020
        leg_ik_link = []
        leg_ik_link.append(IkLink(pmx.bones[knee_name].index, 1, MVector3D(math.radians(-180), 0, 0), MVector3D(math.radians(-0.5), 0, 0)))
        leg_ik_link.append(IkLink(pmx.bones[leg_name].index, 0))
        leg_ik = Ik(pmx.bones[ankle_name].index, 40, 1, leg_ik_link)
        leg_ik_bone = Bone(leg_ik_name, leg_ik_name, pmx.bones[ankle_name].position, 0, 0, flag, MVector3D(0, 0, 1), -1, ik=leg_ik)
        leg_ik_bone.index = len(pmx.bones)
        pmx.bones[leg_ik_bone.name] = leg_ik_bone
        pmx.bone_indexes[leg_ik_bone.index] = leg_ik_bone.name

        toe_ik_link = []
        toe_ik_link.append(IkLink(pmx.bones[ankle_name].index, 0))
        toe_ik = Ik(pmx.bones[toe_name].index, 40, 1, toe_ik_link)
        toe_ik_bone = Bone(toe_ik_name, toe_ik_name, pmx.bones[ankle_name].position, 0, 0, flag, MVector3D(0, -1, 0), -1, ik=toe_ik)
        toe_ik_bone.index = len(pmx.bones)
        pmx.bones[toe_ik_bone.name] = toe_ik_bone
        pmx.bone_indexes[toe_ik_bone.index] = toe_ik_bone.name

def read_bone_csv(bone_csv_path: str):
    model = PmxModel()
    model.name = os.path.splitext(os.path.basename(bone_csv_path))[0]

    with open(bone_csv_path, "r", encoding=get_file_encoding(bone_csv_path)) as f:
        reader = csv.reader(f)

        # 列名行の代わりにルートボーン登録
        # サイジング用ルートボーン
        sizing_root_bone = Bone("SIZING_ROOT_BONE", "SIZING_ROOT_BONE", MVector3D(), -1, 0, 0)
        sizing_root_bone.index = -999

        model.bones[sizing_root_bone.name] = sizing_root_bone
        # インデックス逆引きも登録
        model.bone_indexes[sizing_root_bone.index] = sizing_root_bone.name

        for ridx, row in enumerate(reader):
            if row[0] == "Bone":
                bone = Bone(row[1], row[2], MVector3D(float(row[5]), float(row[6]), float(row[7])), row[13], int(row[3]), \
                            int(row[14]) * 0x0001| int(row[8]) * 0x0002| int(row[9]) * 0x0004 | int(row[10]) * 0x0020 | int(row[11]) * 0x0008 | int(row[12]) * 0x0010)
                bone.index = ridx - 1

                if len(row[15]) > 0:
                    # 表示先が指定されている場合、設定
                    bone.tail_index = row[15]

                if len(row[37]) > 0:
                    # IKターゲットがある場合、IK登録
                    bone.ik = Ik(model.bones[row[37]].index, int(row[38]), math.radians(float(row[39])))

                model.bones[row[1]] = bone
                model.bone_indexes[bone.index] = row[1]
            elif row[0] == "IKLink":
                iklink = IkLink(model.bones[row[2]].index, int(row[3]), MVector3D(float(row[4]), float(row[6]), float(row[8])), MVector3D(float(row[5]), float(row[7]), float(row[9])))
                model.bones[row[1]].ik.link.append(iklink)
    
    for bidx, bone in model.bones.items():
        # 親ボーンINDEXの設定
        if bone.parent_index and bone.parent_index in model.bones:
            bone.parent_index = model.bones[bone.parent_index].index
        else:
            bone.parent_index = -1
        
        if bone.tail_index and bone.tail_index in model.bones:
            bone.tail_index = model.bones[bone.tail_index].index
        else:
            bone.tail_index = -1

    # 指根元ボーン
    if "左手首" in model.bones:
        left_finger_base_bone = Bone("左指根元", "", (model.bones["左親指０"].position + model.bones["左人指１"].position + model.bones["左中指１"].position + model.bones["左薬指１"].position + model.bones["左小指１"].position) / 5, -1, 0, 0)
        left_finger_base_bone.parent_index = model.bones["首"].index
        left_finger_base_bone.tail_index = model.bones["頭"].index
        left_finger_base_bone.index = len(model.bones.keys())
        model.bones[left_finger_base_bone.name] = left_finger_base_bone
        model.bone_indexes[left_finger_base_bone.index] = left_finger_base_bone.name

    if "右手首" in model.bones:
        right_finger_base_bone = Bone("右指根元", "", (model.bones["右親指０"].position + model.bones["右人指１"].position + model.bones["右中指１"].position + model.bones["右薬指１"].position + model.bones["右小指１"].position) / 5, -1, 0, 0)
        right_finger_base_bone.parent_index = model.bones["首"].index
        right_finger_base_bone.tail_index = model.bones["頭"].index
        right_finger_base_bone.index = len(model.bones.keys())
        model.bones[right_finger_base_bone.name] = right_finger_base_bone
        model.bone_indexes[right_finger_base_bone.index] = right_finger_base_bone.name

    return model


PMX_CONNECTIONS = {
    "pelvis": {"mmd": "下半身", "parent": "グルーブ", "tail": "pelvis2", "display": "体幹", "axis": None},
    "pelvis2": {"mmd": "下半身先", "parent": "pelvis", "tail": "", "display": "体幹", "axis": None},
    "left_hip": {"mmd": "左足", "parent": "pelvis", "tail": "left_knee", "display": "左足", "axis": MVector3D(1, 0, 0)},
    "right_hip": {"mmd": "右足", "parent": "pelvis", "tail": "right_knee", "display": "右足", "axis": MVector3D(-1, 0, 0)},
    "left_knee": {"mmd": "左ひざ", "parent": "left_hip", "tail": "left_ankle", "display": "左足", "axis": MVector3D(1, 0, 0)},
    "right_knee": {"mmd": "右ひざ", "parent": "right_hip", "tail": "right_ankle", "display": "右足", "axis": MVector3D(-1, 0, 0)},
    "left_ankle": {"mmd": "左足首", "parent": "left_knee", "tail": "left_foot_index", "display": "左足", "axis": MVector3D(1, 0, 0)},
    "right_ankle": {"mmd": "右足首", "parent": "right_knee", "tail": "right_foot_index", "display": "右足", "axis": MVector3D(-1, 0, 0)},
    "left_foot_index": {"mmd": "左つま先", "parent": "left_ankle", "tail": "", "display": "左足", "axis": None},
    "right_foot_index": {"mmd": "右つま先", "parent": "right_ankle", "tail": "", "display": "右足", "axis": None},
    "spine1": {"mmd": "上半身", "parent": "グルーブ", "tail": "spine2", "display": "体幹", "axis": None},
    "spine2": {"mmd": "上半身2", "parent": "spine1", "tail": "neck", "display": "体幹", "axis": None},
    "neck": {"mmd": "首", "parent": "spine2", "tail": "head", "display": "体幹", "axis": None},
    "head": {"mmd": "頭", "parent": "neck", "tail": "head_tail", "display": "体幹", "axis": MVector3D(1, 0, 0), "parent_axis": MVector3D(1, 0, 0)},
    "left_collar": {"mmd": "左肩", "parent": "spine2", "tail": "left_shoulder", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_collar": {"mmd": "右肩", "parent": "spine2", "tail": "right_shoulder", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "head_tail": {"mmd": "頭先", "parent": "head", "tail": "", "display": "体幹", "axis": None},
    "left_shoulder": {"mmd": "左腕", "parent": "left_collar", "tail": "left_elbow", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_shoulder": {"mmd": "右腕", "parent": "right_collar", "tail": "right_elbow", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "left_elbow": {"mmd": "左ひじ", "parent": "left_shoulder", "tail": "body_left_wrist", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_elbow": {"mmd": "右ひじ", "parent": "right_shoulder", "tail": "body_right_wrist", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "body_left_wrist": {"mmd": "左手首", "parent": "left_elbow", "tail": "body_left_wrist_tail", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "body_right_wrist": {"mmd": "右手首", "parent": "right_elbow", "tail": "body_right_wrist_tail", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "body_left_wrist_tail": {"mmd": "左手首先", "parent": "body_left_wrist", "tail": "", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "body_right_wrist_tail": {"mmd": "右手首先", "parent": "body_right_wrist", "tail": "", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "nose": {"mmd": "nose", "parent": "head", "tail": "", "display": "顔", "axis": None},
    "right_eye": {"mmd": "右目", "parent": "nose", "tail": "", "display": "顔", "axis": None},
    "left_eye": {"mmd": "左目", "parent": "nose", "tail": "", "display": "顔", "axis": None},
    "right_ear": {"mmd": "right_ear", "parent": "nose", "tail": "", "display": "顔", "axis": None},
    "left_ear": {"mmd": "left_ear", "parent": "nose", "tail": "", "display": "顔", "axis": None},
    "left_heel": {"mmd": "左かかと", "parent": "left_ankle", "tail": "", "display": "左足", "axis": None},
    "right_heel": {"mmd": "右かかと", "parent": "right_ankle", "tail": "", "display": "右足", "axis": None},

    "left_wrist": {"mmd": "左手首元", "parent": "body_left_wrist", "tail": "left_middle1", "display": "左手", "axis": MVector3D(1, 0, 0)},
    "right_wrist": {"mmd": "右手首元", "parent": "body_right_wrist", "tail": "right_middle1", "display": "右手", "axis": MVector3D(-1, 0, 0)},
    "left_index1": {"mmd": "左人指１", "parent": "left_wrist", "tail": "left_index2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_index2": {"mmd": "左人指２", "parent": "left_index1", "tail": "left_index3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_index3": {"mmd": "左人指３", "parent": "left_index2", "tail": "left_index", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_middle1": {"mmd": "左中指１", "parent": "left_wrist", "tail": "left_middle2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_middle2": {"mmd": "左中指２", "parent": "left_middle1", "tail": "left_middle3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_middle3": {"mmd": "左中指３", "parent": "left_middle2", "tail": "left_middle", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_pinky1": {"mmd": "左小指１", "parent": "left_wrist", "tail": "left_pinky2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_pinky2": {"mmd": "左小指２", "parent": "left_pinky1", "tail": "left_pinky3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_pinky3": {"mmd": "左小指３", "parent": "left_pinky2", "tail": "left_pinky", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_ring1": {"mmd": "左薬指１", "parent": "left_wrist", "tail": "left_ring2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_ring2": {"mmd": "左薬指２", "parent": "left_ring1", "tail": "left_ring3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_ring3": {"mmd": "左薬指３", "parent": "left_ring2", "tail": "left_ring", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_thumb1": {"mmd": "左親指０", "parent": "left_wrist", "tail": "left_thumb2", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_thumb2": {"mmd": "左親指１", "parent": "left_thumb1", "tail": "left_thumb3", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_thumb3": {"mmd": "左親指２", "parent": "left_thumb2", "tail": "left_thumb", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "right_index1": {"mmd": "右人指１", "parent": "right_wrist", "tail": "right_index2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_index2": {"mmd": "右人指２", "parent": "right_index1", "tail": "right_index3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_index3": {"mmd": "右人指３", "parent": "right_index2", "tail": "right_index", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_middle1": {"mmd": "右中指１", "parent": "right_wrist", "tail": "right_middle2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_middle2": {"mmd": "右中指２", "parent": "right_middle1", "tail": "right_middle3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_middle3": {"mmd": "右中指３", "parent": "right_middle2", "tail": "right_middle", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_pinky1": {"mmd": "右小指１", "parent": "right_wrist", "tail": "right_pinky2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_pinky2": {"mmd": "右小指２", "parent": "right_pinky1", "tail": "right_pinky3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_pinky3": {"mmd": "右小指３", "parent": "right_pinky2", "tail": "right_pinky", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_ring1": {"mmd": "右薬指１", "parent": "right_wrist", "tail": "right_ring2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_ring2": {"mmd": "右薬指２", "parent": "right_ring1", "tail": "right_ring3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_ring3": {"mmd": "右薬指３", "parent": "right_ring2", "tail": "right_ring", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_thumb1": {"mmd": "右親指０", "parent": "right_wrist", "tail": "right_thumb2", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_thumb2": {"mmd": "右親指１", "parent": "right_thumb1", "tail": "right_thumb3", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_thumb3": {"mmd": "右親指２", "parent": "right_thumb2", "tail": "right_thumb", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "left_thumb": {"mmd": "左親指先", "parent": "left_thumb3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_index": {"mmd": "左人差指先", "parent": "left_index3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_middle": {"mmd": "左中指先", "parent": "left_middle3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_ring": {"mmd": "左薬指先", "parent": "left_ring3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "left_pinky": {"mmd": "左小指先", "parent": "left_pinky3", "tail": "", "display": "左指", "axis": MVector3D(1, 0, 0)},
    "right_thumb": {"mmd": "右親指先", "parent": "right_thumb3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_index": {"mmd": "右人差指先", "parent": "right_index3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_middle": {"mmd": "右中指先", "parent": "right_middle3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_ring": {"mmd": "右薬指先", "parent": "right_ring3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
    "right_pinky": {"mmd": "右小指先", "parent": "right_pinky3", "tail": "", "display": "右指", "axis": MVector3D(-1, 0, 0)},
}



VMD_CONNECTIONS = {
    'pelvis': {'direction': ('pelvis', 'pelvis2'), 'up': ('left_hip', 'right_hip'), 'cancel': []},
    'spine1': {'direction': ('spine1', 'spine2'), 'up': ('left_shoulder', 'right_shoulder'), 'cancel': []},
    'spine2': {'direction': ('spine2', 'neck'), 'up': ('left_shoulder', 'right_shoulder'), 'cancel': ['spine1']},
    'neck': {'direction': ('neck', 'head'), 'up': ('left_shoulder', 'right_shoulder'), 'cancel': ['spine1', 'spine2']},
    'head': {'direction': ('head', 'head_tail'), 'up': ('left_ear', 'right_ear'), 'cancel': ['spine1', 'spine2', 'neck']},

    'left_collar': {'direction': ('left_collar', 'left_shoulder'), 'up': ('spine2', 'neck'), 'cross': ('left_shoulder', 'right_shoulder'), 'cancel': ['spine1', 'spine2']},
    'left_shoulder': {'direction': ('left_shoulder', 'left_elbow'), 'up': ('left_collar', 'left_shoulder'), 'cancel': ['spine1', 'spine2', 'left_collar']},
    'left_elbow': {'direction': ('left_elbow', 'body_left_wrist'), 'up': ('left_shoulder', 'left_elbow'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder']},
    # 'body_left_wrist': {'direction': ('body_left_wrist', 'body_left_wrist_tail'), 'up': ('left_elbow', 'body_left_wrist'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow']},
    'left_hip': {'direction': ('left_hip', 'left_knee'), 'up': ('left_hip', 'right_hip'), 'cancel': ['pelvis']},
    'left_knee': {'direction': ('left_knee', 'left_ankle'), 'up': ('left_hip', 'left_knee'), 'cancel': ['pelvis', 'left_hip']},
    'left_ankle': {'direction': ('left_ankle', 'left_foot_index'), 'up': ('left_knee', 'left_ankle'), 'cancel': ['pelvis', 'left_hip', 'left_knee']},

    'right_collar': {'direction': ('right_collar', 'right_shoulder'), 'up': ('spine2', 'neck'), 'cross': ('right_shoulder', 'left_shoulder'), 'cancel': ['spine1', 'spine2']},
    'right_shoulder': {'direction': ('right_shoulder', 'right_elbow'), 'up': ('right_collar', 'right_shoulder'), 'cancel': ['spine1', 'spine2', 'right_collar']},
    'right_elbow': {'direction': ('right_elbow', 'body_right_wrist'), 'up': ('right_shoulder', 'right_elbow'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder']},
    # 'body_right_wrist': {'direction': ('body_right_wrist', 'body_right_wrist_tail'), 'up': ('right_elbow', 'body_right_wrist'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow']},
    'right_hip': {'direction': ('right_hip', 'right_knee'), 'up': ('right_hip', 'left_hip'), 'cancel': ['pelvis']},
    'right_knee': {'direction': ('right_knee', 'right_ankle'), 'up': ('right_hip', 'right_knee'), 'cancel': ['pelvis', 'right_hip']},
    'right_ankle': {'direction': ('right_ankle', 'right_foot_index'), 'up': ('right_knee', 'right_ankle'), 'cancel': ['pelvis', 'right_hip', 'right_knee']},

    'body_left_wrist': {'direction': ('left_wrist', 'left_middle1'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow']},
    'left_thumb1': {'direction': ('left_thumb1', 'left_thumb2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist']},
    'left_thumb2': {'direction': ('left_thumb2', 'left_thumb3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_thumb1']},
    'left_thumb3': {'direction': ('left_thumb3', 'left_thumb'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_thumb1', 'left_thumb2']},
    'left_index1': {'direction': ('left_index1', 'left_index2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist']},
    'left_index2': {'direction': ('left_index2', 'left_index3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_index1']},
    'left_index3': {'direction': ('left_index3', 'left_index'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_index1', 'left_index2']},
    'left_middle1': {'direction': ('left_middle1', 'left_middle2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist']},
    'left_middle2': {'direction': ('left_middle2', 'left_middle3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_middle1']},
    'left_middle3': {'direction': ('left_middle3', 'left_middle'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_middle1', 'left_middle2']},
    'left_ring1': {'direction': ('left_ring1', 'left_ring2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist']},
    'left_ring2': {'direction': ('left_ring2', 'left_ring3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_ring1']},
    'left_ring3': {'direction': ('left_ring3', 'left_ring'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_ring1', 'left_ring2']},
    'left_pinky1': {'direction': ('left_pinky1', 'left_pinky2'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist']},
    'left_pinky2': {'direction': ('left_pinky2', 'left_pinky3'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_pinky1']},
    'left_pinky3': {'direction': ('left_pinky3', 'left_pinky'), 'up': ('left_index1', 'left_pinky1'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow', 'body_left_wrist', 'left_pinky1', 'left_pinky2']},

    'body_right_wrist': {'direction': ('right_wrist', 'right_middle1'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow']},
    'right_thumb1': {'direction': ('right_thumb1', 'right_thumb2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist']},
    'right_thumb2': {'direction': ('right_thumb2', 'right_thumb3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_thumb1']},
    'right_thumb3': {'direction': ('right_thumb3', 'right_thumb'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_thumb1', 'right_thumb2']},
    'right_index1': {'direction': ('right_index1', 'right_index2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist']},
    'right_index2': {'direction': ('right_index2', 'right_index3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_index1']},
    'right_index3': {'direction': ('right_index3', 'right_index'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_index1', 'right_index2']},
    'right_middle1': {'direction': ('right_middle1', 'right_middle2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist']},
    'right_middle2': {'direction': ('right_middle2', 'right_middle3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_middle1']},
    'right_middle3': {'direction': ('right_middle3', 'right_middle'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_middle1', 'right_middle2']},
    'right_ring1': {'direction': ('right_ring1', 'right_ring2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist']},
    'right_ring2': {'direction': ('right_ring2', 'right_ring3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_ring1']},
    'right_ring3': {'direction': ('right_ring3', 'right_ring'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_ring1', 'right_ring2']},
    'right_pinky1': {'direction': ('right_pinky1', 'right_pinky2'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist']},
    'right_pinky2': {'direction': ('right_pinky2', 'right_pinky3'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_pinky1']},
    'right_pinky3': {'direction': ('right_pinky3', 'right_pinky'), 'up': ('right_index1', 'right_pinky1'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow', 'body_right_wrist', 'right_pinky1', 'right_pinky2']},
}
