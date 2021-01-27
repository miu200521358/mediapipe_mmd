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

from mmd.utils.MBezierUtils import join_value_2_bezier, R_x1_idxs, R_y1_idxs, R_x2_idxs, R_y2_idxs, MX_x1_idxs, MX_y1_idxs, MX_x2_idxs, MX_y2_idxs
from mmd.utils.MBezierUtils import MY_x1_idxs, MY_y1_idxs, MY_x2_idxs, MY_y2_idxs, MZ_x1_idxs, MZ_y1_idxs, MZ_x2_idxs, MZ_y2_idxs
from mmd.mmd.VmdWriter import VmdWriter
from mmd.module.MMath import MQuaternion, MVector3D, MVector2D, MMatrix4x4, MRect, fromEulerAngles
from mmd.mmd.VmdData import VmdBoneFrame, VmdMorphFrame, VmdMotion, VmdShowIkFrame, VmdInfoIk, OneEuroFilter
from mmd.mmd.PmxData import PmxModel, Bone, Vertex, Bdef1, Ik, IkLink
from mmd.utils.MServiceUtils import get_file_encoding, calc_global_pos, separate_local_qq

logger = MLogger(__name__, level=1)

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

        motion_dir_path = os.path.join(args.img_dir, "motion")
        os.makedirs(motion_dir_path, exist_ok=True)
        
        # モデルをCSVから読み込む
        model = read_bone_csv(args.bone_config)
        process_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        joints_pattern = re.compile(r'^smooth_(\d+)\.')

        joints_json_pathes = sorted(glob.glob(os.path.join(args.img_dir, "smooth", "*.json")), key=sort_by_numeric)

        motion = VmdMotion()
        all_frame_joints = {}
        target_bone_names = {}

        logger.info("FKボーン角度計算開始", decoration=MLogger.DECORATION_LINE)
        for sidx, joints_json_path in enumerate(tqdm(joints_json_pathes)):
            m = joints_pattern.match(os.path.basename(joints_json_path))
            if m:
                # キーフレの場所を確定（間が空く場合もある）
                fno = int(m.groups()[0])

                frame_joints = {}
                with open(joints_json_path, 'r', encoding='utf-8') as f:
                    frame_joints = json.load(f)
                all_frame_joints[fno] = frame_joints

                for jname, (bone_name, name_list, parent_list, initial_qq, ranges, diff_limits, is_hand, is_head) in VMD_CONNECTIONS.items():
                    if name_list is None:
                        continue

                    bf = VmdBoneFrame(fno)
                    bf.set_name(bone_name)
                    
                    if len(name_list) == 4:
                        rotation = calc_direction_qq(bf.fno, motion, frame_joints, *name_list)
                        initial = calc_bone_direction_qq(bf, motion, model, jname, *name_list)
                    else:
                        rotation = calc_direction_qq2(bf.fno, motion, frame_joints, *name_list)
                        initial = calc_bone_direction_qq2(bf, motion, model, jname, *name_list)

                    qq = MQuaternion()
                    for parent_name in reversed(parent_list):
                        qq *= motion.calc_bf(parent_name, bf.fno).rotation.inverted()
                    bf.rotation = qq * initial_qq * rotation * initial.inverted()

                    motion.regist_bf(bf, bf.name, bf.fno)
                    target_bone_names[bf.name] = diff_limits

        start_fno = sorted(all_frame_joints.keys())[0]
        last_fno = sorted(all_frame_joints.keys())[-1]
        fnos = list(range(start_fno, last_fno + 1))

        logger.info("FKボーン角度チェック開始", decoration=MLogger.DECORATION_LINE)

        with tqdm(total=(len(list(all_frame_joints.keys())) * len(list(VMD_CONNECTIONS.keys())))) as pchar:
            for fidx, (fno, frame_joints) in enumerate(all_frame_joints.items()):
                for jname, (bone_name, name_list, parent_list, initial_qq, ranges, diff_limits, is_hand, is_head) in VMD_CONNECTIONS.items():
                    pchar.update(1)

                    if bone_name not in "頭":
                        continue

                    bf = motion.calc_bf(bone_name, fno)

                    if ranges:
                        # 可動域指定がある場合、制限する
                        local_x_axis = model.get_local_x_axis(bf.name)
                        x_qq, y_qq, z_qq, _ = separate_local_qq(bf.fno, bf.name, bf.rotation, local_x_axis)
                        local_z_axis = MVector3D(0, 0, (-1 if "right" in jname else 1))
                        local_y_axis = MVector3D.crossProduct(local_x_axis, local_z_axis)
                        x_limited_qq = MQuaternion.fromAxisAndAngle(local_x_axis, max(ranges["x"]["min"], min(ranges["x"]["max"], x_qq.toDegree() * MVector3D.dotProduct(local_x_axis, x_qq.vector()))))
                        y_limited_qq = MQuaternion.fromAxisAndAngle(local_y_axis, max(ranges["y"]["min"], min(ranges["y"]["max"], y_qq.toDegree() * MVector3D.dotProduct(local_y_axis, y_qq.vector()))))
                        z_limited_qq = MQuaternion.fromAxisAndAngle(local_z_axis, max(ranges["z"]["min"], min(ranges["z"]["max"], z_qq.toDegree() * MVector3D.dotProduct(local_z_axis, z_qq.vector()))))
                        bf.rotation = y_limited_qq * x_limited_qq * z_limited_qq

                        motion.regist_bf(bf, bf.name, bf.fno, is_key=bf.key)

        logger.info("スムージング開始", decoration=MLogger.DECORATION_LINE)

        with tqdm(total=(len(list(target_bone_names.keys())) * len(fnos))) as pchar:
            for bone_name in target_bone_names.keys():
                rxfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
                ryfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)
                rzfilter = OneEuroFilter(freq=30, mincutoff=1, beta=0.000000000000001, dcutoff=1)

                for fidx, fno in enumerate(fnos):
                    pchar.update(1)
                    now_bf = motion.calc_bf(bone_name, fno)

                    euler = now_bf.rotation.toEulerAngles()
                    xv = euler.x()
                    yv = euler.y()
                    zv = euler.z()

                    prev_fno, next_fno = motion.get_bone_prev_next_fno(bone_name, fno=fno, is_key=True)
                    prev_bf = motion.calc_bf(bone_name, prev_fno)
                    next_bf = motion.calc_bf(bone_name, next_fno)

                    if fno not in all_frame_joints:
                        # キーフレがないフレームの場合、前後の線形補間
                        if fidx == 0:
                            xv = 0
                            yv = 0
                            zv = 0
                        else:
                            now_rot = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ((fno - prev_fno) / (next_fno - prev_fno)))
                            now_euler = now_rot.toEulerAngles()
                            xv = now_euler.x()
                            yv = now_euler.y()
                            zv = now_euler.z()

                    if fidx > 0:
                        # 前のキーフレから大きく変化しすぎてる場合、前後の線形補間をコピーしてスルー
                        dot = MQuaternion.dotProduct(now_bf.rotation, prev_bf.rotation)
                        if dot < 1 - ((now_bf.fno - prev_bf.fno) * (0.2 if bone_name in ["上半身", "下半身"] else 0.1)):
                            now_rot = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ((fno - prev_fno) / (next_fno - prev_fno)))
                            now_euler = now_rot.toEulerAngles()
                            xv = now_euler.x()
                            yv = now_euler.y()
                            zv = now_euler.z()

                    xv = rxfilter(xv, fno)
                    yv = ryfilter(yv, fno)
                    zv = rzfilter(zv, fno)

                    now_bf.rotation = MQuaternion.fromEulerAngles(xv, yv, zv)
                    motion.regist_bf(now_bf, now_bf.name, now_bf.fno, is_key=now_bf.key)

        logger.info("モーション生成開始", decoration=MLogger.DECORATION_LINE)
        motion_path = os.path.join(motion_dir_path, "output_{0}.vmd".format(process_datetime))
        writer = VmdWriter(model, motion, motion_path)
        writer.write()

        logger.info("モーション生成終了: {0}", motion_path, decoration=MLogger.DECORATION_BOX)

        logger.info('モーション生成処理全件終了', decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("モーション生成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False

def calc_direction_qq(bf: VmdBoneFrame, motion: VmdMotion, joints: dict, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str):
    direction_from_vec = get_vec3(joints["joints"], direction_from_name)
    direction_to_vec = get_vec3(joints["joints"], direction_to_name)
    up_from_vec = get_vec3(joints["joints"], up_from_name)
    up_to_vec = get_vec3(joints["joints"], up_to_name)

    direction = (direction_to_vec - direction_from_vec).normalized()
    up = (up_to_vec - up_from_vec).normalized()
    cross = MVector3D.crossProduct(direction, up)
    qq = MQuaternion.fromDirection(direction, cross)

    return qq

def calc_direction_qq2(bf: VmdBoneFrame, motion: VmdMotion, joints: dict, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str, cross_from_name: str, cross_to_name: str):
    direction_from_vec = get_vec3(joints["joints"], direction_from_name)
    direction_to_vec = get_vec3(joints["joints"], direction_to_name)
    up_from_vec = get_vec3(joints["joints"], up_from_name)
    up_to_vec = get_vec3(joints["joints"], up_to_name)
    cross_from_vec = get_vec3(joints["joints"], cross_from_name)
    cross_to_vec = get_vec3(joints["joints"], cross_to_name)

    direction = (direction_to_vec - direction_from_vec).normalized()
    up = (up_to_vec - up_from_vec).normalized()
    cross = (cross_to_vec - cross_from_vec).normalized()
    qq = MQuaternion.fromDirection(direction, MVector3D.crossProduct(up, cross))

    return qq

def calc_bone_direction_qq(bf: VmdBoneFrame, motion: VmdMotion, model: PmxModel, jname: str, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str):
    direction_from_vec = get_bone_vec3(model, direction_from_name)
    direction_to_vec = get_bone_vec3(model, direction_to_name)
    up_from_vec = get_bone_vec3(model, up_from_name)
    up_to_vec = get_bone_vec3(model, up_to_name)

    direction = (direction_to_vec - direction_from_vec).normalized()
    up = (up_to_vec - up_from_vec).normalized()
    qq = MQuaternion.fromDirection(direction, up)

    return qq

def calc_bone_direction_qq2(bf: VmdBoneFrame, motion: VmdMotion, model: PmxModel, jname: str, direction_from_name: str, direction_to_name: str, up_from_name: str, up_to_name: str, cross_from_name: str, cross_to_name: str):
    direction_from_vec = get_bone_vec3(model, direction_from_name)
    direction_to_vec = get_bone_vec3(model, direction_to_name)
    up_from_vec = get_bone_vec3(model, up_from_name)
    up_to_vec = get_bone_vec3(model, up_to_name)
    cross_from_vec = get_bone_vec3(model, cross_from_name)
    cross_to_vec = get_bone_vec3(model, cross_to_name)

    direction = (direction_to_vec - direction_from_vec).normalized()
    up = (up_to_vec - up_from_vec).normalized()
    cross = (cross_to_vec - cross_from_vec).normalized()
    qq = MQuaternion.fromDirection(direction, MVector3D.crossProduct(up, cross))

    return qq

def get_bone_vec3(model: PmxModel, joint_name: str):
    bone_name, _, _, _, _, _, _, _ = VMD_CONNECTIONS[joint_name]
    if bone_name in model.bones:
        return model.bones[bone_name].position
    
    return MVector3D()

def get_vec3(joints: dict, jname: str):
    if jname in joints:
        joint = joints[jname]
        return MVector3D(joint["x"], joint["y"], joint["z"])
    else:
        if jname == "spine1":
            # 腰くらい
            right_hip_vec = get_vec3(joints, "right_hip")
            left_hip_vec = get_vec3(joints, "left_hip")
            return (right_hip_vec + left_hip_vec) / 2
        elif jname == "spine3":
            # 首根元くらい
            right_arm_vec = get_vec3(joints, "right_arm")
            left_arm_vec = get_vec3(joints, "left_arm")
            return (right_arm_vec + left_arm_vec) / 2
        elif jname == "body_left_middle":
            # 中指くらい
            left_index_vec = get_vec3(joints, "body_left_index")
            left_pinky_vec = get_vec3(joints, "body_left_pinky")
            return (left_index_vec + left_pinky_vec) / 2
        elif jname == "body_right_middle":
            # 中指くらい
            right_index_vec = get_vec3(joints, "body_right_index")
            right_pinky_vec = get_vec3(joints, "body_right_pinky")
            return (right_index_vec + right_pinky_vec) / 2
    
    return MVector3D()


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
                            int(row[8]) * 0x0002| int(row[9]) * 0x0004 | int(row[10]) * 0x0020 | int(row[11]) * 0x0008 | int(row[12]) * 0x0010)
                bone.index = ridx - 1

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

    # 首根元ボーン
    if "左肩" in model.bones and "右肩" in model.bones:
        neck_base_vertex = Vertex(-1, (model.bones["左肩"].position + model.bones["右肩"].position) / 2 + MVector3D(0, -0.1, 0), MVector3D(), [], [], Bdef1(-1), -1)
        neck_base_vertex.position.setX(0)
        neck_base_bone = Bone("首根元", "base of neck", neck_base_vertex.position.copy(), -1, 0, 0)

        if "上半身2" in model.bones:
            # 上半身2がある場合、表示先は、上半身2
            neck_base_bone.parent_index = model.bones["上半身2"].index
            neck_base_bone.tail_index = model.bones["上半身2"].index
        elif "上半身" in model.bones:
            neck_base_bone.parent_index = model.bones["上半身"].index
            neck_base_bone.tail_index = model.bones["上半身"].index

        neck_base_bone.index = len(model.bones.keys())
        model.bones[neck_base_bone.name] = neck_base_bone
        model.bone_indexes[neck_base_bone.index] = neck_base_bone.name

    # 鼻ボーン
    if "頭" in model.bones and "首" in model.bones:
        nose_bone = Bone("鼻", "nose", MVector3D(0, model.bones["頭"].position.y(), model.bones["頭"].position.z() - 0.5), -1, 0, 0)
        nose_bone.parent_index = model.bones["首"].index
        nose_bone.tail_index = model.bones["頭"].index
        nose_bone.index = len(model.bones.keys())
        model.bones[nose_bone.name] = nose_bone
        model.bone_indexes[nose_bone.index] = nose_bone.name

    # 頭頂ボーン
    if "頭" in model.bones:
        head_top_bone = Bone("頭頂", "top of head", MVector3D(0, model.bones["頭"].position.y() + 1, 0), -1, 0, 0)
        head_top_bone.parent_index = model.bones["鼻"].index
        head_top_bone.index = len(model.bones.keys())
        model.bones[head_top_bone.name] = head_top_bone
        model.bone_indexes[head_top_bone.index] = head_top_bone.name

    if "右足首" in model.bones:
        right_heel_bone = Bone("右かかと", "", MVector3D(model.bones["右足首"].position.x(), 0, model.bones["右足首"].position.z()), -1, 0, 0)
        right_heel_bone.parent_index = model.bones["右つま先"].index
        right_heel_bone.index = len(model.bones.keys())
        model.bones[right_heel_bone.name] = right_heel_bone
        model.bone_indexes[right_heel_bone.index] = right_heel_bone.name

    if "左足首" in model.bones:
        left_heel_bone = Bone("左かかと", "", MVector3D(model.bones["左足首"].position.x(), 0, model.bones["左足首"].position.z()), -1, 0, 0)
        left_heel_bone.parent_index = model.bones["左つま先"].index
        left_heel_bone.index = len(model.bones.keys())
        model.bones[left_heel_bone.name] = left_heel_bone
        model.bone_indexes[left_heel_bone.index] = left_heel_bone.name

    if "右つま先" in model.bones:
        right_big_toe_bone = Bone("右足親指", "", model.bones["右つま先"].position + MVector3D(0.5, 0, 0), -1, 0, 0)
        right_big_toe_bone.parent_index = model.bones["右つま先"].index
        right_big_toe_bone.index = len(model.bones.keys())
        model.bones[right_big_toe_bone.name] = right_big_toe_bone
        model.bone_indexes[right_big_toe_bone.index] = right_big_toe_bone.name

        right_small_toe_bone = Bone("右足小指", "", model.bones["右つま先"].position + MVector3D(-0.5, 0, 0), -1, 0, 0)
        right_small_toe_bone.parent_index = model.bones["右つま先"].index
        right_small_toe_bone.index = len(model.bones.keys())
        model.bones[right_small_toe_bone.name] = right_small_toe_bone
        model.bone_indexes[right_small_toe_bone.index] = right_small_toe_bone.name

    if "左つま先" in model.bones:
        left_big_toe_bone = Bone("左足親指", "", model.bones["左つま先"].position + MVector3D(-0.5, 0, 0), -1, 0, 0)
        left_big_toe_bone.parent_index = model.bones["左つま先"].index
        left_big_toe_bone.index = len(model.bones.keys())
        model.bones[left_big_toe_bone.name] = left_big_toe_bone
        model.bone_indexes[left_big_toe_bone.index] = left_big_toe_bone.name

        left_small_toe_bone = Bone("左足小指", "", model.bones["左つま先"].position + MVector3D(0.5, 0, 0), -1, 0, 0)
        left_small_toe_bone.parent_index = model.bones["左つま先"].index
        left_small_toe_bone.index = len(model.bones.keys())
        model.bones[left_small_toe_bone.name] = left_small_toe_bone
        model.bone_indexes[left_small_toe_bone.index] = left_small_toe_bone.name

    return model


VMD_CONNECTIONS = {
    'right_eye': ("右目", None, None, MQuaternion(), None, None, False, False),
    'left_eye': ("左目", None, None, MQuaternion(), None, None, False, False),
    'spine3': ("首根元", None, None, MQuaternion(), None, None, False, False),
    'spine1': ("上半身", ['spine3', 'spine1', 'left_arm', 'right_arm', 'spine3', 'spine1'], [], MQuaternion.fromEulerAngles(20, 0, 0), None, {"rot": 0.0005, "mov": 0, "sub": True}, False, False),
    'head': ("頭", ['head', 'spine3', 'left_eye', 'right_eye', 'head', 'spine3'], ["上半身"], MQuaternion.fromEulerAngles(-20, 0, 0), \
        {"x": {"min": -30, "max": 40}, "y": {"min": -50, "max": 50}, "z": {"min": -30, "max": 30}}, {"rot": 0.001, "mov": 0, "sub": True}, False, True),
    'right_shoulder': ("右肩", ['spine3', 'right_arm', 'spine1', 'spine3', 'spine3', 'right_arm'], ["上半身"], MQuaternion(), None, {"rot": 0.001, "mov": 0, "sub": True}, False, False),
    'left_shoulder': ("左肩", ['spine3', 'left_arm', 'spine1', 'spine3', 'spine3', 'left_arm'], ["上半身"], MQuaternion(), None, {"rot": 0.001, "mov": 0, "sub": True}, False, False),
    'right_arm': ("右腕", ['right_arm', 'right_elbow', 'spine3', 'right_arm', 'right_arm', 'right_elbow'], ["上半身", "右肩"], MQuaternion(), None, {"rot": 0.001, "mov": 0, "sub": True}, False, False),
    'left_arm': ("左腕", ['left_arm', 'left_elbow', 'spine3', 'left_arm', 'left_arm', 'left_elbow'], ["上半身", "左肩"], MQuaternion(), None, {"rot": 0.001, "mov": 0, "sub": True}, False, False),
    'right_elbow': ("右ひじ", ['right_elbow', 'right_wrist', 'right_arm', 'right_elbow', 'right_elbow', 'right_wrist'], ["上半身", "右肩", "右腕"], MQuaternion(), None, {"rot": 0.001, "mov": 0, "sub": True}, False, False),
    'left_elbow': ("左ひじ", ['left_elbow', 'left_wrist', 'left_arm', 'left_elbow', 'left_elbow', 'left_wrist'], ["上半身", "左肩", "左腕"], MQuaternion(), None, {"rot": 0.001, "mov": 0, "sub": True}, False, False),
    'right_wrist': ("右手首", ['right_wrist', 'right_middle1', 'right_index1', 'right_pinky1', 'right_wrist', 'right_middle1'], ["上半身", "右肩", "右腕", "右ひじ"], MQuaternion(), None, None, False, False),
    'left_wrist': ("左手首", ['left_wrist', 'left_middle1', 'left_index1', 'left_pinky1', 'left_wrist', 'left_middle1'], ["上半身", "左肩", "左腕", "左ひじ"], MQuaternion(), None, None, False, False),
    'right_thumb1': ("右親指０", ['right_thumb1', 'right_thumb2', 'right_index1', 'right_pinky1', 'right_thumb1', 'right_thumb2'], ["右肩", "右腕", "右ひじ", "右手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_thumb1': ("左親指０", ['left_thumb1', 'left_thumb2', 'left_index1', 'left_pinky1', 'left_thumb1', 'left_thumb2'], ["左肩", "左腕", "左ひじ", "左手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_thumb2': ("右親指１", ['right_thumb2', 'right_thumb3', 'right_index1', 'right_pinky1', 'right_thumb2', 'right_thumb3'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右親指０"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_thumb2': ("左親指１", ['left_thumb2', 'left_thumb3', 'left_index1', 'left_pinky1', 'left_thumb2', 'left_thumb3'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左親指０"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_thumb3': ("右親指２", ['right_thumb3', 'right_thumb4', 'right_index1', 'right_pinky1', 'right_thumb3', 'right_thumb4'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右親指０", "右親指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_thumb3': ("左親指２", ['left_thumb3', 'left_thumb4', 'left_index1', 'left_pinky1', 'left_thumb3', 'left_thumb4'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左親指０", "左親指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_thumb4': ("右親差指先", None, None, MQuaternion(), None, None, True, False),
    'left_thumb4': ("左親差指先", None, None, MQuaternion(), None, None, True, False),
    'right_index1': ("右人指１", ['right_index1', 'right_index2', 'right_index1', 'right_pinky1', 'right_index1', 'right_index2'], ["上半身", "右肩", "右腕", "右ひじ", "右手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_index1': ("左人指１", ['left_index1', 'left_index2', 'left_index1', 'left_pinky1', 'left_index1', 'left_index2'], ["上半身", "左肩", "左腕", "左ひじ", "左手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_index2': ("右人指２", ['right_index2', 'right_index3', 'right_index1', 'right_pinky1', 'right_index2', 'right_index3'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右人指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_index2': ("左人指２", ['left_index2', 'left_index3', 'left_index1', 'left_pinky1', 'left_index2', 'left_index3'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左人指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_index3': ("右人指３", ['right_index3', 'right_index4', 'right_index1', 'right_pinky1', 'right_index3', 'right_index4'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右人指１", "右人指２"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_index3': ("左人指３", ['left_index3', 'left_index4', 'left_index1', 'left_pinky1', 'left_index3', 'left_index4'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左人指１", "左人指２"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_index4': ("右人差指先", None, None, MQuaternion(), None, None, True, False),
    'left_index4': ("左人差指先", None, None, MQuaternion(), None, None, True, False),
    'right_middle1': ("右中指１", ['right_middle1', 'right_middle2', 'right_index1', 'right_pinky1', 'right_middle1', 'right_middle2'], ["上半身", "右肩", "右腕", "右ひじ", "右手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_middle1': ("左中指１", ['left_middle1', 'left_middle2', 'left_index1', 'left_pinky1', 'left_middle1', 'left_middle2'], ["上半身", "左肩", "左腕", "左ひじ", "左手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_middle2': ("右中指２", ['right_middle2', 'right_middle3', 'right_index1', 'right_pinky1', 'right_middle2', 'right_middle3'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右中指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_middle2': ("左中指２", ['left_middle2', 'left_middle3', 'left_index1', 'left_pinky1', 'left_middle2', 'left_middle3'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左中指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_middle3': ("右中指３", ['right_middle3', 'right_middle4', 'right_index1', 'right_pinky1', 'right_middle3', 'right_middle4'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右中指１", "右中指２"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_middle3': ("左中指３", ['left_middle3', 'left_middle4', 'left_index1', 'left_pinky1', 'left_middle3', 'left_middle4'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左中指１", "左中指２"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_middle4': ("右中指先", None, None, MQuaternion(), None, None, True, False),
    'left_middle4': ("左中指先", None, None, MQuaternion(), None, None, True, False),
    'right_ring1': ("右薬指１", ['right_ring1', 'right_ring2', 'right_index1', 'right_pinky1', 'right_ring1', 'right_ring2'], ["上半身", "右肩", "右腕", "右ひじ", "右手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_ring1': ("左薬指１", ['left_ring1', 'left_ring2', 'left_index1', 'left_pinky1', 'left_ring1', 'left_ring2'], ["上半身", "左肩", "左腕", "左ひじ", "左手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_ring2': ("右薬指２", ['right_ring2', 'right_ring3', 'right_index1', 'right_pinky1', 'right_ring2', 'right_ring3'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右薬指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_ring2': ("左薬指２", ['left_ring2', 'left_ring3', 'left_index1', 'left_pinky1', 'left_ring2', 'left_ring3'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左薬指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_ring3': ("右薬指３", ['right_ring3', 'right_ring4', 'right_index1', 'right_pinky1', 'right_ring3', 'right_ring4'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右薬指１", "右薬指２"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_ring3': ("左薬指３", ['left_ring3', 'left_ring4', 'left_index1', 'left_pinky1', 'left_ring3', 'left_ring4'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左薬指１", "左薬指２"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_ring4': ("右薬指先", None, None, MQuaternion(), None, None, True, False),
    'left_ring4': ("左薬指先", None, None, MQuaternion(), None, None, True, False),
    'right_pinky1': ("右小指１", ['right_pinky1', 'right_pinky2', 'right_index1', 'right_pinky1', 'right_pinky1', 'right_pinky2'], ["上半身", "右肩", "右腕", "右ひじ", "右手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_pinky1': ("左小指１", ['left_pinky1', 'left_pinky2', 'left_index1', 'left_pinky1', 'left_pinky1', 'left_pinky2'], ["上半身", "左肩", "左腕", "左ひじ", "左手首"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_pinky2': ("右小指２", ['right_pinky2', 'right_pinky3', 'right_index1', 'right_pinky1', 'right_pinky2', 'right_pinky3'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右小指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_pinky2': ("左小指２", ['left_pinky2', 'left_pinky3', 'left_index1', 'left_pinky1', 'left_pinky2', 'left_pinky3'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左小指１"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_pinky3': ("右小指３", ['right_pinky3', 'right_pinky4', 'right_index1', 'right_pinky1', 'right_pinky3', 'right_pinky4'], ["上半身", "右肩", "右腕", "右ひじ", "右手首", "右小指１", "右小指２"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'left_pinky3': ("左小指３", ['left_pinky3', 'left_pinky4', 'left_index1', 'left_pinky1', 'left_pinky3', 'left_pinky4'], ["上半身", "左肩", "左腕", "左ひじ", "左手首", "左小指１", "左小指２"], MQuaternion(), None, {"rot": 0.002, "mov": 0, "sub": False}, True, False),
    'right_pinky4': ("右小指先", None, None, MQuaternion(), None, None, True, False),
    'left_pinky4': ("左小指先", None, None, MQuaternion(), None, None, True, False),
}
