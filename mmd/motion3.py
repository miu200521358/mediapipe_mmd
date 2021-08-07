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

        for mmname in ["全ての親", "センター", "グルーブ"]:
            target_bone_global_vecs[mmname] = {}
        
        logger.info("モーション結果位置計算開始", decoration=MLogger.DECORATION_LINE)

        with tqdm(total=(len(smooth_json_pathes) * len(PMX_CONNECTIONS.keys()))) as pchar:
            for sidx, smooth_json_path in enumerate(smooth_json_pathes):
                m = smooth_pattern.match(os.path.basename(smooth_json_path))
                if m:
                    # キーフレの場所を確定（間が空く場合もある）
                    fno = int(m.groups()[0])

                    frame_joints = {}
                    with open(smooth_json_path, 'r', encoding='utf-8') as f:
                        frame_joints = json.load(f)
                    
                    if np.all(start_heel == 9999999):
                        # 最初のpelvis2を原点として登録する(奥行きのみ)
                        z = frame_joints["joints"]['right_heel']["wz"] if frame_joints["joints"]['right_heel']["wy"] > frame_joints["joints"]['left_heel']["wy"] else frame_joints["joints"]['right_heel']["wz"]
                        start_heel = np.array([0, 0, z]) * MIKU_METER
                        image_size = MVector3D(frame_joints["image"]["width"], frame_joints["image"]["height"], frame_joints["image"]["width"])

                    if "joints" in frame_joints:
                        for jname, joint in frame_joints["joints"].items():
                            if jname not in target_bone_global_vecs:
                                target_bone_global_vecs[jname] = {}
                            target_bone_global_vecs[jname][fno] = (np.array([-joint["wx"], -joint["wy"], joint["wz"]]) * MIKU_METER) - start_heel
                        
                        if 'pelvis2' in frame_joints["joints"]:
                            fnos.append(fno)
                            target_bone_global_vecs["全ての親"][fno] = np.array([0, 0, 0])
                            z = frame_joints["joints"]['right_heel']["wz"] if frame_joints["joints"]['right_heel']["wy"] > frame_joints["joints"]['left_heel']["wy"] else frame_joints["joints"]['right_heel']["wz"]
                            target_bone_global_vecs["センター"][fno] = np.array([(image_size.x() / 2 - (frame_joints["joints"]['pelvis2']["x"] * image_size.x())), frame_joints["joints"]['pelvis2']["y"] * image_size.y(), z * MIKU_METER]) - start_heel
                            target_bone_global_vecs["グルーブ"][fno] = np.array([0, 0, 0])
                        else:
                            pchar.update(len(PMX_CONNECTIONS.keys()))

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
            
            logger.info("モーション(移動)計算開始", decoration=MLogger.DECORATION_LINE)

            with tqdm(total=(len(PMX_CONNECTIONS.keys()) * (len(fnos)))) as pchar:
                for jidx, (jname, pconn) in enumerate(PMX_CONNECTIONS.items()):
                    if jname not in target_bone_global_vecs:
                        pchar.update(1)
                        continue

                    # ボーン登録
                    create_bone(trace_model, jname, pconn, target_bone_global_vecs)

                    mname = pconn['mmd']
                    pmname = PMX_CONNECTIONS[pconn['parent']]['mmd'] if pconn['parent'] in PMX_CONNECTIONS else pconn['parent']

                    # モーションも登録
                    for fno in fnos:
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

                    # トレースモデルの初期姿勢
                    trace_direction_from_vec = trace_model.bones[direction_from_mname].position
                    trace_direction_to_vec = trace_model.bones[direction_to_mname].position
                    trace_direction = (trace_direction_to_vec - trace_direction_from_vec).normalized()

                    trace_up_from_vec = trace_model.bones[up_from_mname].position
                    trace_up_to_vec = trace_model.bones[up_to_mname].position
                    trace_up = (trace_up_to_vec - trace_up_from_vec).normalized()

                    trace_direction_up = MVector3D.crossProduct(trace_direction, trace_up).normalized()
                    trace_stance_qq = MQuaternion.fromDirection(trace_direction, trace_direction_up)

                    trace_local_x_axis = trace_model.get_local_x_axis(mname)

                    direction_from_links = trace_model.create_link_2_top_one(direction_from_mname, is_defined=False)
                    direction_to_links = trace_model.create_link_2_top_one(direction_to_mname, is_defined=False)
                    up_from_links = trace_model.create_link_2_top_one(up_from_mname, is_defined=False)
                    up_to_links = trace_model.create_link_2_top_one(up_to_mname, is_defined=False)

                    for fno in fnos:
                        now_direction_from_vec = calc_global_pos_from_mov(trace_model, direction_from_links, trace_mov_motion, fno)
                        now_direction_to_vec = calc_global_pos_from_mov(trace_model, direction_to_links, trace_mov_motion, fno)
                        now_up_from_vec = calc_global_pos_from_mov(trace_model, up_from_links, trace_mov_motion, fno)
                        now_up_to_vec = calc_global_pos_from_mov(trace_model, up_to_links, trace_mov_motion, fno)

                        # トレースモデルの回転量 ------------
                        now_direction = (now_direction_to_vec - now_direction_from_vec).normalized()
                        now_up = (now_up_to_vec - now_up_from_vec).normalized()

                        now_direction_up = MVector3D.crossProduct(now_direction, now_up).normalized()
                        now_direction_qq = MQuaternion.fromDirection(now_direction, now_direction_up)

                        cancel_qq = MQuaternion()
                        for cancel_jname in jconn["cancel"]:
                            cancel_qq *= trace_rot_motion.calc_bf(PMX_CONNECTIONS[cancel_jname]['mmd'], fno).rotation

                        now_qq = cancel_qq.inverted() * now_direction_qq * trace_stance_qq.inverted()

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
                for mname in trace_rot_motion.bones:
                    parent_name = "センター"
                    if mname not in ["全ての親", "センター", "グルーブ"]:
                        parent_name = trace_model.bone_indexes[trace_model.bones[mname].parent_index]

                    parent_local_x_axis = MVector3D(1, 0, 0) if parent_name in ["左腕", "左ひじ", "左手首"] else MVector3D(-1, 0, 0) if parent_name in ["右腕", "右ひじ", "右手首"] else None
                    target_local_x_axis = MVector3D(1, 0, 0) if mname in ["左腕", "左ひじ", "左手首"] else MVector3D(-1, 0, 0) if mname in ["右腕", "右ひじ", "右手首"] else None
                    trace_parent_local_x_qq = trace_model.get_local_x_qq(parent_name, parent_local_x_axis)
                    trace_target_local_x_qq = trace_model.get_local_x_qq(mname, target_local_x_axis) if mname not in ["全ての親", "センター", "グルーブ"] else MQuaternion()
                    miku_parent_local_x_qq = miku_model.get_local_x_qq(parent_name, parent_local_x_axis)
                    miku_target_local_x_qq = miku_model.get_local_x_qq(mname, target_local_x_axis) if mname not in ["全ての親", "センター", "グルーブ"] else MQuaternion()

                    parent_local_x_qq = miku_parent_local_x_qq.inverted() * trace_parent_local_x_qq
                    target_local_x_qq = miku_target_local_x_qq.inverted() * trace_target_local_x_qq

                    miku_local_x_axis = miku_model.get_local_x_axis(mname)
                    miku_local_y_axis = MVector3D.crossProduct(miku_local_x_axis, MVector3D(0, 0, 1))

                    for fno in fnos:
                        rot_bf = trace_rot_motion.calc_bf(mname, fno)
                        rot_parent_bf = trace_rot_motion.calc_bf(parent_name, fno)
                        
                        miku_bf = trace_miku_motion.calc_bf(mname, fno)
                        miku_parent_bf = trace_miku_motion.calc_bf(parent_name, fno)

                        miku_bf.position = rot_bf.position.copy()
                        new_miku_qq = rot_bf.rotation.copy()

                        if (len(mname) > 2 and mname[2] == "指") or mname[1:] in ["ひじ", "ひざ"]:
                            # ひじ・指・ひざは念のためX捩り除去
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, new_miku_qq, miku_local_x_axis)
                            new_miku_qq = now_yz_qq

                        if mname[1:] not in ["肩", "足首"]:
                            new_miku_qq = miku_parent_bf.rotation.inverted() * rot_parent_bf.rotation * new_miku_qq
                                                
                        if mname[1:] in ["肩"] or mname in ["首"]:
                            new_miku_qq = parent_local_x_qq.inverted() * new_miku_qq
                        elif mname in ["頭"]:
                            new_miku_qq = parent_local_x_qq.inverted() * new_miku_qq * target_local_x_qq
                        elif mname[1:] in ["腕"]:
                            new_miku_qq = new_miku_qq * target_local_x_qq
                        elif mname[1:] in ["手首"]:
                            new_miku_qq = miku_parent_local_x_qq.inverted() * new_miku_qq * target_local_x_qq
                        else:
                            new_miku_qq = miku_parent_local_x_qq.inverted() * new_miku_qq * target_local_x_qq

                        if len(mname) > 2 and mname[2] == "指":
                            # 指は正方向にしか曲がらない
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, new_miku_qq, miku_local_x_axis)
                            new_miku_qq = MQuaternion.fromAxisAndAngle(MVector3D(0, 0, -1), now_yz_qq.toDegree())
                        elif mname[1:] in ["ひじ"]:
                            # ひじは念のためX捩り除去
                            _, _, _, now_yz_qq = MServiceUtils.separate_local_qq(fno, mname, new_miku_qq, miku_local_x_axis)
                            new_miku_qq = now_yz_qq
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

def create_bone(trace_model: PmxModel, jname: str, jconn: dict, target_bone_global_vecs: dict):
    # MMDボーン名
    mname = jconn["mmd"]
    if mname in trace_model.bones:
        return

    joints = list(target_bone_global_vecs[jname].values())
    parent_joints = list(target_bone_global_vecs[jconn["parent"]].values())
    # 親ボーン
    parent_bone = trace_model.bones[PMX_CONNECTIONS[jconn["parent"]]['mmd']] if jconn["parent"] in PMX_CONNECTIONS else trace_model.bones[jconn["parent"]]

    bone_length = np.median(np.linalg.norm(np.array(joints) - np.array(parent_joints), ord=2, axis=1))

    # # 親からの相対位置
    # if "指" in jconn["display"]:
    #     # 指は完全にミクに合わせる
    #     bone_relative_pos = miku_model.bones[mname].position - miku_model.bones[parent_bone.name].position
    # elif "足" == mname[-1] or "ひざ" == mname[-2:] or "足首" == mname[-2:]:
    #     # 足は方向はミクに合わせる。長さはトレース元
    #     bone_relative_pos = miku_model.bones[mname].position - miku_model.bones[parent_bone.name].position
    #     bone_relative_pos *= bone_length / bone_relative_pos.length()
    # else:
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


# 平滑化
def smooth_values(delimiter: int, values: list):
    smooth_vs = []
    data = np.array(values)

    # 前後のフレームで平均を取る
    if len(data) > delimiter:
        move_avg = np.convolve(data, np.ones(delimiter)/delimiter, 'valid')
        # 移動平均でデータ数が減るため、前と後ろに同じ値を繰り返しで補填する
        fore_n = int((delimiter - 1)/2)
        back_n = delimiter - 1 - fore_n
        smooth_vs = np.hstack((np.tile([move_avg[0]], fore_n), move_avg, np.tile([move_avg[-1]], back_n)))
    else:
        avg = np.mean(data)
        smooth_vs = np.tile([avg], len(data))

    smooth_vs *= delimiter / 9

    return smooth_vs

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
    "pelvis": {"mmd": "下半身", "parent": "グルーブ", "tail": "pelvis2", "display": "体幹"},
    "pelvis2": {"mmd": "下半身先", "parent": "pelvis", "tail": "", "display": "体幹"},
    "left_hip": {"mmd": "左足", "parent": "pelvis", "tail": "left_knee", "display": "左足"},
    "right_hip": {"mmd": "右足", "parent": "pelvis", "tail": "right_knee", "display": "右足"},
    "left_knee": {"mmd": "左ひざ", "parent": "left_hip", "tail": "left_ankle", "display": "左足"},
    "right_knee": {"mmd": "右ひざ", "parent": "right_hip", "tail": "right_ankle", "display": "右足"},
    "left_ankle": {"mmd": "左足首", "parent": "left_knee", "tail": "left_foot_index", "display": "左足"},
    "right_ankle": {"mmd": "右足首", "parent": "right_knee", "tail": "right_foot_index", "display": "右足"},
    "left_foot_index": {"mmd": "左つま先", "parent": "left_ankle", "tail": "", "display": "左足"},
    "right_foot_index": {"mmd": "右つま先", "parent": "right_ankle", "tail": "", "display": "右足"},
    "left_heel": {"mmd": "左かかと", "parent": "left_ankle", "tail": "", "display": "左足"},
    "right_heel": {"mmd": "右かかと", "parent": "right_ankle", "tail": "", "display": "右足"},
    "spine1": {"mmd": "上半身", "parent": "グルーブ", "tail": "spine2", "display": "体幹"},
    "spine2": {"mmd": "上半身2", "parent": "spine1", "tail": "spine3", "display": "体幹"},
    "left_collar": {"mmd": "左肩", "parent": "spine2", "tail": "left_shoulder", "display": "左手"},
    "right_collar": {"mmd": "右肩", "parent": "spine2", "tail": "right_shoulder", "display": "右手"},
    "left_shoulder": {"mmd": "左腕", "parent": "left_collar", "tail": "left_elbow", "display": "左手"},
    "right_shoulder": {"mmd": "右腕", "parent": "right_collar", "tail": "right_elbow", "display": "右手"},
    "left_elbow": {"mmd": "左ひじ", "parent": "left_shoulder", "tail": "left_wrist", "display": "左手"},
    "right_elbow": {"mmd": "右ひじ", "parent": "right_shoulder", "tail": "right_wrist", "display": "右手"},
    "left_wrist": {"mmd": "左手首", "parent": "left_elbow", "tail": "left_f_base", "display": "左手"},
    "right_wrist": {"mmd": "右手首", "parent": "right_elbow", "tail": "right_f_base", "display": "右手"},
    "left_f_base": {"mmd": "左指根元", "parent": "left_wrist", "tail": "", "display": "左指"},
    "right_f_base": {"mmd": "右指根元", "parent": "right_wrist", "tail": "", "display": "右指"},
    "left_thumb": {"mmd": "左親指先", "parent": "left_wrist", "tail": "", "display": "左指"},
    "left_index": {"mmd": "左人差指先", "parent": "left_wrist", "tail": "", "display": "左指"},
    "left_pinky": {"mmd": "左小指先", "parent": "left_wrist", "tail": "", "display": "左指"},
    "right_thumb": {"mmd": "右親指先", "parent": "right_wrist", "tail": "", "display": "右指"},
    "right_index": {"mmd": "右人差指先", "parent": "right_wrist", "tail": "", "display": "右指"},
    "right_pinky": {"mmd": "右小指先", "parent": "right_wrist", "tail": "", "display": "右指"},
    "spine3": {"mmd": "首", "parent": "spine2", "tail": "neck", "display": "体幹"},
    "neck": {"mmd": "頭", "parent": "spine3", "tail": "head", "display": "体幹"},
    "head": {"mmd": "頭先", "parent": "neck", "tail": "", "display": "体幹"},
    "nose": {"mmd": "鼻", "parent": "head", "tail": "", "display": "顔"},
    "right_eye": {"mmd": "右目", "parent": "nose", "tail": "", "display": "顔"},
    "left_eye": {"mmd": "左目", "parent": "nose", "tail": "", "display": "顔"},
    "センター": {"mmd": "センター", "parent": "全ての親", "tail": "pelvis2", "display": "センター"},
    "グルーブ": {"mmd": "グルーブ", "parent": "センター", "tail": "pelvis2", "display": "センター"},
}



VMD_CONNECTIONS = {
    'pelvis': {'direction': ('pelvis', 'pelvis2'), 'up': ('left_hip', 'right_hip'), 'cancel': []},
    'spine1': {'direction': ('spine1', 'spine2'), 'up': ('left_shoulder', 'right_shoulder'), 'cancel': []},
    'spine2': {'direction': ('spine2', 'spine3'), 'up': ('left_shoulder', 'right_shoulder'), 'cancel': ['spine1']},
    'spine3': {'direction': ('spine3', 'neck'), 'up': ('left_eye', 'right_eye'), 'cancel': ['spine1', 'spine2']},
    'neck': {'direction': ('neck', 'head'), 'up': ('left_eye', 'right_eye'), 'cancel': ['spine1', 'spine2', 'spine3']},

    'left_collar': {'direction': ('left_collar', 'left_shoulder'), 'up': ('spine2', 'spine3'), 'cancel': ['spine1', 'spine2']},
    'left_shoulder': {'direction': ('left_shoulder', 'left_elbow'), 'up': ('left_collar', 'left_shoulder'), 'cancel': ['spine1', 'spine2', 'left_collar']},
    'left_elbow': {'direction': ('left_elbow', 'left_wrist'), 'up': ('left_shoulder', 'left_elbow'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder']},
    'left_wrist': {'direction': ('left_wrist', 'left_f_base'), 'up': ('left_index', 'left_pinky'), 'cancel': ['spine1', 'spine2', 'left_collar', 'left_shoulder', 'left_elbow']},
    'left_hip': {'direction': ('left_hip', 'left_knee'), 'up': ('pelvis', 'left_hip'), 'cancel': ['pelvis']},
    'left_knee': {'direction': ('left_knee', 'left_ankle'), 'up': ('left_hip', 'left_knee'), 'cancel': ['pelvis', 'left_hip']},
    'left_ankle': {'direction': ('left_ankle', 'left_foot_index'), 'up': ('left_knee', 'left_ankle'), 'cancel': ['pelvis', 'left_hip', 'left_knee']},

    'right_collar': {'direction': ('right_collar', 'right_shoulder'), 'up': ('spine2', 'spine3'), 'cancel': ['spine1', 'spine2']},
    'right_shoulder': {'direction': ('right_shoulder', 'right_elbow'), 'up': ('right_collar', 'right_shoulder'), 'cancel': ['spine1', 'spine2', 'right_collar']},
    'right_elbow': {'direction': ('right_elbow', 'right_wrist'), 'up': ('right_shoulder', 'right_elbow'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder']},
    'right_wrist': {'direction': ('right_wrist', 'right_f_base'), 'up': ('right_index', 'right_pinky'), 'cancel': ['spine1', 'spine2', 'right_collar', 'right_shoulder', 'right_elbow']},
    'right_hip': {'direction': ('right_hip', 'right_knee'), 'up': ('pelvis', 'right_hip'), 'cancel': ['pelvis']},
    'right_knee': {'direction': ('right_knee', 'right_ankle'), 'up': ('right_hip', 'right_knee'), 'cancel': ['pelvis', 'right_hip']},
    'right_ankle': {'direction': ('right_ankle', 'right_foot_index'), 'up': ('right_knee', 'right_ankle'), 'cancel': ['pelvis', 'right_hip', 'right_knee']},
}
