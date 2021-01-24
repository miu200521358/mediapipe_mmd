# -*- coding: utf-8 -*-
import os
import glob

import numpy as np
from tqdm import tqdm
import cv2
import mediapipe as mp

from mmd.utils.MLogger import MLogger

logger = MLogger(__name__, level=1)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def execute(args):
    try:
        logger.info('人物指推定開始: %s', args.img_dir, decoration=MLogger.DECORATION_BOX)

        # For webcam input:
        hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow('MediaPipe Hands', image)
            # qキーが押下されたタイミングで終了する
            if cv2.waitKey(5) & 0xFF == 27:
                break

        hands.close()
        cap.release()

        logger.info('人物指推定終了: %s', args.img_dir, decoration=MLogger.DECORATION_BOX)
        
        return True
    except Exception as e:
        logger.critical("指推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False
