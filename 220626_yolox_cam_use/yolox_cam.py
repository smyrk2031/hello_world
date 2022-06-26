import os
import time
import warnings
from pprint import pprint
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
import cv2
import torch


# import importlib
# postprocess = importlib.import_module("YOLOX-main.yolox.utils.boxes.postprocess")
# ValTransform = importlib.import_module("YOLOX-main.yolox.data.data_augment.ValTransform")
# COCO_CLASSES = importlib.import_module("YOLOX-main.yolox.data.datasets.COCO_CLASSES")
# get_exp = importlib.import_module("YOLOX-main.yolox.exp.get_exp")

# YOLOXの必要モジュールを読込む_カレントディレクトリがYOLOX内にないといけないのがいまいち
# loguruの部分でno-module-nameエラーが出る。。
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES

#import yolox.exp as exp_
from yolox.exp import get_exp


# WEBカメラ起動中に、YOLOXで推論結果を出力し続ける機能



# モデルの設定
MODEL_FILE = "weights/yolox_s.pth"
# MODEL_FILE = "weights/yolox_x.pth"
#MODEL_FILE = "yolox_x.pth"


MODEL_FILE.split("/")[-1].split(".pth")[0]
print(MODEL_FILE)
print(MODEL_FILE.split("/")[-1].split(".pth")[0])


# 推論用パラメータ
test_size = (224, 224) #(640, 640)    #(960, 960)
num_classes = 80
confthre =0.25  #0.1
nmsthre = 0.45


print(" YOLOモデルの取得")
#model = exp.get_model()
#model = exp_.yolox_base.Exp.get_model()
#exp_my = get_exp(exp_file=None, exp_name=MODEL_FILE.split(".")[0])  #"yolox_x")    # ここ苦戦した
exp_my = get_exp(exp_file=None, exp_name=MODEL_FILE.split("/")[-1].split(".pth")[0])  #"yolox_x")    # ここ苦戦した
model = exp_my.get_model()
# model.cuda()
model.eval()


# get custom trained checkpoint
#ckpt_file = "./YOLOX_outputs/cots_config/best_ckpt.pth"
#ckpt = torch.load(ckpt_file, map_location="cpu")
#model.load_state_dict(ckpt["model"])

print("torch_load実行")
ckpt = torch.load(MODEL_FILE, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)



# 推論_YOLOX
def yolox_inference(img, model, test_size): 
    print("yolox-infarenceファンクション")
    bboxes = []
    bbclasses = []
    scores = []
    
    preproc = ValTransform(legacy = False)

    tensor_img, _ = preproc(img, None, test_size)
    tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
    tensor_img = tensor_img.float()
    tensor_img = tensor_img#.cuda()

    with torch.no_grad():
        outputs = model(tensor_img)
        outputs = postprocess(
                    outputs, num_classes, confthre,
                    nmsthre, class_agnostic=True
                )

    if outputs[0] is None:
        return [], [], []
    
    outputs = outputs[0].cpu()
    bboxes = outputs[:, 0:4]

    bboxes /= min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    bbclasses = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]
    
    return bboxes, bbclasses, scores

def draw_yolox_predictions(img, bboxes, scores, bbclasses, confthre, classes_dict):
# def draw_yolox_predictions(img, bboxes, scores, bbclasses, confthre):
    print("draw_yolox_predictionsファンクション")
    print("BBOXの取得数は：", str(len(bboxes)))
    
    for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(bbclasses[i])
            # 犬猫以外は無視する
            #print("id：", str(classes_dict[cls_id]))
            
            #if classes_dict[cls_id] in ["cat", "dog"]:
            score = scores[i]
            #print("判定結果", str(cls_id))
            #print("スコア", str(score))

            if score < confthre:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            # BBOXのサイズを一時的に取得して、一番大きいものを記録
            #bb_size = (x1-x0) * (y1-y0)
            # 画像内にBBOXを上書きする
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(img, '{}:{:.1f}%'.format(classes_dict[cls_id], score * 100), (x0, y0 - 3), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), thickness = 1)
    return img



# メイン処理
# set_catdog = [15,16] #["cat", "dog"] #[15,16] #取得するクラスを限定化する
test_size = (224, 224) # 小さくすれば早くなって精度下がる


# カメラの画像を取得する
print("カメラの画像を取得する")
cap = cv2.VideoCapture(0)
t_fps = time.time()
while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    #フレームが取得できなかった場合は、画面を閉じる
    if not ret:
        break
    
    # FPS計算
    # now_fps = cap.get(cv2.CAP_PROP_FPS)
    now_fps = round((1 / (time.time() - t_fps)),1)
    t_fps = time.time()

    # YOLO推論を入れる
    bboxes, bbclasses, scores = yolox_inference(frame, model, test_size)
  
    # 取得したクラス番号を、カテゴリ名に変更する
    get_classes = np.array(bbclasses)

    # もし推論で物体検出したら、なにか処理する
    # 

    # frameに、YOLOX検出矩形を表示する
    frame  = draw_yolox_predictions(frame, bboxes, scores, bbclasses, confthre, COCO_CLASSES)


    # ウィンドウに出力
    disp_text = "FPS : "+str(now_fps) 
    cv2.putText(frame, disp_text, org=(100, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(0, 255, 0), thickness=2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    # Escキーを入力されたら画面を閉じる
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
