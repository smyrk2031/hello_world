import os
import glob
import cv2
mypath = os.getcwd()

save_capnum = 0
cap = cv2.VideoCapture(1)

while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    #フレームが取得できなかった場合は、画面を閉じる
    if not ret:
        break
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    # Escキーを入力されたら画面を閉じる
    if key == 27:
        break
    # spaceキーを入力されたら撮影画像を保存する
    elif key == 32:
        print("撮影&保存★")
        #cv2.imwrite("C:/Users/kushi/TechLife/makes/AI/081_YOLO/01_220101_YOLOX/YOLOX-main/custom_dataset/01_orginal_traindatas/custom_img_" + str()+".jpg", frame)
        cv2.imwrite(str(mypath) + "/custom_dataset/01_orginal_traindatas/custom_img_" + str(save_capnum)+".jpg", frame)
        save_capnum += 1

cap.release()
cv2.destroyAllWindows()
