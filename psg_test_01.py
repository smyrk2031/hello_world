import os
import sys
import time
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg


"""
要件A, GPS表示
1: 左半分に散布図でGPSデータを表示するようにする
2: 地図情報のインストールをどうするか・・走行波形を事前にcsvで同フォルダに保存して、それを登録する感じ？
3: 

要件B, 設定した入力値を2つ表示する
1: GPSが基準位置に入ったら強調表示する
2: 目標値と実値と、その差分を表示する
3: 差分の大小を一目で見やすくする
4: 
5: 


# 注意点
1: Pythonを使うので、リアルタイム表示処理は遅い、、timeを使うと100ms以下は精度悪くなる

"""

# 図の描画関数
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def main():

    # ------------------------------------------------------
    # カラム設定
    # ------------------------------------------------------

    # 表示するデータの初期値
    input_data_ini_1 = 0
    input_data_ini_2 = 0

    # 背景色を変化させる_7段階で分かりやすいようにする_ 緑ならOK_(青 → 緑 → 赤)
    bg_colors = ["#0000FF","#1E90FF","#00EEEE","#00FF00","#FFD700","#FFA500","#FF0000"]
    bg_color_1 = bg_colors[3]
    bg_color_2 = bg_colors[3]

    # カラム1
    col_1 = [
        [sg.Text('GPS表示', size=(10, 1), text_color="orange", font=('Times New Roman',40,"bold"))],
        [sg.Canvas(size=(320, 480), key='-CANVAS_1-')],
        ]
    col_2 = [
        [sg.Text(' ', size=(20, 1))],
        [sg.Text(' ', size=(20, 1))],
        [sg.Text(' ', size=(20, 1))],
        [sg.Text(' ', size=(20, 1))],
        [sg.Text('アクセル開度 [%]', size=(20, 1), text_color="white", font=('Times New Roman',36,"bold"))],
        [sg.Text(input_data_ini_1, size=(4, 1), text_color="black", font=('Times New Roman',56,"bold"), background_color=bg_color_1)],
        [sg.Text('ステアリング舵角 [deg]', size=(20, 1), text_color="white", font=('Times New Roman',36,"bold"))],
        [sg.Text(input_data_ini_2, size=(4, 1), text_color="black", font=('Times New Roman',56,"bold"), background_color=bg_color_2)],
        [sg.Text(' ', size=(20, 1))],
        [sg.Text(' ', size=(20, 1))],
        ]
    layout = [
                # 上部カラム
                [
                    sg.Column(col_1, vertical_alignment='True'),
                    sg.Column(col_2, vertical_alignment='True'),
                ]
            ]
            
            #     # 下部カラム
            #     [sg.Text('GPSプロット', size=(150, 100)), sg.InputText('1.0', key='-SPEED-', enable_events=True, size=(20, 1))]
            # ]
    # ------------------------------------------------------
    # Window設定
    # ------------------------------------------------------
    sg.theme('Topanga')   # GUIテーマの変更
    window = sg.Window(
                'テストコース_走行状態再現ナビ', # タイトルバーのタイトル
                layout, # 採用するレイアウトの変数
                finalize=True,
                auto_size_text=True,
                location=(0, 0),
                # no_titlebar=True, # タイトルバー無しにしたい時はコメントアウトを解除
                )
    # window.Maximize()   # フルスクリーン化したい時にはコメントアウトを解除

    # canvas_elem_1 = window['-CANVAS_1-'] # CANVAS_1 = 折れ線アニメーショングラフ
    canvas_elem_1 = window['-CANVAS_1-'] # CANVAS_2 = 赤点の点滅グラフ
    # canvas_elem_3 = window['-CANVAS_3-'] # CANVAS_3 = 円の回転アニメーショングラフ
    
    # m_box_1 = window['-M_BOX_1-']   # 左下左メッセージボックス
    # m_box_2 = window['-M_BOX_2-']   # 左下右メッセージボックス

    # canvas_1 = canvas_elem_1.TKCanvas
    canvas_1 = canvas_elem_1.TKCanvas
    # canvas_3 = canvas_elem_3.TKCanvas


    # ------------------------------------------------------
    # プロット設定
    #-------------------------------------------------------

    # matplotlibスタイル（'dark_background'）
    plt.style.use('dark_background') 

    # グラフサイズ変更（figsize=(横インチ ,縦インチ)
    fig_1 = Figure(figsize=(6, 4)) 

    # axesオブジェクト設定（1行目・1列・1番目）
    ax_1 = fig_1.add_subplot(111)

    ax_1.xaxis.set_visible(False)# 軸消去
    ax_1.yaxis.set_visible(False)
    
    # x軸, y軸のラベル
    ax_1.set_xlabel("X axis")
    ax_1.set_ylabel("Y axis")

    # グリッドの描画
    ax_1.grid()

    # グラフの描画_描画の更新に使う
    fig_agg_1 = draw_figure(canvas_1, fig_1)

    # サンプルデータ読込
    # ランダム数値データの用意
    NUM_DATAPOINTS = 10000 # ランダムデータ用の数値ポイント最大値
    dpts = [np.sqrt(1-np.sin(x)) for x in range(NUM_DATAPOINTS)] # ランダム数値リスト
    # dpts_2 = [np.sqrt(1-np.sin(x)) for x in range(NUM_DATAPOINTS)]*2 # ランダム数値リスト

    # 地図情報と自己位置ポインタの重ね合わせ表示で、リアルタイム表示したい
    # 

    # ------------------------------------------------------
    # 描画設定
    #-------------------------------------------------------
    print("描画のループに入る")
    # 描画ループ
    while True:
        for i in range(len(dpts)):
            event, values = window.read(timeout=10)

            if event in ('Exit', None):
                exit(69)

            # グラフ描画のクリア
            ax_1.cla()

            # ax_1のグリッド描画
            ax_1.grid()

            # グラフ1の描画
            data_points = int(100)
            # ax_1.plot(range(data_points), dpts[i:i+data_points],  color='yellow')
            # ax_1.plot(dpts[i:i+data_points], dpts_2[i:i+data_points],  color='yellow')


            # 動的な走行操作支援機能
            # 背景色の変更
            # bg_color_1 = bg_colors[i]
            # bg_color_2 = bg_colors[i]
            

            # グラフの描画
            fig_agg_1.draw()
            # time.sleep(3)

        window.close()

if __name__ == '__main__':
    main()