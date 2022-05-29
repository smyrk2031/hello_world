from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg


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

    # カラム1
    col_1 = [
        [sg.Canvas(size=(640, 480), key='-CANVAS_2-')],
        ]
    # カラム2
    col_2 = [
        [sg.Canvas(size=(640, 480), key='-CANVAS_1-')],
        [sg.Slider(range=(10, 500), default_value=40, size=(20, 10), orientation='h', key='-SLIDER-DATAPOINTS-')],
        [sg.Button('ボタン１１')],
        [sg.Button('ボタン２２')],
        [sg.Button('ボタン３３')]
        ]
    layout = [
                # 上部カラム
                [
                    sg.Column(col_1, vertical_alignment='True'),
                    sg.Column(col_2, vertical_alignment='True', justification='True')
                ],
            
                # 下部カラム
                [
                    sg.T(size=(45,5), key='-M_BOX_1-', background_color='black'),
                    sg.Multiline(size=(45,5), key='-M_BOX_2-'),
                    sg.Canvas(size=(640, 480), key='-CANVAS_3-')
                ]
            ]

    # ------------------------------------------------------
    # Window設定
    # ------------------------------------------------------
    sg.theme('Topanga')   # GUIテーマの変更
    window = sg.Window(
                'テスト_PySimpleGUI', # タイトルバーのタイトル
                layout, # 採用するレイアウトの変数
                finalize=True,
                auto_size_text=True,
                location=(0, 0),
                # no_titlebar=True, # タイトルバー無しにしたい時はコメントアウトを解除
                )
    # window.Maximize()   # フルスクリーン化したい時にはコメントアウトを解除

    canvas_elem_1 = window['-CANVAS_1-'] # CANVAS_1 = 折れ線アニメーショングラフ
    canvas_elem_2 = window['-CANVAS_2-'] # CANVAS_2 = 赤点の点滅グラフ
    canvas_elem_3 = window['-CANVAS_3-'] # CANVAS_3 = 円の回転アニメーショングラフ
    
    m_box_1 = window['-M_BOX_1-']   # 左下左メッセージボックス
    m_box_2 = window['-M_BOX_2-']   # 左下右メッセージボックス

    canvas_1 = canvas_elem_1.TKCanvas
    canvas_2 = canvas_elem_2.TKCanvas
    canvas_3 = canvas_elem_3.TKCanvas
    
    
    # ------------------------------------------------------
    # プロット設定
    #-------------------------------------------------------

    # matplotlibスタイル（'dark_background'）
    plt.style.use('dark_background') 

    # グラフサイズ変更（figsize=(横インチ ,縦インチ)
    fig_1 = Figure(figsize=(2, 2))    
    fig_2 = Figure(figsize=(6, 4)) 
    fig_3 = Figure(figsize=(1, 1)) 

    # axesオブジェクト設定（1行目・1列・1番目）
    ax_1 = fig_1.add_subplot(111)
    ax_2 = fig_2.add_subplot(111)
    ax_3 = fig_3.add_subplot(111)

    ax_1.xaxis.set_visible(False) # 軸消去
    ax_1.yaxis.set_visible(False)
    ax_2.xaxis.set_visible(False)
    ax_2.yaxis.set_visible(False)
    ax_3.xaxis.set_visible(False)
    ax_3.yaxis.set_visible(False)
    
    # x軸, y軸のラベル
    ax_1.set_xlabel("X axis")
    ax_1.set_ylabel("Y axis")

    # グリッドの描画
    ax_1.grid()

    # グラフの描画
    fig_agg_1 = draw_figure(canvas_1, fig_1)
    fig_agg_2 = draw_figure(canvas_2, fig_2)
    fig_agg_3 = draw_figure(canvas_3, fig_3)

    # ランダム数値データの用意
    NUM_DATAPOINTS = 10000 # ランダムデータ用の数値ポイント最大値
    dpts = [np.sqrt(1-np.sin(x)) for x in range(NUM_DATAPOINTS)] # ランダム数値リスト

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
            ax_2.cla()
            ax_3.cla()

            ax_1.grid() # ax_1のグリッド描画

            # グラフ1の描画
            # 黄色折れ線グラフのアニメーション
            data_points = int(values['-SLIDER-DATAPOINTS-']) # スライダーによるデータ数の変更（グラフ1用）
            ax_1.plot(range(data_points), dpts[i:i+data_points],  color='yellow')

            # グラフ2の描画
            # マップ上の赤丸の点滅: iが偶数であるときに点滅
            if i % 2 == 0:
                ax_2.plot([0, 1], [2, 3], 'ro') # x座標リスト、y座標リスト、ro = 赤丸

            # グラフ3の描画
            # 円-楕円 描画
            ellipse = patch.Ellipse(xy=(0.5, 0.5), width=np.sin(i/3), height=1, fill=False, ec='yellow')
            ax_3.add_patch(ellipse)


            # グラフの描画
            fig_agg_1.draw()
            fig_agg_2.draw()
            fig_agg_3.draw()


            # ------------------------------------------------------
            # メッセージ内容設定
            # ------------------------------------------------------

            # iが各数字の倍数であるときに色を変更しながらメッセージとして表示する
            if i % 5 == 0:
                window['-M_BOX_2-'].print(str(i)+' は 5 の倍数です。', text_color='green')
            if i % 7 == 0:
                window['-M_BOX_2-'].print(str(i)+' は 7 の倍数です。', text_color='black')
            if i % 13 == 0:
                window['-M_BOX_2-'].print(str(i)+' は 13 の倍数です。', text_color='red')

            # message.txtに書かれている内容を表示する
            with open('message.txt',encoding="utf-8") as message_file: # メッセージボックス入力
                window['-M_BOX_1-'].Update(message_file.read())

    window.close()


if __name__ == '__main__':
    main()