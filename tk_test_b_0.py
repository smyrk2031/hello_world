import os 
import csv
import tkinter as tk
from tkinter import filedialog


class Dialog_menu_screen(object):

    def __init__(self,screen_title,screen_size,path_label,dialog_btn,start_btn):
        self.screen_title = screen_title
        self.screen_size = screen_size
        self.path_label = path_label
        self.dialog_btn = dialog_btn
        self.start_btn = start_btn


    #メイン画面表示
    def show_main_screen(self):
        root = tk.Tk()
        root.title(self.screen_title)
        root.geometry(self.screen_size)

        #ウィジェット作成
        #lambdaで引数付きのメソッドを読み出せる。
        path_label = tk.Label(root,text=self.path_label)
        path_box = tk.Entry(root,width=45)
        dialog_btn = tk.Button(root,text=self.dialog_btn,command=lambda:self.show_dialog(path_box))
        start_btn = tk.Button(root,width=10,text=self.start_btn,command=lambda:self.execute_button(path_box.get()))

        #ウィジェット位置
        path_box.grid(row=0,column=1,padx=10,pady=10)
        dialog_btn.grid(row=0,column=2,padx=10,pady=10)
        path_label.grid(row=0,column=0,padx=10,pady=10)
        start_btn.grid(row=1,column=2,padx=10,pady=10)

        root.mainloop()


    #ファイルダイアログを表示
    def show_dialog(self,path_box):
        file_type = [('csvテキスト','*.csv')] 
        folder_place = 'C:\\pg'
        file_path = filedialog.askopenfilename(filetypes = file_type, initialdir = folder_place) 
        # file_path = filedialog.askdirectory() 

        #キャンセルボタンを押さないなら
        if len(file_path) > 0 :
            path_box.delete(0,tk.END)
            path_box.insert(tk.END,file_path)


    #csvデータを表示。
    def execute_button(self,path_box):
        except_list = []
        with open(path_box,mode='r') as f:
            read_file = csv.reader(f)
            list_data = [row for row in read_file]
            print(list_data)
            return list_data


class Dialog_directory_screen(Dialog_menu_screen):
    #フォルダダイアログを表示
    def show_dialog(self,path_box):
        folder_place = 'C:\\pg'
        folder_path = filedialog.askdirectory(initialdir = folder_place) 

        #キャンセルボタンを押さないなら
        if len(folder_path) > 0 :
            path_box.delete(0,tk.END)
            path_box.insert(tk.END,folder_path)


    def execute_button(self,path_box):
        print(os.listdir(path_box))


if __name__ =="__main__":
    screen_title = "csv読み込み"
    screen_size = "500x90"
    path_label = "フォルダの場所:"
    dialog_btn = 'フォルダの場所...'
    start_btn = '実行'

    # Tkinter:ファイル取得のインスタンス作成
    csv_instance = Dialog_menu_screen(screen_title,screen_size,path_label,dialog_btn,start_btn)
    csv_instance.show_main_screen()

    # Tkinter:フォルダ取得のインスタンス作成
    screen_title = "フォルダ内容閲覧"
    directry_instance = Dialog_directory_screen(screen_title,screen_size,path_label,dialog_btn,start_btn)
    directry_instance.show_main_screen()

