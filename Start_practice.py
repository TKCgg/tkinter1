#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tkinter

#クラスの作成
class Start_Script():
    
    def __init__(self):

        #ウィンドウの作成
        root = tkinter.Tk()
        root.title('機械学習モデル')
        root.minsize(640, 480)
        root.option_add("*font", ["メイリオ", 14])

        #キャンバス作成
        canvas = tkinter.Canvas(root, width=640, height=480)
        canvas.create_rectangle(60, 60, 560, 400, fill='gray78')
        canvas.pack()

        # ラベル配置
        label1 = tkinter.Label(text="モデル選択", fg="red")
        label1.place(x=270,y=10)

        #ボタン配置
        button1 = tkinter.Button(text="LightGBM")
        button1.place(x=120, y=420)

        button2 = tkinter.Button(text="Xgboost")
        button2.place(x=420, y=420)
        
        root.mainloop()


# In[ ]:




