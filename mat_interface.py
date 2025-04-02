import sys
import os
# sys.path.append(r"D:\Projects\all_model\video_model\video_predict\ultralytics")
import tkinter as tk
from tkinter import filedialog, messagebox

from my_predict_s import mat_behavior_predict
from clip_video.clip_video import video_cut
from USV_model.USV_pred import getFormatData, Trainer
from USV_model.USV_train import getFormatData_train, Trainer_train

from modelSet import ML_concert

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Multifunctional interface")
    root.geometry("900x700")
    root.configure(bg="#f0f0f0")  # 更柔和的背景颜色

    button_style = {"padx": 10, "pady": 5, "bg": "#5cb85c", "fg": "white", "font": ("Arial", 10, "bold")}
    label_style = {"bg": "#f0f0f0", "fg": "#333333", "font": ("Arial", 10)}
    frame_style = {"padx": 10, "pady": 10, "bg": "#ffffff", "relief": "solid", "bd": 2}


    # 左上 - 超声波预测
    def ultrasound_prediction_run():
        if model_file and table_file and save_folder:
            print(str(table_file))
            pre_read_excel_address = str(table_file)
            with open("address.txt", "w", encoding="utf-8") as file:
                file.write(str(table_file))

            getFormatData(pre_read_excel_address)  # 数据预处理：数据清洗和词向量读要预测数据
            trainer = Trainer()
            trainer.save_model_address = model_file
            trainer.load_model_address = model_file
            trainer.save_excel = table_file
            trainer.save_csv_address = save_folder + "\output.csv"
            trainer.test()  # 预测
            result = f"Ultrasonic prediction complete!\nmodel file: {model_file}\nform document: {table_file}\nSave folder: {save_folder}"
            label_result_ultrasound.config(text=result)
        else:
            messagebox.showwarning("warning", "Make sure all files and folders are selected!")


    def select_ultrasound_model():
        global model_file
        model_file = filedialog.askopenfilename(filetypes=[("model file", "*.pt")])
        label_model_ultrasound.config(text=model_file)


    def select_ultrasound_table():
        global table_file
        table_file = filedialog.askopenfilename(filetypes=[("form document", "*.xlsx")])
        label_table_ultrasound.config(text=table_file)


    def select_ultrasound_save_folder():
        global save_folder
        save_folder = filedialog.askdirectory()
        label_save_ultrasound.config(text=save_folder)


    frame_ultrasound = tk.LabelFrame(root, text="Ultrasonic prediction", **frame_style)
    frame_ultrasound.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    button_model_ultrasound = tk.Button(frame_ultrasound, text="Select model file", command=select_ultrasound_model,
                                        **button_style)
    button_model_ultrasound.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    label_model_ultrasound = tk.Label(frame_ultrasound, text="No model file was selected", **label_style)
    label_model_ultrasound.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

    button_table_ultrasound = tk.Button(frame_ultrasound, text="Select form file", command=select_ultrasound_table,
                                        **button_style)
    button_table_ultrasound.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
    label_table_ultrasound = tk.Label(frame_ultrasound, text="No form file is selected", **label_style)
    label_table_ultrasound.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

    button_save_ultrasound = tk.Button(frame_ultrasound, text="Select save folder", command=select_ultrasound_save_folder,
                                       **button_style)
    button_save_ultrasound.grid(row=4, column=0, padx=5, pady=5, sticky="ew")
    label_save_ultrasound = tk.Label(frame_ultrasound, text="No save folder is selected", **label_style)
    label_save_ultrasound.grid(row=5, column=0, padx=5, pady=5, sticky="ew")

    button_run_ultrasound = tk.Button(frame_ultrasound, text="Run", command=ultrasound_prediction_run, **button_style)
    button_run_ultrasound.grid(row=6, column=0, padx=5, pady=10, sticky="ew")
    label_result_ultrasound = tk.Label(frame_ultrasound, text="", **label_style)
    label_result_ultrasound.grid(row=7, column=0, padx=5, pady=5, sticky="ew")


    # 左下 - 超声波模型训练
    def ultrasound_training_run():
        if model_file_training and threshold_value:
            train_excel_address = model_file_training
            thresh = 1.5
            getFormatData_train(train_excel_address, thresh)  # 数据预处理：数据清洗和词向量读要训练的数据
            trainer = Trainer_train()
            trainer.save_model_address = '\my_model.pt'
            trainer.load_model_address = '\my_model.pt'
            trainer.train(epochs=100)  # 数据训练
            trainer.test()  # 测试
            result = f"model file: {model_file_training}\nthreshold value: {threshold_value}"
            label_result_training.config(text=result)
        else:
            messagebox.showwarning("warning", "Make sure all files and thresholds are entered!")


    def select_training_model():
        global model_file_training
        model_file_training = filedialog.askopenfilename(filetypes=[("Training file", "*.xlsx")])
        label_model_training.config(text=model_file_training)


    def set_threshold_value():
        global threshold_value
        threshold_value = entry_threshold.get()
        label_threshold.config(text=f"threshold value: {threshold_value}")


    frame_training = tk.LabelFrame(root, text="Training file", **frame_style)
    frame_training.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    button_model_training = tk.Button(frame_training, text="Select training file", command=select_training_model, **button_style)
    button_model_training.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    label_model_training = tk.Label(frame_training, text="No training file is selected", **label_style)
    label_model_training.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

    entry_threshold = tk.Entry(frame_training, font=("Arial", 10))
    entry_threshold.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
    button_threshold = tk.Button(frame_training, text="Set threshold", command=set_threshold_value, **button_style)
    button_threshold.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
    label_threshold = tk.Label(frame_training, text="No threshold is set", **label_style)
    label_threshold.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

    button_run_training = tk.Button(frame_training, text="Run", command=ultrasound_training_run, **button_style)
    button_run_training.grid(row=5, column=0, padx=5, pady=10, sticky="ew")
    label_result_training = tk.Label(frame_training, text="", **label_style)
    label_result_training.grid(row=6, column=0, padx=5, pady=5, sticky="ew")


    # 右上 - 视频裁剪
    def video_clipping_run():
        if video_file_clipping and save_folder_clipping:
            input_address = video_file_clipping
            output_address = save_folder_clipping + "\outpy.avi"
            video_cut(input_address, output_address)
            result = f"Cropping is complete!\nvideo file: {video_file_clipping}\nSave folder: {save_folder_clipping}"
            label_result_clipping.config(text=result)
        else:
            messagebox.showwarning("warning", "Make sure all files and folders are selected!")


    def select_video_file_clipping():
        global video_file_clipping
        video_file_clipping = filedialog.askopenfilename(filetypes=[("video file", "*.mp4;*.avi;*.mov")])
        label_video_clipping.config(text=video_file_clipping)


    def select_save_folder_clipping():
        global save_folder_clipping
        save_folder_clipping = filedialog.askdirectory()
        label_save_clipping.config(text=save_folder_clipping)


    frame_clipping = tk.LabelFrame(root, text="Video clipping", **frame_style)
    frame_clipping.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    button_video_clipping = tk.Button(frame_clipping, text="Select video file", command=select_video_file_clipping, **button_style)
    button_video_clipping.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    label_video_clipping = tk.Label(frame_clipping, text="No video file is selected", **label_style)
    label_video_clipping.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

    button_save_clipping = tk.Button(frame_clipping, text="Select video file", command=select_save_folder_clipping,
                                     **button_style)
    button_save_clipping.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
    label_save_clipping = tk.Label(frame_clipping, text="No video file is selected", **label_style)
    label_save_clipping.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

    button_run_clipping = tk.Button(frame_clipping, text="Run", command=video_clipping_run, **button_style)
    button_run_clipping.grid(row=4, column=0, padx=5, pady=10, sticky="ew")
    label_result_clipping = tk.Label(frame_clipping, text="", **label_style)
    label_result_clipping.grid(row=5, column=0, padx=5, pady=5, sticky="ew")


    # 右下 - 小鼠交配行为识别
    def mouse_behavior_run():
        if video_file_behavior and save_folder_behavior:
            video_address = video_file_behavior
            output_address = save_folder_behavior + "\Output_s.txt"
            mat_behavior_predict(video_address, output_address)
            result = f"finish！\nvideo file: {video_file_behavior}\nSave folder: {save_folder_behavior}"
            label_result_behavior.config(text=result)
        else:
            messagebox.showwarning("warning", "Make sure all files and folders are selected！")


    def select_video_file_behavior():
        global video_file_behavior
        video_file_behavior = filedialog.askopenfilename(filetypes=[("video file", "*.mp4;*.avi;*.mov")])
        label_video_behavior.config(text=video_file_behavior)


    def select_save_folder_behavior():
        global save_folder_behavior
        save_folder_behavior = filedialog.askdirectory()
        label_save_behavior.config(text=save_folder_behavior)


    frame_behavior = tk.LabelFrame(root, text="Recognition of mating behavior in mice", **frame_style)
    frame_behavior.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    button_video_behavior = tk.Button(frame_behavior, text="Select video file", command=select_video_file_behavior, **button_style)
    button_video_behavior.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    label_video_behavior = tk.Label(frame_behavior, text="No video file is selected", **label_style)
    label_video_behavior.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

    button_save_behavior = tk.Button(frame_behavior, text="Select save folder", command=select_save_folder_behavior,
                                     **button_style)
    button_save_behavior.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
    label_save_behavior = tk.Label(frame_behavior, text="No save folder is selected", **label_style)
    label_save_behavior.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

    button_run_behavior = tk.Button(frame_behavior, text="Run", command=mouse_behavior_run, **button_style)
    button_run_behavior.grid(row=4, column=0, padx=5, pady=10, sticky="ew")
    label_result_behavior = tk.Label(frame_behavior, text="", **label_style)
    label_result_behavior.grid(row=5, column=0, padx=5, pady=5, sticky="ew")

    # 新增 - 机器学习模型预测模块
    global model_name
    global table_file_ml

    def ml_model_prediction_run():
        # if not table_file_ml:
        #     messagebox.showwarning("Warning", "Please select a table file!")
        #     return
        # if not model_name:
        #     messagebox.showwarning("Warning", "Please select a model!")
        #     return
        model_name = model_var.get()
        PR_AUC, F1_Score = ML_concert(table_file_ml, model_name)
        result = f"Select the {model_name} model, running successfully!\n{model_name} - PR AUC: {PR_AUC:.4f} - F1 Score: {F1_Score:.4f}'\n对应模型回归曲线图保存在本地路径“模型.png"
        label_result_ml.config(text=result)

    def select_ml_table():
        global table_file_ml
        table_file_ml = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if table_file_ml and table_file_ml.endswith(".xlsx"):
            label_table_ml.config(text=table_file_ml)
        else:
            messagebox.showerror("Error", "Invalid file format. Please select an .xlsx file.")
            table_file_ml = None


    def set_model_name(selected_model):
        label_model_ml.config(text=f"Selected model: {selected_model}")  # 使用选中的模型名称更新标签


    frame_ml = tk.LabelFrame(root, text="Other training models", **frame_style)
    frame_ml.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

    button_table_ml = tk.Button(frame_ml, text="Select file", command=select_ml_table, **button_style)
    button_table_ml.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    label_table_ml = tk.Label(frame_ml, text="No file selected", **label_style)
    label_table_ml.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

    model_var = tk.StringVar(frame_ml)
    model_var.set("KNN")  # 默认选择 KNN
    model_menu = tk.OptionMenu(frame_ml, model_var, "KNN", "SVM", "Random Forest", "CatBoost", "DNN", "LSTM",
                               "LightGBM", command=set_model_name)
    model_menu.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
    label_model_ml = tk.Label(frame_ml, text=f"Selected model: {model_var.get()}", **label_style)
    label_model_ml.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

    button_run_ml = tk.Button(frame_ml, text="Run", command=ml_model_prediction_run, **button_style)
    button_run_ml.grid(row=4, column=0, padx=5, pady=10, sticky="ew")

    label_result_ml = tk.Label(frame_ml, text="", **label_style)
    label_result_ml.grid(row=5, column=0, padx=5, pady=5, sticky="ew")

    # 调整窗口布局
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)

    root.mainloop()

    # video_address = r"../../video_data/input/test_mating.mp4"
    # output_address = "../../video_data/output/Output_s.txt"
    # mat_behavior_predict(video_address, output_address)

    # input_address = r"D:\Projects\all_model\video_model\video_data\input\test_clip.mp4"
    # output_address = r"'outpy.avi'"
    # video_cut(input_address, output_address)

    # input_address = r'./USV_data/input/usv_test.xlsx'
    # getFormatData(input_address)  # 数据预处理：数据清洗和词向量读要预测数据
    # trainer = Trainer()
    # trainer.module = "./USV_data/model/gru.pt"
    # trainer.input_address = r'./USV_data/input/usv_test.xlsx'
    # trainer.output_address = r"./USV_data/output/output.csv"
    # # trainer.train(epochs=100)  # 数据训练
    # trainer.test()  # 预测

    # input_address = r'../USV_data/input/usv_train.xlsx'
    # getFormatData_USV(input_address)  # 数据预处理：数据清洗和词向量读要训练的数据
    # trainer = Trainer_USV()
    # trainer.output_module = r'../USV_data/model/my_model.pt'
    # trainer.train(epochs=100)  # 数据训练
    # trainer.test()  # 测试
