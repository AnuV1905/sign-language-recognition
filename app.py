from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import model_from_json, Sequential
import operator
import time
import sys, os
import matplotlib.pyplot as plt
# import hunspell
from spellchecker import SpellChecker
from string import ascii_uppercase


get_custom_objects()['Sequential'] = Sequential
class Application:
    def __init__(self):
        self.directory = 'model'
        self.hs = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
      
        
        # Load models
        self.loaded_model = self.load_model("model-bw")
        self.loaded_model_dru = self.load_model("model-bw_dru")
        self.loaded_model_tkdi = self.load_model("model-bw_tkdi")
        self.loaded_model_smn = self.load_model("model-bw_smn")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("Sign language recognition")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        
        self.root.geometry("900x900")
        self.panel = tk.Label(self.root)
        self.panel.place(x = 135, y = 10, width = 640, height = 640)
        self.panel2 = tk.Label(self.root) # initialize image panel
        self.panel2.place(x = 460, y = 95, width = 310, height = 310)
        
        self.T = tk.Label(self.root)
        self.T.place(x=31,y = 17)
        self.T.config(text = "Sign Language Recognition",font=("courier",25,"bold"))
        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 500,y=600)
        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10,y = 650)
        self.T1.config(text="Character :",font=("Courier",20,"bold"))
        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 220,y=670)
        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 690)
        self.T2.config(text ="Word :",font=("Courier",20,"bold"))
        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 350,y=710)
        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10,y = 710)
        self.T3.config(text ="Sentence :",font=("Courier",20,"bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x = 250,y = 820)
        self.T4.config(text = "Suggestions",fg="red",font = ("Courier",20,"bold"))

        self.btcall = tk.Button(self.root,command = self.action_call,height = 0,width = 0)
        self.btcall.config(text = "About",font = ("Courier",14))
        self.btcall.place(x = 725, y = 0)

        self.bt1=tk.Button(self.root, command=self.action1,height = 0,width = 0)
        self.bt1.place(x = 26,y=750)
        
        self.bt2=tk.Button(self.root, command=self.action2,height = 0,width = 0)
        self.bt2.place(x = 325,y=750)
        
        self.bt3=tk.Button(self.root, command=self.action3,height = 0,width = 0)
        self.bt3.place(x = 625,y=750)
        
        self.bt4=tk.Button(self.root, command=self.action4,height = 0,width = 0)
        self.bt4.place(x = 125,y=800)
        
        self.bt5=tk.Button(self.root, command=self.action5,height = 0,width = 0)
        self.bt5.place(x = 425,y=800)
        
        self.str=""

        self.word=""
        self.current_symbol="Empty"
        self.photo="Empty"
        self.video_loop()

    def load_model(self, name):
        json_path = os.path.join(self.directory, f"{name}.json")
        weights_path = os.path.join(self.directory, f"{name}.h5")
        with open(json_path, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        print(f"{name} model input shape:", model.input_shape)
        return model

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            self.predict(res)
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol,font=("Courier",50))
            self.panel4.config(text=self.word,font=("Courier",40))
            self.panel5.config(text=self.str,font=("Courier",40))
            predicts=list(self.hs.candidates(self.word))
            predicts = list(predicts)  # convert to list to use indexing like predicts[0], predicts[1], etc.
            
            buttons = [self.bt1, self.bt2, self.bt3, self.bt4, self.bt5]
            for i, btn in enumerate(buttons):
                btn.config(text=predicts[i] if i < len(predicts) else "", font=("Courier", 20))
                       
        self.root.after(30, self.video_loop)
        
    def predict(self,test_image):
        # input_shape = self.loaded_model.input_shape[1:3]
        test_image = cv2.resize(test_image, (128, 128))
        test_image = test_image.reshape(1, 128, 128, 1)
        print("Input shape:", test_image.shape)
        # if test_image.ndim == 5:
        #     test_image = np.squeeze(test_image, axis=0)
        # print("After squeeze:", test_image.shape)
        result = self.loaded_model.predict(test_image)
        result_dru = self.loaded_model_dru.predict(test_image)
        result_tkdi = self.loaded_model_tkdi.predict(test_image)
        result_smn = self.loaded_model_smn.predict(test_image)
        prediction = {'blank': result[0][0]}
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        #LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        #LAYER 2
        if(self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
            prediction = {} 
            prediction['D'] = result_dru[0][0]
            prediction['R'] = result_dru[0][1]
            prediction['U'] = result_dru[0][2]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
            prediction = {}
            prediction['D'] = result_tkdi[0][0]
            prediction['I'] = result_tkdi[0][1]
            prediction['K'] = result_tkdi[0][2]
            prediction['T'] = result_tkdi[0][3]
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):
            prediction1 = {}
            prediction1['M'] = result_smn[0][0] 
            prediction1['N'] = result_smn[0][1]
            prediction1['S'] = result_smn[0][2]
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            if(prediction1[0][0] == 'S'):
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]
        if(self.current_symbol == 'blank'):
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1
        if(self.ct[self.current_symbol] > 60):
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if(len(self.str) > 16):
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol
    def action1(self):
        predicts = list(self.hs.candidates(self.word))
        if len(predicts) > 0:
            self.word = ""
            self.str += " "
            self.str += predicts[0]

    def action2(self):
        predicts = list(self.hs.candidates(self.word))
        if len(predicts) > 1:
            self.word = ""
            self.str += " "
            self.str += predicts[1]

    def action3(self):
        predicts = list(self.hs.candidates(self.word))
        if len(predicts) > 2:
            self.word = ""
            self.str += " "
            self.str += predicts[2]

    def action4(self):
        predicts = list(self.hs.candidates(self.word))
        if len(predicts) > 3:
            self.word = ""
            self.str += " "
            self.str += predicts[3]

    def action5(self):
        predicts = list(self.hs.candidates(self.word))
        if len(predicts) > 4:
            self.word = ""
            self.str += " "
            self.str += predicts[4]

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()

    def action_call(self) :
        
        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("900x900")
        
        # test_image = cv2.imread("Pictures/sir.jpg", 1)
        # # test_image = cv2.resize(test_image, (300, 300))
        # cv2.imwrite("Pictures/sir.png", test_image)
        # return 
        
        self.tx = tk.Label(self.root1)
        self.tx.place(x = 330,y = 20)
        self.tx.config(text = "Efforts By", fg="red", font = ("Courier",30,"bold"))

        self.photo1 = tk.PhotoImage(file='Pictures/anushka.jpg')
        self.w1 = tk.Label(self.root1, image = self.photo1)
        self.w1.place(x = 20, y = 105)
        self.tx6 = tk.Label(self.root1)
        self.tx6.place(x = 20,y = 250)
        self.tx6.config(text = "Anushka Verma\n2202073", font = ("Courier",15,"bold"))

        self.photo2 = tk.PhotoImage(file='Pictures/nikhil.jpg')
        self.w2 = tk.Label(self.root1, image = self.photo2)
        self.w2.place(x = 200, y = 105)
        self.tx2 = tk.Label(self.root1)
        self.tx2.place(x = 200,y = 250)
        self.tx2.config(text = "Nikhil Ranjan\n2202002", font = ("Courier",15,"bold"))

        
        self.photo3 = tk.PhotoImage(file='Pictures/zeba.jpg')
        self.w3 = tk.Label(self.root1, image = self.photo3)
        self.w3.place(x = 380, y = 105)
        self.tx3 = tk.Label(self.root1)
        self.tx3.place(x = 380,y = 250)
        self.tx3.config(text = "Zeba Khalid\n2202015", font = ("Courier",15,"bold"))
        
        

        
        self.tx7 = tk.Label(self.root1)
        self.tx7.place(x = 170,y = 360)
        self.tx7.config(text = "Under the supervision of", fg="red", font = ("Courier",30,"bold"))

        self.photo6 = tk.PhotoImage(file='Pictures/sir.png')
        self.w6 = tk.Label(self.root1, image = self.photo6)
        self.w6.place(x = 350, y = 420)
        self.tx6 = tk.Label(self.root1)
        self.tx6.place(x = 230,y = 670)
        self.tx6.config(text = "Dr. Amitesh Kumar", font = ("Courier",30,"bold"))

print("Starting Application...")
pba = Application()
pba.root.mainloop()
