import os
import tkinter as tk
from tkinter import *


root= Tk()
root.title('Main runner')
root.geometry('800x450')
#root.configure(fg=)

def KNC():
    os.startfile(r"F:\Tkinter_Module\final_project\models\KNC.py",operation="open")

def Kmeans():
    os.startfile(r"F:\Tkinter_Module\final_project\models\Kmeans.py",operation="open")

def NBCls():
    os.startfile(r"F:\Tkinter_Module\final_project\models\NaiveBayes.py",operation="open")

def SVMCls():
    os.startfile(r"F:\Tkinter_Module\final_project\models\SVM-Clas.py",operation="open")

def SVMReg():
    os.startfile(r"F:\Tkinter_Module\final_project\models\SVM-Reg.py",operation="open")

def Lasso():
    os.startfile(r"F:\Tkinter_Module\final_project\models\Lasso Reg.py",operation="open")

def Ridge():
    os.startfile(r"F:\Tkinter_Module\final_project\models\Ridge Reg.py",operation="open")

def LinReg():
    os.startfile(r"F:\Tkinter_Module\final_project\models\LinearReg.py",operation="open")

def LogReg():
    os.startfile(r"F:\Tkinter_Module\final_project\models\LogisticReg.py",operation="open")

def MLPCls():
    os.startfile(r"F:\Tkinter_Module\final_project\models\MLP Cls.py",operation="open")

def DecReg():
    os.startfile(r"F:\Tkinter_Module\final_project\models\Decision - Reg.py",operation="open")

def DecCls():
    os.startfile(r"C:\Users\MAHE\Downloads\Tkinter_Module\final_project\models\Decision - Cls.py",operation="open")
    
def Best_Class():
    os.startfile(r"C:\Users\MAHE\Downloads\Tkinter_Module\final_project\models\Best_Classifier.py",operation="open")

def Best_Reg():
    os.startfile(r"C:\Users\MAHE\Downloads\Tkinter_Module\final_project\models\Best_Regressor.py",operation="open")
    

Label(root,text="Choose the model based on the type of problem",font="System").place(x=200,y=20) 


Label(root,text="Regression",font="System").place(x=150,y=75)
Label(root,text="Clustering",font="System").place(x=350,y=75)
Label(root,text="Classification",font="System").place(x=550,y=75)


Button(root,text='K-Neighbors',activebackground="black",command=KNC,activeforeground="white").place(x=550,y=280)
Button(root,text='K-Means',command=Kmeans,activebackground="black",activeforeground="white").place(x=350,y=130)
Button(root,text='Multinomial Naive-Bayes',activebackground="black",command=NBCls,activeforeground="white").place(x=550,y=220)
Button(root,text='SVM Classifier',activebackground="black",command=SVMCls,activeforeground="white").place(x=550,y=130)
Button(root,text='SVM Regressor',activebackground="black",command=SVMReg,activeforeground="white").place(x=150,y=220)
Button(root,text='Lasso Regression',activebackground="black",command=Lasso,activeforeground="white").place(x=150,y=190)
Button(root,text='Ridge Regression',activebackground="black",command=Ridge,activeforeground="white").place(x=150,y=160)
Button(root,text='Linear Regression',activebackground="black",command=LinReg,activeforeground="white").place(x=150,y=130)
Button(root,text='Logistic Regression',activebackground="black",command=LogReg,activeforeground="white").place(x=550,y=250)
Button(root,text='MLP Classifier',activebackground="black",command=MLPCls,activeforeground="white").place(x=550,y=160)
Button(root,text='DecisionTree Regressor',activebackground="black",command=DecReg,activeforeground="white").place(x=150,y=250)
Button(root,text='DecisionTree Classifier',activebackground="black",command=DecCls,activeforeground="white").place(x=550,y=190)

Label(root,text= "Best Classifier and Regressor  ",font="System").place(x=250,y=350)

Button(root,text='Best Classifier',activebackground="black",command=Best_Class,activeforeground="white").place(x=250,y=390)
Button(root,text='Best Regressor',activebackground="black",command=Best_Reg,activeforeground="white").place(x=400,y=390)

root.mainloop()
