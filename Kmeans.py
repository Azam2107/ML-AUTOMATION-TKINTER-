import tkinter as tk
from tkinter import *
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,r2_score
from sklearn.impute import SimpleImputer
from tkinter.filedialog import askopenfilename
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


from sklearn.cluster import KMeans

feature_col =[]
target_col = []
labs=None
y_kmeans=None

root= Tk()
root.title('Kmeans')
root.geometry('800x750')

def data():
    global filename
    filename = askopenfilename(initialdir=r'C:\Users\surya\Desktop\CDAC Noida\ML\Files',title = "Select file")
    e1.insert(0, filename)
    e1.config(text=filename)

    global file
    file = pd.read_csv(filename)
    for i in file.columns:
        box1.insert(END,i)

    for i in file.columns:
        if type(file[i][0]) == np.float64 :
            file[i].fillna(file[i].mean(), inplace=True)
        elif type(file[i][0]) == np.int64 :  
            file[i].fillna(file[i].median(), inplace=True)
        elif type(file[i][0]) == type(""):
            imp_ = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            s = imp_.fit_transform(file[i].values.reshape(-1, 1))
            file[i] = s

    colss=file.columns
    global X_Axis
    X_Axis = StringVar()
    X_Axis.set('X-axis')
    choose = ttk.Combobox(root, width=22, textvariable=X_Axis)
    choose['values'] = (tuple(colss))
    choose.place(x=400, y=20)

    global Y_Axis
    Y_Axis = StringVar()
    Y_Axis.set('Y-axis')
    choose = ttk.Combobox(root, width=22, textvariable=Y_Axis)

    choose['values'] = (tuple(colss))
    choose.place(x=400, y=40)
    global graphtype
    graphtype = StringVar()
    graphtype .set('Graph')
    choose = ttk.Combobox(root, width=22, textvariable=graphtype)
    choose['values'] = ('scatter','line','bar','hist','corr','pie',"clustered")
    choose.place(x=400, y=60)

def getx():
    x_v = []
    s = box1.curselection()
    global feature_col
    for i in s:
        if i not in feature_col:
            feature_col.append((file.columns)[i])
            x_v = feature_col
    for i in x_v:
        box2.insert(END,i)


def gety():
    y_v = []
    global target_col
    s = box1.curselection()
    for j in s:
        if j not in target_col:
            target_col.append((file.columns)[j])
            y_v=target_col

    for i in y_v:
        box3.insert(END,i)


def plot():

    fig = Figure(figsize=(6,6), dpi=70)
    global X_Axis
    global Y_Axis
    global graphtype
    u=graphtype.get()

    if u=='scatter':
        plot1 = fig.add_subplot(111)
        plt.scatter(file[X_Axis.get()], file[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u=='line':
        plot1 = fig.add_subplot(111)
        plt.plot(file[X_Axis.get()], file[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u=='bar':
        plot1 = fig.add_subplot(111)
        plt.bar(file[X_Axis.get()], file[Y_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u=='hist':
        plot1 = fig.add_subplot(111)
        plt.hist(file[X_Axis.get()])
        plt.xlabel(X_Axis.get())
        plt.ylabel(Y_Axis.get())
        plt.show()

    if u=='corr':
        plot1 = fig.add_subplot(111)
        sns.heatmap(file.corr())
        plt.show()

    if u=='pie':
        plot1 = fig.add_subplot(111)
        plt.pie(file[Y_Axis.get()].value_counts(),labels=file[Y_Axis.get()].unique())
        plt.show()
    
        
    if u=='clustered':
        plot1 = fig.add_subplot(111)
        plt.scatter(file[X_Axis.get()], file[Y_Axis.get()],c=labs)
        plt.show()

def elbow():
    
    fig = Figure(figsize=(6,6), dpi=70)
    global X_Axis
    global Y_Axis
    global graphtype
    u=graphtype.get()
    plottr=[]
    for i in range(1,11):
        kmeans=KMeans(n_clusters=i,init="k-means++",max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(file[feature_col])
        plottr.append(kmeans.inertia_)

    plot1 = fig.add_subplot(111)
    
    plot1=plt.plot(range(1, 11), plottr)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('plottr') 
    plt.show()

def model():

    x = file[feature_col]
    
    kmeans = KMeans(n_clusters = int(n_clus.get()), init = 'k-means++',algorithm=algo.get(), max_iter = 300, n_init = 10, random_state = 0)
    y_kmeans = kmeans.fit_predict(x)
    global labs
    labs=kmeans.labels_

    accuracy = silhouette_score(x,kmeans.labels_)

    Label(root,text=f'accuracy : {accuracy}', font=('Helvetica', 10, 'bold'), bg="light blue", relief="solid").place(x=20,y=550)
    

    return accuracy,None

def files():
    with open(r"C:\Users\musan\Desktop\model summary","w",encoding="utf-8") as file:
        file.write("You have use Kmeans model \n")
        file.write("\n")
        file.write(f"The columns used for clustering are {feature_col} \n")
        file.write("\n")
        file.write(f"The Hyper parameters used in the model are initiated as No. of centroids - {int(n_clus.get())} , algorithm - {algo.get()}\n")
        file.write("\n")
        file.write(f"The accuracy of the model is {model()[0]}")
        file.write("\n")

     

listbox=Listbox(root,selectmode="multiple")
listbox.pack

n_clus=tk.StringVar()
choose=ttk.Combobox(root,width=30,textvariable=n_clus)
choose['values']=('1','2','3','4')
choose.place(x=200,y=360)
Label(root,font="System",text="No. of centroids").place(x=20,y=360)

algo=tk.StringVar()
choose=ttk.Combobox(root,width=30,textvariable=algo)
choose['values']=('auto','full','elkan')
choose.place(x=200,y=390)
Label(root,font="System",text="algorithm").place(x=20,y=390)

l1=Label(root, text='Select Data File')
l1.grid(row=0, column=0)
e1 = Entry(root,text='')
e1.grid(row=0, column=1)
Button(root,text='open', command=data,activeforeground="white",activebackground="black").grid(row=0, column=2)

box1 = Listbox(root,selectmode='multiple')
box1.grid(row=10, column=0)


box2 = Listbox(root)
box2.grid(row=10, column=1)
Button(root, text='Select data for kmeans', command=getx,activeforeground="white",activebackground="black").grid(row=12,column=1)

Button(root,text = "Plot",command = plot,activeforeground="white",activebackground="black").place(x=600, y=50)

Button(root,text= "plot elbow",command=elbow,activeforeground="white",activebackground="black").place(x=150,y=250)

Button(root,text="Run Model",command=model,activeforeground="white",activebackground="black").place(x=150,y=500)

Button(root,text= "Summary",command=files,activeforeground="white",activebackground="black").place(x=250,y=500)

root.mainloop()




