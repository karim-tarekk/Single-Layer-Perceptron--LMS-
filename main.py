from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import sys
from model import *
from visual import *

class PrintConsole(): #class to print everything appers in console in a specific textbox in GUI
    def __init__(self, textbox): 
        self.textbox = textbox 

    def write(self, text):
        self.textbox.insert(tk.END, text)

    def flush(self):
        self.textbox.delete('1.0', END)


def getvalues(): # Function that read user inputs and return them to use them in initializing our model 
    featureBoxvalue1 = fcmbox1.get()
    featureBoxvalue2 = fcmbox2.get()
    classComboValue = classcombo.get()
    lrateValue = float(lrateEntry.get())
    ThresholdValue = float(ThresholdEntry.get())
    EpochValue = int(EpochsEntry.get())
    biasCheck = int(biasVar.get())
    msg = messagebox.showinfo("Done", "Done... Now learning!")
    return featureBoxvalue1, featureBoxvalue2, classComboValue, lrateValue, ThresholdValue, biasCheck, EpochValue

# VARS
accSTR = None
# Create window
frame = Tk()
frame.geometry("500x500")
frame.title("Model")
frame.configure(bg='#c7d1eb')
confusionMatrix = Text(frame,width=35, height=6)
confusionMatrix.place(x=200, y=382)
# create instance of file like object
con = PrintConsole(confusionMatrix)

# replace sys.stdout with our object
sys.stdout = con


# Used LABELS for GUI
featurevar = StringVar()
featureLabel = Label(frame, textvariable=featurevar)
featurevar.set("Select 2 features:")
featureLabel.place(x=190, y=7)

classesvar = StringVar()
classeslabel = Label(frame, textvariable=classesvar)
classesvar.set("Select a combination of 2 classes:")
classeslabel.place(x=140, y=65)

lratevar = StringVar()
lrateLabel = Label(frame, textvariable=lratevar)
lratevar.set('Enter learning rate:')
lrateLabel.place(x=80, y=139)

Epochsvar = StringVar()
EpochsLabel = Label(frame, textvariable=Epochsvar)
Epochsvar.set('Enter Epochs:')
EpochsLabel.place(x=80, y=173)

ThresholdVar = StringVar()
ThresholdLabel = Label(frame, textvariable=ThresholdVar)
ThresholdVar.set("Enter Threshold:")
ThresholdLabel.place(x=80, y=205)

biasVar = StringVar()
biasLabel = Label(frame, textvariable=biasVar)
biasVar.set("Check to add bias:")
biasLabel.place(x=136, y=248)

classNoteVar = StringVar()
classNoteLabel = Label(frame, textvariable=classNoteVar)
classNoteVar.set("C1=Adelie, C2=Gentoo, C3=Chinstrap")
classNoteLabel.place(x=260, y=93)

accuracyVar = StringVar()
accuracyLabel = Label(frame, textvariable=accuracyVar)
accuracyVar.set("Accuracy:")
accuracyLabel.place(x=10, y=390)


ConfVar = StringVar()
ConfLabel = Label(frame, textvariable=ConfVar)
ConfVar.set("Confusion Matrix:")
ConfLabel.place(x=90, y=433)

# COMBO BOXES
fcmbox1 = ttk.Combobox(frame,
                     value=('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g'),
                     state='readonly')
fcmbox1.place(x=20, y=30)

fcmbox2 = ttk.Combobox(frame,
                     value=('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g'),
                     state='readonly')
fcmbox2.place(x=300, y=30)

classcombo = ttk.Combobox(frame,
                        value=('C1 & C2', 'C1 & C3', 'C2 & C3'),
                        state='readonly')
classcombo.place(x=100, y=93)

# TEXT BOXES
lrateEntry = Entry(frame)
lrateEntry.place(x=200, y=139)

EpochsEntry = Entry(frame)
EpochsEntry.place(x=200, y=173)

ThresholdEntry = Entry(frame)
ThresholdEntry.place(x=200, y=205)

accuracyText = Text(frame, state='disabled', width=6, height=0.5)
accuracyText.place(x=78, y=390)

# CHECKBOX
biasVar = IntVar()
biasCheck = Checkbutton(frame, text="Bias", variable=biasVar)
biasCheck.place(x=250, y=245)

def StartModel(): # function for initializing model by calling getvalues function and check what classes selected by user and also convert bais Availability to true and false
    # get class names as c1,c2 and c3 then use them inside the constructor of the model then call function Epochs to train the model then use Tmodel to test the model
    # then draw Confusion Matrix inside the textbox then show the graph for 2 class
    F1, F2, ClassLabel, lrateValue, Threshold, biasCheck, epochs = getvalues()
    bs = False
    C1 = C2 = None
    if biasCheck == 1:
        bs = True
    if ClassLabel == 'C1 & C2':
        C1 = "c1"
        C2 = "c2"
    elif ClassLabel == 'C1 & C3':
        C1 = "c1"
        C2 = "c3"
    elif ClassLabel == 'C2 & C3':
        C1 = "c2"
        C2 = "c3"
    TestModel = SLPModel(lrateValue, Threshold, epochs, bs, F1, F2, C1, C2)
    TestModel.Epochs()
    TestModel.TModel()
    accSTR = TestModel.Accuracy()
    accuracyText.configure(state='normal')
    accuracyText.delete('1.0', END)
    accuracyText.insert('end', accSTR)
    accuracyText.configure(state='disabled')
    frame.after(500,con.flush())
    frame.after(1000,TestModel.ConfusionMatrix())
    TestModel.Graph()

# CONFIRMATION BUTTON
confirmB = Button(frame, text="Run", command=StartModel, width=15, height=2, state=tk.ACTIVE, activebackground='green',font= ('Helvetica 13 bold'))
confirmB.place(x=160, y=320)


frame.mainloop()

# if ClassLabel.




