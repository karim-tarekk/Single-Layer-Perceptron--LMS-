import matplotlib.pyplot as plt
from preprocess import *
import math 
import random


def visualG(f1,f2): # Function to check 2 features with all classes used to do document
    # [NOTE] -----> Black For GentooClass, Green For AdelieClass, Red For ChinstrapClass
    AdelieClass, GentooClass, ChinstrapClass = preprocess()
    x1 = AdelieClass[f1]
    x2 = GentooClass[f1]
    x3 = ChinstrapClass[f1]

    y1= AdelieClass[f2]
    y2= GentooClass[f2]
    y3= ChinstrapClass[f2]

    label = f1 + " VS " + f2
    plt.figure(label)
    plt.scatter(x1,y1 ,c="green")
    plt.scatter(x2,y2, c="black")
    plt.scatter(x3,y3, c="red")
    plt.xlabel(f1)
    plt.ylabel(f2)

    plt.show()
