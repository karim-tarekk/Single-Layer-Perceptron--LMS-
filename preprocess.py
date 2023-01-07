import pandas as pd
import numpy as np


def preprocess(): # Function to Read dataset and count Male and Female in each class to replace Null values with the one which has more occured in that class
    # after that replace Male with 1 and Female with -1
    df = pd.read_csv("penguins.csv")
    AdelieClass = df[(df['species'] == 'Adelie')]
    GentooClass = df[(df['species'] == 'Gentoo')]
    ChinstrapClass = df[(df['species'] == 'Chinstrap')]
    AdelieMaleCtn = AdelieClass['gender'].value_counts()["male"]
    AdelieFemaleCtn = AdelieClass['gender'].value_counts()["female"]
    GentooMaleCtn = GentooClass['gender'].value_counts()["male"]
    GentooFemaleCtn = GentooClass['gender'].value_counts()["female"]
    ChinstrapMaleCtn = ChinstrapClass['gender'].value_counts()["male"]
    ChinstrapFemaleCtn = ChinstrapClass['gender'].value_counts()["female"]
    if AdelieMaleCtn > AdelieFemaleCtn:
        AdelieClass.fillna("male", inplace=True)
    else:
        AdelieClass.fillna("female", inplace=True)
    if GentooMaleCtn > GentooFemaleCtn:
        GentooClass.fillna("male", inplace=True)
    else:
        GentooClass.fillna("female", inplace=True)
    if ChinstrapMaleCtn > ChinstrapFemaleCtn:
        ChinstrapClass.fillna("male", inplace=True)
    else:
        ChinstrapClass.fillna("female", inplace=True)
    AdelieClass.replace("male", 1, inplace=True)
    AdelieClass.replace("female", -1, inplace=True)
    GentooClass.replace("male", 1, inplace=True)
    GentooClass.replace("female", -1, inplace=True)
    ChinstrapClass.replace("male", 1, inplace=True)
    ChinstrapClass.replace("female", -1, inplace=True)
    return AdelieClass, GentooClass, ChinstrapClass


def fitdata (x1, x2, class1, class2): # Function for returning train [data, label] and  test [data, label] for the required classes with required Features Selected by user
    # [NOTE] ----> c1 = "Adelie" , c2 = "Gentoo", c3 = "Chinstrap"
    AdelieClass, GentooClass, ChinstrapClass = preprocess()
    
    if class1 == "c1" and class2 == "c2": # check if c1 and c2 selected then get first 30 entries in c1 class and add first 30 entries then shuffle dataframe
        # replace C1 with -1 and C2 with 1 return labels in Trainlabels data in TrainData
        # get last 20 entry in c1 and c2 and add them to TestSet also replace C1 with -1 and C2 with 1
        # return labels in TestLabels data in TestData
        # These all applied for other classes
        # Train Data
        TrainSet = AdelieClass[:30]
        TrainSet = TrainSet.append(GentooClass[:30], ignore_index=True)
        TrainSet = TrainSet.sample(frac=1).reset_index(drop=True)
        TrainLables = TrainSet["species"]
        TrainLables.replace("Adelie", -1, inplace=True)
        TrainLables.replace("Gentoo", 1, inplace=True)
        TrainData = TrainSet[[x1, x2]]
        ################################################################################################################################
        # Test Data
        TestSet = AdelieClass[30:50]
        TestSet = TestSet.append(GentooClass[30:50], ignore_index=True)
        TestLabels = TestSet["species"]
        TestLabels.replace("Adelie", -1, inplace=True)
        TestLabels.replace("Gentoo", 1, inplace=True)
        TestData = TestSet[[x1, x2]]
    elif class1 == "c1" and class2 == "c3":
        # Train Data
        TrainSet = AdelieClass[:30]
        TrainSet = TrainSet.append(ChinstrapClass[:30], ignore_index=True)
        TrainSet = TrainSet.sample(frac=1).reset_index(drop=True)
        TrainLables = TrainSet["species"]
        TrainLables.replace("Adelie", -1, inplace=True)
        TrainLables.replace("Chinstrap", 1, inplace=True)
        TrainData = TrainSet[[x1, x2]]
        ################################################################################################################################
        # Test Data
        TestSet = AdelieClass[30:50]
        TestSet = TestSet.append(ChinstrapClass[30:50], ignore_index=True)
        TestLabels = TestSet["species"]
        TestLabels.replace("Adelie", -1, inplace=True)
        TestLabels.replace("Chinstrap", 1, inplace=True)
        TestData = TestSet[[x1, x2]]
    elif class1 == "c2" and class2 == "c3":
        # Train Data
        TrainSet = GentooClass[:30]
        TrainSet = TrainSet.append(ChinstrapClass[:30], ignore_index=True)
        TrainSet = TrainSet.sample(frac=1).reset_index(drop=True)
        TrainLables = TrainSet["species"]
        TrainLables.replace("Gentoo", -1, inplace=True)
        TrainLables.replace("Chinstrap", 1, inplace=True)
        TrainData = TrainSet[[x1, x2]]
        ################################################################################################################################
        # Test Data
        TestSet = GentooClass[30:50]
        TestSet = TestSet.append(ChinstrapClass[30:50], ignore_index=True)
        TestLabels = TestSet["species"]
        TestLabels.replace("Gentoo", -1, inplace=True)
        TestLabels.replace("Chinstrap", 1, inplace=True)
        TestData = TestSet[[x1, x2]]
    return TrainLables, TrainData, TestLabels, TestData








