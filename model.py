from logging import basicConfig
from preprocess import *
import random
import matplotlib.pyplot as plt
import math

# Start of Model Class

class SLPModel:

    #initial values for used values
    eta = TruePositive = TrueNegative = AllPositive = Min = Max = NumEpochs = 0
    Threshold = None
    SumSquared = 0
    bais = weight1 = weight2 = 0
    BaisControl = WeightsChange = False
    x1 = x2 = cl1 = cl2 = TL = TD = TesL = TesD = None
    # Constructor for Model that get LearningRate,number of Epochs, bais Availability
    def __init__(self, LearningRate, thres, Epo, baisAva, Feature1Name, Feature2Name, class1, class2):
        self.eta = LearningRate
        self.Threshold = thres
        self.BaisControl = baisAva
        self.x1 = Feature1Name
        self.x2 = Feature2Name
        self.cl1 = class1
        self.cl2 = class2
        self.NumEpochs = Epo
        self.bais = round(random.uniform(-1, 1), 2)
        self.weight1 = round(random.uniform(-1, 1), 2)
        self.weight2 = round(random.uniform(-1, 1), 2)
        self.TL, self.TD, self.TesL, self.TesD = fitdata(self.x1, self.x2 , self.cl1, self.cl2)

    def Epochs(self): # Function for Epochs get every row and its target and call Model Function after looping over whole train data check if WeightsChange == false then no Weights changed
        for i in range(self.NumEpochs):
            for j in range(self.TL.count()): 
                row = self.TD.iloc[j] 
                target = self.TL[j]
                self.Model(row, target)
            self.SumSquared = self.ErrorSum()
            LMSvalue = self.LMS()
            if LMSvalue < self.Threshold:
                break

    def ErrorSum(self): # Function For Calculating Error Sum
        sum = 0
        for i in range(self.TL.count()): 
            row = self.TD.iloc[i] 
            target = self.TL[i]
            predicted = self.CalculateOutput(row[0],row[1])
            loss = self.CalculateLoss(predicted,target)
            sum += 0.5 * (loss * loss)
        return sum

    def LMS(self):
        return self.SumSquared / self.TL.count()
    
    def Model(self, X_i, T_i): # Function for our SLP first calculate output then apply signum function to get if it is -1 or 1 
        # then check if predicted == target if yes then calculate loss between predicted and target values
        # if bais is Available then we have to update bais also else only weights got updated
        p_i = self.CalculateOutput(X_i[0],X_i[1])
        if p_i != T_i:
            los = self.CalculateLoss(p_i,T_i)
            if self.BaisControl == True:
                self.weight1 = self.UpdateWeight(los, X_i[0], self.weight1)
                self.weight2 = self.UpdateWeight(los, X_i[1], self.weight2)
                self.bais = self.UpdateWeight(los, 1, self.bais)
            else:
                self.weight1 = self.UpdateWeight(los, X_i[0], self.weight1)
                self.weight2 = self.UpdateWeight(los, X_i[1], self.weight2)

    def TModel(self): # Function for Training the model get every row and its target then calculate output then apply signum function to get if it is -1 or 1 
        # If and else if for calculating TP and TN for Confusion Matrix 
        # also AllPositive for calculating Accuracy
        for i in range(self.TesL.count()):
            row = self.TesD.iloc[i] 
            target = self.TesL[i]
            output = self.CalculateOutput(row[0],row[1])
            p_i = self.Signum(output)
            if p_i == target == -1:
                self.TruePositive += 1
            elif p_i == target == 1:
                self.TrueNegative += 1
            if p_i == target:
                self.AllPositive += 1

    def CalculateOutput(self, feature1Value, feature2Value): # Function to Calculate Output by using feature1 and feature2 values
        # if bais is Available then we use bais with our equation
        if self.BaisControl == True:
           return  (self.weight1 * feature1Value) + (self.weight2 * feature2Value) + self.bais
        else:
           return  (self.weight1 * feature1Value) + (self.weight2 * feature2Value)

    def Signum(self, value): # Function for Signum Positive or Zero values will be 1 else will be -1
        if value >= 0: 
            return 1 
        else: 
            return - 1

    def CalculateLoss(self, P, T): # Function for calculating Loss where target - Predicted
        return T - P

    def UpdateWeight(self, loss, x, oldW):
        return oldW + (self.eta * loss * x)

    def createMatrix(self,size):
        # This Function Creates the matrix with "-" in each cell
        matrix = [] # make the matrix empty
        # the nested for loops creates the matrix and add "-" in each cell
        for i in range(size ):
            matrix2 = []
            for j in range(size):
                matrix2.append("-")
            matrix.append(matrix2)
        return matrix

    def ConfusionMatrix(self): # Function For adding Confusion Matrix data
        mat = self.createMatrix(3)
        mat[0][0]= "A|P"
        mat[1][0] = mat[0][1] = self.cl1
        mat[2][0] = mat[0][2] = self.cl2
        mat[1][1] = self.TruePositive
        mat[2][2] = self.TrueNegative
        FN = (self.TesL.count()/2) - self.TruePositive 
        FP = (self.TesL.count()/2) - self.TrueNegative
        mat[1][2] = FN
        mat[2][1] = FP
        self.printMatrix(mat)

    def printMatrix(self, matrix):  # Function to print Confusion Matrix 
        for z in range(len(matrix)):
            print(matrix[z])
        print()

    def Accuracy(self): # Function to Calculate Accuracy
        acc = (self.AllPositive / self.TesL.count()) * 100
        return acc

    def Graph(self): # Function to Display graph of selected classes on which each class on an axis with hyperplane that separets between 2 classes
        self.LineRange()
        c11 = self.TesD[:20][self.x1]
        c12 = self.TesD[:20][self.x2]
        c21 = self.TesD[20:][self.x1]
        c22 = self.TesD[20:][self.x2]
        text = self.cl1 +" VS "+self.cl2 + " by using feature 1: "+ self.x1 + " and feature 2:"+self.x2
        plt.figure(text)
        plt.scatter(c11,c12,c="Red")
        plt.scatter(c21,c22,c="black")
        plt.xlabel(self.cl1)
        plt.ylabel(self.cl2)
        x = np.array(range(self.Min,self.Max))
        y = self.LineEquation(x)
        plt.plot(x,y, c="blue")
        plt.show()

    def LineEquation(self,x): # Function of the Equation of hyperplane that separets between 2 classes
           return (-(self.weight1 / self.weight2)* x) -( self.bais / self.weight2)

    def LineRange(self): # Function to get Min and max points to apply them in LineEquation to plot hyperplane
        min1 = min(self.TesD[self.x1])
        min2 = min(self.TesD[self.x2])
        if min1 > min2:
            self.Min = int(min2)
        else:
            self.Min = int(min1)
        max1 = max(self.TesD[self.x1])
        max2 = max(self.TesD[self.x2])
        if max1 > max2:
            self.Max = int(max1)
        else:
            self.Max = int(max2)