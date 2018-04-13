# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'GraphicalUserInterface.ui'
# Created by: PyQt4 UI code generator 4.11.4


from PyQt4 import QtCore, QtGui

import ProjectCUDA
import ClusterFormatn
import Matrix

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle

import os
import sys

if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))
elif __file__:
    os.chdir(os.path.dirname(__file__))
    
    

Image=None
ImageName=None

TrainingName=None
clusterLineInput=None
Cluster=None


CompatiblityName=None
matrixLineInput=None
matrix=None

HardImg=None

GPURelax=None

Iterations=None

textInfo=''

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
        
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))

        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(813, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        
        
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 813, 26))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        
        self.menuMenu = QtGui.QMenu(self.menubar)
        self.menuMenu.setObjectName(_fromUtf8("menuMenu"))
        
        self.menuDisplay = QtGui.QMenu(self.menubar)
        self.menuDisplay.setObjectName(_fromUtf8("menuDisplay"))
        
        self.menuOpen = QtGui.QMenu(self.menubar)
        self.menuOpen.setObjectName(_fromUtf8("menuOpen"))
        
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        MainWindow.insertToolBarBreak(self.toolBar)
        
        
        
        self.ImageDirectory = QtGui.QLabel(self.centralwidget)                  #Image Directory Label
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.ImageDirectory.setFont(font)
        self.ImageDirectory.setObjectName(_fromUtf8("ImageDirectory"))
        self.gridLayout.addWidget(self.ImageDirectory, 0, 0, 1, 1)
        
        self.DirectoryEdit = QtGui.QLineEdit(self.centralwidget)                #Directory Edit Line Edit
        self.DirectoryEdit.setObjectName(_fromUtf8("DirectoryEdit"))
        self.gridLayout.addWidget(self.DirectoryEdit, 0, 1, 1, 2)
        
        self.ImageBrowse = QtGui.QPushButton(self.centralwidget)                #Image Browse Push Button
        self.ImageBrowse.setObjectName(_fromUtf8("ImageBrowse"))
        self.gridLayout.addWidget(self.ImageBrowse, 0, 3, 1, 1)
        self.ImageBrowse.setStatusTip('Browse Image for Classification')
        self.ImageBrowse.clicked.connect(self.getDirectoryImage)
        
        self.Help = QtGui.QPushButton(self.centralwidget)                       #Help push button
        self.Help.setObjectName(_fromUtf8("Help"))
        self.Help.setStatusTip('Information on How to use the GUI')
        self.gridLayout.addWidget(self.Help, 0, 4, 1, 1)
        self.Help.clicked.connect(self.HelpClick)
        
        self.TrainingCluster = QtGui.QLabel(self.centralwidget)                 #Training Cluster Label
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.TrainingCluster.setFont(font)
        self.TrainingCluster.setObjectName(_fromUtf8("TrainingCluster"))
        self.gridLayout.addWidget(self.TrainingCluster, 1, 0, 2, 1)
        
        self.TrainingEdit = QtGui.QTextEdit(self.centralwidget)                 #Training Text Edit 
        self.TrainingEdit.setObjectName(_fromUtf8("TrainingEdit"))
        self.gridLayout.addWidget(self.TrainingEdit, 1, 1, 2, 2)

        
        self.TrainingCheck = QtGui.QCheckBox(self.centralwidget)                #Training Check Button
        self.TrainingCheck.setObjectName(_fromUtf8("TrainingCheck"))
        self.TrainingCheck.setStatusTip('Use previous data')
        self.gridLayout.addWidget(self.TrainingCheck, 1, 3, 1, 1)
        self.TrainingCheck.stateChanged.connect(self.checkBoxTraining)
        
        self.TrainingGenerate = QtGui.QPushButton(self.centralwidget)           #Training Generate Push Button
        self.TrainingGenerate.setObjectName(_fromUtf8("TrainingGenerate"))
        self.TrainingGenerate.setStatusTip('Create new cluster data')
        self.gridLayout.addWidget(self.TrainingGenerate, 2, 3, 1, 1)
        self.TrainingGenerate.clicked.connect(self.TrainingGenerateClick)
        
        self.NumberClasses = QtGui.QLabel(self.centralwidget)                   #Number of Classes label
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.NumberClasses.setFont(font)
        self.NumberClasses.setObjectName(_fromUtf8("NumberClasses"))
        self.gridLayout.addWidget(self.NumberClasses, 3, 1, 1, 1)
        
        self.ValueNumberClasses = QtGui.QLabel(self.centralwidget)              #Value of Number of Classes label 
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.ValueNumberClasses.setFont(font)
        self.ValueNumberClasses.setObjectName(_fromUtf8("ValueNumberClasses"))
        self.gridLayout.addWidget(self.ValueNumberClasses, 3, 2, 1, 1)
        
        self.CompatibilityMatrix = QtGui.QLabel(self.centralwidget)             #Compatibility Matrix Label
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.CompatibilityMatrix.setFont(font)
        self.CompatibilityMatrix.setObjectName(_fromUtf8("CompatibilityMatrix"))
        self.gridLayout.addWidget(self.CompatibilityMatrix, 4, 0, 2, 1)
        
        self.CompatibilityEdit = QtGui.QTextEdit(self.centralwidget)            #Compatibility Matrix Text Edit 
        self.CompatibilityEdit.setObjectName(_fromUtf8("CompatibilityEdit"))
        self.gridLayout.addWidget(self.CompatibilityEdit, 4, 1, 2, 2)
        
        self.CompatibilityCheck = QtGui.QCheckBox(self.centralwidget)           #Compatibility Matrix Check Button
        self.CompatibilityCheck.setObjectName(_fromUtf8("CompatibilityCheck"))
        self.CompatibilityCheck.setStatusTip('Load Compatibility Matrix')
        self.gridLayout.addWidget(self.CompatibilityCheck, 4, 3, 1, 1)
        self.CompatibilityCheck.stateChanged.connect(self.checkBoxMatrix)
        
        self.CompatibilityGenerate = QtGui.QPushButton(self.centralwidget)      #Compatibility Generate Push Button
        self.CompatibilityGenerate.setObjectName(_fromUtf8("CompatibilityGenerate"))
        self.CompatibilityGenerate.setStatusTip('Create Compatibility Matrix')
        self.gridLayout.addWidget(self.CompatibilityGenerate, 5, 3, 1, 1)
        self.CompatibilityGenerate.clicked.connect(self.MatrixGenerateClick)
        
        
        
        self.Run = QtGui.QPushButton(self.centralwidget)                        #Run push button
        self.Run.setObjectName(_fromUtf8("Run"))
        self.Run.setStatusTip('Start execution')
        self.gridLayout.addWidget(self.Run, 6, 1, 1, 1)
        self.Run.clicked.connect(self.RunFunction)
        
        self.ProgramOutput = QtGui.QTextEdit(self.centralwidget)                #Program output
        self.ProgramOutput.setObjectName(_fromUtf8("ProgramOutput"))
        self.gridLayout.addWidget(self.ProgramOutput, 7, 0, 1, 5)
        
    
        
        self.csre = QtGui.QAction(QtGui.QIcon('.\CSRE.png'), 'About CSRE', MainWindow)
        self.toolBar.addAction(self.csre)
        self.csre.triggered.connect(self.CSREclick)
        
        self.actionReset = QtGui.QAction(MainWindow)
        self.actionReset.setObjectName(_fromUtf8("actionReset"))
        
        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.actionQuit.setShortcut("Ctrl+Q")
        self.actionQuit.setStatusTip('Leave The App')
        self.actionQuit.triggered.connect(self.closeApplication)
        
        self.actionInformation = QtGui.QAction(MainWindow)
        self.actionInformation.setObjectName(_fromUtf8("actionInformation"))
        self.actionInformation.setShortcut("Ctrl+A")
        self.actionInformation.setStatusTip('Set Number of Iterations')
        self.actionInformation.triggered.connect(self.Iteration)
        
        self.actionInput_Image = QtGui.QAction(MainWindow)
        self.actionInput_Image.setObjectName(_fromUtf8("actionInput_Image"))
        self.actionInput_Image.setShortcut("Ctrl+I")
        self.actionInput_Image.setStatusTip('Display Image File')
        self.actionInput_Image.triggered.connect(DisplayImage)
        
        self.actionMaximum_Likelihood = QtGui.QAction(MainWindow)
        self.actionMaximum_Likelihood.setObjectName(_fromUtf8("actionMaximum_Likelihood"))
        self.actionMaximum_Likelihood.setShortcut("Ctrl+M")
        self.actionMaximum_Likelihood.setStatusTip('Display Maximum Likelihood Classification')
        self.actionMaximum_Likelihood.triggered.connect(DisplayMaxLikeImage)
        
        self.actionRelaxation_Labelling = QtGui.QAction(MainWindow)
        self.actionRelaxation_Labelling.setObjectName(_fromUtf8("actionRelaxation_Labelling"))
        self.actionRelaxation_Labelling.setShortcut("Ctrl+R")
        self.actionRelaxation_Labelling.setStatusTip('Display Relaxation Labelling Classification')
        self.actionRelaxation_Labelling.triggered.connect(DisplayRelaxImage)
        
        self.actionSource_Code = QtGui.QAction(MainWindow)
        self.actionSource_Code.setObjectName(_fromUtf8("actionSource_Code"))
        
        self.menuMenu.addAction(self.actionReset)
        self.menuMenu.addAction(self.actionQuit)
        self.menuMenu.addSeparator()
        self.menuMenu.addAction(self.actionInformation)
        self.menuDisplay.addAction(self.actionInput_Image)
        self.menuDisplay.addAction(self.actionMaximum_Likelihood)
        self.menuDisplay.addAction(self.actionRelaxation_Labelling)
        self.menuOpen.addAction(self.actionSource_Code)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuDisplay.menuAction())
        self.menubar.addAction(self.menuOpen.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
#################################################### Associated with Image loading ##########################################        
    def getDirectoryImage(self):
        global ImageName
        global textInfo    
        ImageName = QtGui.QFileDialog.getOpenFileName(MainWindow, 'Open File',os.getcwd(),'*.jpg; *.png; *.tif')
        self.DirectoryEdit.setText(str(ImageName))
        self.imgDim=ReadImage(ImageName)
        textInfo+=('Image Dimensions : '+str(self.imgDim)+'\n')
        self.ProgramOutput.setText(textInfo)
        
#################################################### Associated with Training Clusters ##########################################         
        
    def getDirectoryTraining(self):   
        global TrainingName 
        global Cluster
        global textInfo
        TrainingName = QtGui.QFileDialog.getOpenFileName(MainWindow, 'Open File',os.getcwd(),'*.csv')
        self.TrainingEdit.setText(str(TrainingName))
        [Cluster, self.classDim]=ProjectCUDA.read_file(TrainingName)
        textInfo+= ('Classify Image dimesions : '+str(self.classDim)+'\n')
        self.ProgramOutput.setText(textInfo)
        self.ValueNumberClasses.setText(str(self.classDim[-1]))
               
        
    def checkBoxTraining(self,state):
        if(state == QtCore.Qt.Checked):
            self.getDirectoryTraining()
            self.TrainingGenerate.setEnabled(False)
        else:
            self.TrainingEdit.setText('')
            self.TrainingGenerate.setEnabled(True)
            
            
    def TrainingGenerateClick(self):
        try:
            eval(self.TrainingEdit.toPlainText()) 
            if(type(eval(self.TrainingEdit.toPlainText()))!=dict):
                QtGui.QMessageBox.information(MainWindow, 'Incorrect Input', "Input type not dictionary", QtGui.QMessageBox.Ok) 
                self.TrainingEdit.setText('')    
            else:
                global clusterLineInput
                global Cluster
                global textInfo
                clusterLineInput=eval(self.TrainingEdit.toPlainText())
                ClusterFormatn.ClusterFormation(Image,clusterLineInput)     
                [Cluster, self.classDim]=ProjectCUDA.read_file('Cluster.csv')
                textInfo+= ('Classify Image dimesions : '+str(self.classDim)+'\n')
                self.ProgramOutput.setText(textInfo)
                self.ValueNumberClasses.setText(str(self.classDim[-1]))     
            
        except:
            QtGui.QMessageBox.information(MainWindow, 'Invalid Input', "Check input formatting", QtGui.QMessageBox.Ok)  
            
        
            
            
#################################################### Associated with Compatibility Matrix ##########################################             
            
    
    def getDirectoryMatrix(self):   
        global CompatibilityName 
        global matrix
        global textInfo
        CompatibilityName = QtGui.QFileDialog.getOpenFileName(MainWindow, 'Open File',os.getcwd(),'*.pkl')
        self.CompatibilityEdit.setText(str(CompatibilityName))
        pickle_in=open(CompatibilityName,'rb')
        matrix=pickle.load(pickle_in)
        textInfo+= ('\nCompatibility Matrix :\n'+str(matrix)+'\n')
        self.ProgramOutput.setText(textInfo)
    
    
    def checkBoxMatrix(self,state):
        if(state == QtCore.Qt.Checked):
            self.getDirectoryMatrix()
            self.CompatibilityGenerate.setEnabled(False)
        else:
            self.CompatibilityEdit.setText('')
            self.CompatibilityGenerate.setEnabled(True)        
            
            
    def MatrixGenerateClick(self):
        try:
            eval(self.CompatibilityEdit.toPlainText()) 
            if(type(eval(self.CompatibilityEdit.toPlainText()))!=list):
                QtGui.QMessageBox.information(MainWindow, 'Incorrect Input', "Input type not List", QtGui.QMessageBox.Ok) 
                self.CompatibilityEdit.setText('')
            else:
                global matrixLineInput
                global matrix
                global textInfo
                matrixLineInput=eval(self.CompatibilityEdit.toPlainText())
                matrix=Matrix.RelaxMatrix(matrixLineInput) 
                if((np.shape(matrix)[0]==np.shape(matrix)[1]) and (np.shape(matrix)[0]==self.classDim[-1])):    
                    textInfo+= ('\nCompatibility Matrix :\n'+str(matrix)+'\n')
                    self.ProgramOutput.setText(textInfo)
                else:
                    QtGui.QMessageBox.information(MainWindow, 'Incorrect Input', "Dimensions do not match or dimension not equal to number of Classes", QtGui.QMessageBox.Ok) 
                    self.CompatibilityEdit.setText('')
                    matrix=None     
            
        except:
            QtGui.QMessageBox.information(MainWindow, 'Invalid Input', "Check input formatting", QtGui.QMessageBox.Ok)
            
                
            
#################################################### Associated with RUN ##########################################     

    
    def RunFunction(self): 
        global HardImg
        global Iterations
        global GPURelax
        global textInfo
        import time
        if(Iterations==None):
            QtGui.QMessageBox.information(MainWindow, 'Set Iterations', "Set the value in Menu --> Iterations", QtGui.QMessageBox.Ok)
        else:
            self.start=time.time()  
            textInfo+=('\nComputing Maximum Likelihood Probabilities and Relaxation Labelling\n(Advanced Information available in console output)\n') 
            self.ProgramOutput.setText(textInfo) 
            self.maxLike=ProjectCUDA.MaximumLikelihood(Cluster,Image)
            [HardImg,ClassImg_GPU]=self.maxLike.Compute()
            ProjectCUDA.display_image(HardImg[:,:,:],'Hard Class')
            self.relaxLabel=ProjectCUDA.RelaxationLabelling(matrix)
            for i in range(Iterations):
                [GPURelax,RelaxImg_GPU]=self.relaxLabel.Compute(ClassImg_GPU)
                ClassImg_GPU=RelaxImg_GPU
                label='Relax Class '+str(i+1)
            GPURelax=self.maxLike.HardDecision(GPURelax[:,:,:])
            ProjectCUDA.display_image(GPURelax,label)
            self.end=time.time()
            textInfo+=('\nDone. Time taken: %f seconds'%(self.end-self.start))
            self.ProgramOutput.setText(textInfo)
            plt.show()
#################################################### Set Iteration #################################################         
        
     
    def Iteration(self):
        form = Iteration()
        #form.show()
        form.exec_()       
                
                    
#################################################### CSRE ################################################# 


    def CSREclick(self):
        QtGui.QMessageBox.information(MainWindow, 'About CSRE', "Centre of Studies in Resources Engineering\nIndian Institiute of Technology Bombay\nAuthor: Megh Shukla\nGuide: Prof BK Mohan\nSubmitted as a part of Course Work Project for Advanced Satellite Image Processing\n", QtGui.QMessageBox.Ok)
        
        

#################################################### Set Iteration ################################################# 


    def HelpClick(self):
        QtGui.QMessageBox.information(MainWindow, 'Help', "Steps to follow:\n1. Load Image using browse button\n\n2. If training cluster exists, check the browse box and load it. Otherwise, enter values in the form of dictionary, where class values form keys and for each class, we will have list specifying: [ topLeftX, topLeftY, bottomRightX, bottomRightY] coordinates. For each class value enclose all clusters within '[ ]'. Eg : {0: [[720,300,755,380],[585,330,602,365]]\n\n3. If compatibility matrix exists, check the browse box and load. Otherwise, enter values in the form of list, with each row of matrix enclosed im '[ ]', and all the rows enclosed in a set of '[ ]'.\n\n4. Clicking on generate defines the data set for processing.\n\n5. Run will initiate Maximum Likelihood and Relaxation Labelling classification\n\nYou can view results of Classification be clicking Display and selecting appropriate action from the Menu tab}", QtGui.QMessageBox.Ok)


#################################################### Close application #################################################                            
            
    def closeApplication(self):
        self.choice = QtGui.QMessageBox.question(MainWindow, 'Exit', "Are you sure you want to exit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if self.choice == QtGui.QMessageBox.Yes:
            print("Exiting...")
            sys.exit()
        else:
            pass    

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Relaxation Labelling", None))
        MainWindow.setWindowIcon(QtGui.QIcon('.\IITB.png'))
        
        self.ImageDirectory.setText(_translate("MainWindow", "Image Directory         :", None))
        self.ImageBrowse.setText(_translate("MainWindow", "Browse", None))
        self.Help.setText(_translate("MainWindow", "Help", None))
        self.TrainingCluster.setText(_translate("MainWindow", "Training Cluster         :", None))
        self.TrainingCheck.setText(_translate("MainWindow", "Browse", None))
        self.TrainingGenerate.setText(_translate("MainWindow", "Generate", None))
        self.NumberClasses.setText(_translate("MainWindow", "Number of Classes Detected : ", None))
        self.ValueNumberClasses.setText(_translate("MainWindow", "0", None))
        self.CompatibilityMatrix.setText(_translate("MainWindow", "Compatibility Matrix : ", None))
        self.CompatibilityCheck.setText(_translate("MainWindow", "Browse", None))
        self.CompatibilityGenerate.setText(_translate("MainWindow", "Generate", None))
        self.Run.setText(_translate("MainWindow", "Run", None))
        
        self.menuMenu.setTitle(_translate("MainWindow", "Menu", None))
        self.menuDisplay.setTitle(_translate("MainWindow", "Display", None))
        self.menuOpen.setTitle(_translate("MainWindow", "Open", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "Tool Bar", None))
        self.actionReset.setText(_translate("MainWindow", "Reset", None))
        self.actionQuit.setText(_translate("MainWindow", "Quit", None))
        self.actionInformation.setText(_translate("MainWindow", "Iterations", None))
        self.actionInput_Image.setText(_translate("MainWindow", "Input Image", None))
        self.actionMaximum_Likelihood.setText(_translate("MainWindow", "Maximum Likelihood", None))
        self.actionRelaxation_Labelling.setText(_translate("MainWindow", "Relaxation Labelling", None))
        self.actionSource_Code.setText(_translate("MainWindow", "Source Code", None))
        
        
class Iteration(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Iteration, self).__init__(parent)

        self.le = QtGui.QLineEdit()
        self.le.setObjectName("Iteration")
        self.le.setText("")

        self.pb = QtGui.QPushButton()
        self.pb.setObjectName("Ok")
        self.pb.setText("OK") 

        layout = QtGui.QFormLayout()
        layout.addWidget(self.le)
        layout.addWidget(self.pb)

        self.setLayout(layout)
        self.pb.clicked.connect(self.button_click)
        self.setWindowTitle("Set iterations")

    def button_click(self):
        global Iterations
        try:
            Iterations=int((self.le.text()))
            self.close()
        except:
            QtGui.QMessageBox.information(self, 'Invalid Input', "Check input", QtGui.QMessageBox.Ok)
       
        
        
        
def ReadImage(label):
    global Image
    global textInfo
    if(label!=''):
        [Image,image_dimensions]=ProjectCUDA.read_image(label)
        return image_dimensions
    
def DisplayImage():                                                             #If no label specified by browse, reads the last opened file
    if(ImageName!=None or ImageName!=''):
        ProjectCUDA.display_image(Image[:,:,:],'Input Image')
        plt.show()

        
def DisplayMaxLikeImage():                                                             #If no label specified by browse, reads the last opened file
    if(type(HardImg)==type(None)):
        pass
    else:
        ProjectCUDA.display_image(HardImg[:,:,:],'Maximum Likelihood Image')
        plt.show()

                    

def DisplayRelaxImage():                                                             #If no label specified by browse, reads the last opened file
    if(type(GPURelax)==type(None)):
        pass
    else:
        ProjectCUDA.display_image(GPURelax[:,:,:],'Relaxation Labelling Image')
        plt.show()



if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

