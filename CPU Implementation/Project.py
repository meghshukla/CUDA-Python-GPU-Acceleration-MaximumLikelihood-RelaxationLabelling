import pickle
from osgeo import gdal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as clr
from gdalconst import *
import ClusterFormatn
import time

classColours=[[0, 0, 255],[0, 255, 0],[0, 255, 255],[255, 255, 0], [255, 255, 255]]
img_dim=None

class MaximumLikelihood(object):
    def __init__(self,cluster,image):
        self.Cluster=cluster
        self.image_=image
        
    def Mean(self):                                                            #Mu is in form np. matrix, shape : clusters x bands
        self.Mu=[]
        for i in range(len(self.Cluster)):
            self.Mean_=[]
            for j in self.Cluster[i].columns:
                self.Mean_.append(np.mean(self.Cluster[i][j]))
            self.Mu.append(self.Mean_)
        self.Mu=np.matrix(self.Mu,dtype=np.float32)        
        del i,j    
        
        print("Mean : "+str(np.shape(self.Mu)))
        print(self.Mu)
        print("Inverse Covariance : "+str(np.shape(self.covMatrix)))
        print(self.covMatrix)
        print("Det : "+str(np.shape(self.classDet)))
        print(self.classDet)
        
        
    def Covariance(self):
        self.CovMatrix=[]
        self.classDet=[]
        for i in range(len(self.Cluster)):
            self.Cluster[i].drop('Class',axis=1,inplace=True)
            self.CovMatrix.append(np.cov(self.Cluster[i],rowvar=False)) 
            self.classDet.append(np.linalg.det(self.CovMatrix[i])) 
            
        self.CovMatrix=list(map(np.linalg.inv,self.CovMatrix))
        self.covMatrix=np.zeros((img_dim[2],img_dim[2],len(self.CovMatrix)),dtype=np.float32)
        for j in range(len(self.CovMatrix)):  
            self.covMatrix[:,:,j]=np.matrix(self.CovMatrix[j],dtype=np.float32) 
        self.classDet=list(map(np.sqrt,self.classDet))   
        self.classDet=np.array(self.classDet,dtype=np.float32)   
        del i,j,self.CovMatrix
        
        
    def Gaussian(self,pixels,mu,cov,det):
        self.MuDiff=pixels-mu  
        self.MuDiff=np.asmatrix(self.MuDiff)
        self.Exp1=(self.MuDiff*(cov)*(self.MuDiff.transpose()))
        self.Exp=pow(np.e,-0.5*(self.Exp1[0,0]))
        self.denom=(pow(2*(np.pi),(len(mu)/2.0)))*det
        return (self.Exp/self.denom)    
        
        
    def Classify(self):
        self.Classified_=np.zeros(np.shape(self.image_)[:-1]+(len(self.Cluster),))
        for i in range(np.shape(self.image_)[0]):
            print(i)
            for j in range(np.shape(self.image_)[1]):
                for k in range(len(self.Cluster)):
                    self.Classified_[i,j][k]=self.Gaussian(self.image_[i,j],self.Mu[k,:],self.covMatrix[:,:,k],self.classDet[k])
                if(i==530 and j==160):
                    print('Class probability values : '+str(self.Classified_[i,j]))
                
        del i,j,k
        
        
    def HardDecision(self,CLASSimg):
        self.Hard_=np.zeros(np.shape(self.image_)[:-1]+(3,))
        for i in range(np.shape(self.image_)[0]):
            for j in range(np.shape(self.image_)[1]): 
                self.a=list(CLASSimg[i,j])
                self.Hard_[i,j]=classColours[self.a.index(max(self.a))]
                del self.a
        del i,j  
        return self.Hard_          
         
      
    def Compute(self):
        self.Covariance()   
        self.Mean()
        self.Classify()  
        return self.HardDecision(self.Classified_),self.Classified_
               
        
    def getMean(self):
        return self.Mu
    
    def getCovariance(self):                          
        return self.covMatrix
        
    def getClassified(self):
        return self.Classified_  
        
 
class RelaxationLabelling(object):
    def __init__(self,CompatMatrix,ClassImage,windowSize):
        self.CompatMatrix=CompatMatrix
        self.ClassImage=ClassImage
        self.RelaxedImage=self.ClassImage.copy()
        self.windowSize=windowSize
        
    def LabelEval(self,Wdw):
        self.temp=[]
        for c in range(np.shape(Wdw)[2]):                                      #Decide the class for centre pixel    
            self.Qi=0  
            for k in range(np.shape(Wdw)[0]):                                         #Iterate over all neighbours
                for l in range(np.shape(Wdw)[1]):
                    if(k!=int(self.windowSize/2) and l!=int(self.windowSize/2)):
                        #self.weight=(pow(pow(k-int(self.windowSize/2),2)+pow(l-int(self.windowSize/2),2),0.5))/self.weightTotal 
                        self.weight=1/self.weightTotal
                        self.Qij=np.matrix(Wdw[k,l])*np.matrix(self.CompatMatrix[c,:]).transpose()                                   #For each neighbour
                        self.Qi+=self.weight*self.Qij                                                                                #Collecting all neighbours together
            self.temp.append(Wdw[int(self.windowSize/2),int(self.windowSize/2)][c]*self.Qi)
            
        self.temp1=sum(self.temp)    
        for t in range(len(self.temp)):
            self.temp[t]/=self.temp1
        del c,k,l
        return list(self.temp)                
                               
        
    def Convolution(self):
        #Weight Calculation
        [k,l]=[0,0]
        self.weightTotal=0
        while(k<=self.windowSize/2):
            while(l<self.windowSize/2):
                self.weightTotal+=1
                #self.weightTotal+=pow(pow(k-int(self.windowSize/2),2)+pow(l-int(self.windowSize/2),2),0.5)
                l+=1
            k+=1    
        self.weightTotal*=4.0
        del k,l
        
        for i in range(np.shape(self.ClassImage)[0]-self.windowSize+1):
            print(i)
            for j in range(np.shape(self.ClassImage)[1]-self.windowSize+1):   
                self.window=self.ClassImage[i:i+self.windowSize,j:j+self.windowSize,:]
                self.RelaxedImage[i+int(self.windowSize/2),j+int(self.windowSize/2)]=self.LabelEval(self.window)  
        return self.RelaxedImage               


def read_image():
    global img_dim
    imgPath='E:/IIT Bombay/IITB/Sem2/ASIP/powai-ikonos.jpg'
    dataset = gdal.Open(imgPath, GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    Img=np.zeros([rows,cols,bands])
    for i in range(bands):
        band = dataset.GetRasterBand(i+1)
        Img[:,:,i] = band.ReadAsArray(0, 0, cols, rows).astype(np.float)
    #Resize purposes for fitting into GUI     
    #img=cv2.resize(Img, (0,0), fx=0.25, fy=0.25)
    Img=Img[450:1201,1100:2201,:]
    img_dim=np.shape(Img)
    return Img
    
def display_image(image,string): 
    Norm = clr.Normalize(vmin = np.min(image), vmax = np.max(image), clip = False) 
    data_disp=Norm(image) 
    plt.figure()      
    plt.imshow(data_disp)
    plt.title(string)    
    plt.show()   

def read_file(string):
    cluster=pd.read_csv(string)
    cluster.set_index('ID',inplace=True)
    Cluster=[]
    for i in range(0,max(cluster['Class'])+1):
        Cluster.append(cluster[cluster.Class==i])
    return Cluster
    

def script():
    pickle_in=open('E:/IIT Bombay/IITB/Sem2/ASIP/Matrix.pkl','rb')
    CmpMatrix=pickle.load(pickle_in)
    start=time.time()
    image=read_image()
    display_image(image,'Input Image')
    ClusterFormatn.ClusterFormation(image)
    Cluster=read_file('E:/IIT Bombay/IITB/Sem2/ASIP/Cluster.csv')
    maxLike=MaximumLikelihood(Cluster,image)
    [hardImg,ClassImg]=maxLike.Compute()
    display_image(hardImg,'Hard Class')
    end=time.time()
    print('Total time taken : %f seconds after Maximum Likelihood Classification'%(end-start))
    
    pickle_M=open('E:/IIT Bombay/IITB/Sem2/ASIP/tempClass.pkl','wb')
    pickle.dump(ClassImg,pickle_M)
    pickle_M.close()
    '''
    
    pickle_in=open('E:/IIT Bombay/IITB/Sem2/ASIP/tempClass.pkl','rb')
    ClassImg=pickle.load(pickle_in)
    '''
    
    relaxLabel=RelaxationLabelling(CmpMatrix,ClassImg,3)
    relax=relaxLabel.Convolution()
    display_image(maxLike.HardDecision(relax),'relax labeled')
    
    end=time.time()
    print('Total time taken : %f seconds after Relaxation Labelling for 1 iteration'%(end-start))
    
    pickle_T=open('E:/IIT Bombay/IITB/Sem2/ASIP/RelaxClass.pkl','wb')
    pickle.dump(relax,pickle_T)
    pickle_T.close()
    
    #print(maxLike.getMean())
    #print(maxLike.getMean()[0].keys())
    #print(maxLike.getCovariance())
    #Cluster[0].drop('Class',axis=1,inplace=True)
    #print(Cluster[0])
    #print(np.mean(Cluster[0]))
    #print(np.mean(Cluster[0])['Blue'])  
    #print(Cluster[0].columns)
    #print(Cluster[0][0:1])
    
script()    