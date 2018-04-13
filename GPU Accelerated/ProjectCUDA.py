import pickle
from osgeo import gdal
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib import colors as clr
from numba import cuda
from numba import float64
import time

import ClusterFormatn

classColours=[[0, 0, 255],[0, 255, 0],[0, 255, 255],[255, 255, 0], [255, 255, 255], [0, 0, 0]]
img_dim=None
classImg_dim=None
imgBands=0                                                                      #CUDA JIT will compile with these values, changes made after the kernel code, they will not
classBands=0                                                                    #reflect in kernel. Use cuda.jit(*no args*) to compile when called


############################################################### INPUT HANDLING FUNCTIONS #######################################################                                    

#Run on CPU
def read_image(imgPath):
    global img_dim
    global imgBands
    dataset = gdal.Open(imgPath)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    Img=np.zeros([rows,cols,bands])
    for i in range(bands):
        band = dataset.GetRasterBand(i+1)
        Img[:,:,i] = band.ReadAsArray(0, 0, cols, rows).astype(np.float)
    Img=Img[450:1218,1100:2220,:]
    img_dim=list(np.shape(Img))
    imgBands=img_dim[-1]
    return Img,img_dim
    
#Run on CPU    
def display_image(image,string): 
    Norm = clr.Normalize(vmin = 0, vmax = np.max(image), clip = False) 
    data_disp=Norm(image) 
    plt.figure()      
    plt.imshow(data_disp)
    plt.title(string)        
    
#Run on CPU    
def read_file(string):
    global classImg_dim
    global classBands
    cluster=pd.read_csv(string)
    cluster.set_index('ID',inplace=True)
    ######If you want to normalize######
    #for j in cluster.columns:
    #    if(j=='Class'):
    #        pass
    #    else:
    #        cluster[j]=cluster[j]/255
    Cluster=[]
    for i in range(0,max(cluster['Class'])+1):
        Cluster.append(cluster[cluster.Class==i])
    classImg_dim=img_dim[:-1]+[i+1] 
    classBands=i+1   
    return Cluster,classImg_dim    



######################################################################### MAXIMUM LIKELIHOOD ####################################################################              
        
class MaximumLikelihood(object):
    def __init__(self,cluster,image):
        self.Cluster=cluster
        self.image_=image
    
    #Run on CPU        
    def Mean(self):                                                             #Sets 'Mu' which is in form np. matrix, shape : clusters x bands, dtype = np.float16
        self.Mu=[]
        for i in range(len(self.Cluster)):
            self.Mean_=[]
            for j in self.Cluster[i].columns:
                self.Mean_.append(np.mean(self.Cluster[i][j]))
            self.Mu.append(self.Mean_)
        self.Mu=np.matrix(self.Mu,dtype=np.float64)        
        del i,j,self.Mean_    
    
    #Run on CPU    
    def Covariance(self):                                                       #Sets covMatrix and classDet, both converted to Max Like suitable form, dtype = np.float16
        self.CovMatrix=[]
        self.classDet=[]
        for i in range(len(self.Cluster)):
            self.Cluster[i].drop('Class',axis=1,inplace=True)
            self.CovMatrix.append(np.cov(self.Cluster[i],rowvar=False)) 
            self.classDet.append(np.linalg.det(self.CovMatrix[i])) 
            
        self.CovMatrix=list(map(np.linalg.inv,self.CovMatrix))
        self.covMatrix=np.zeros((img_dim[2],img_dim[2],len(self.CovMatrix)),dtype=np.float64)
        for j in range(len(self.CovMatrix)):  
            self.covMatrix[:,:,j]=np.matrix(self.CovMatrix[j],dtype=np.float64) 
        self.classDet=list(map(np.sqrt,self.classDet))   
        self.classDet=np.array(self.classDet,dtype=np.float64)   
        del i,j,self.CovMatrix
        
    
    #Run on CPU
    def ThreadBlockClassify(self):
        self.ThreadsPerBlockClassify=(16,16)                                                           #Hard coded, should not let the user choose this, TPB = 256 
        self.bpgX=(img_dim[0]+self.ThreadsPerBlockClassify[0]-1)//self.ThreadsPerBlockClassify[0]
        self.bpgY=(img_dim[1]+self.ThreadsPerBlockClassify[1]-1)//self.ThreadsPerBlockClassify[1]
        self.BlocksPerGridClassify=(self.bpgX,self.bpgY)
        del self.bpgX,self.bpgY
        print(str(self.BlocksPerGridClassify))
        
     
    '''       
    #Run on CPU
    def ThreadBlockNormalize(self):
        self.ThreadsPerBlockNormalize=(8,8,4)                                                           #Hard coded, should not let the user choose this, TPB = 256 
        self.bpgX=(img_dim[0]+self.ThreadsPerBlockNormalize[0]-1)//self.ThreadsPerBlockNormalize[0]
        self.bpgY=(img_dim[1]+self.ThreadsPerBlockNormalize[1]-1)//self.ThreadsPerBlockNormalize[1]
        self.bpgZ=(img_dim[2]+self.ThreadsPerBlockNormalize[2]-1)//self.ThreadsPerBlockNormalize[2]
        self.BlocksPerGridNormalize=(self.bpgX,self.bpgY,self.bpgZ)
        del self.bpgX,self.bpgY,self.bpgZ  
        
     
    @cuda.jit('void(float64[:,:,:],int16[:])')    
    def Normalize(kernel_ClassifiedGPU,kernel_img_dimGPU):
        threadIDx=cuda.threadIdx.x
        threadIDy=cuda.threadIdx.y
        threadIDz=cuda.threadIdx.z
        blockIDx=cuda.blockIdx.x
        blockIDy=cuda.blockIdx.y
        blockIDz=cuda.blockIdx.z
        blockX=cuda.blockDim.x
        blockY=cuda.blockDim.y
        blockZ=cuda.blockDim.z
        i=(blockX*blockIDx)+threadIDx
        j=(blockY*blockIDy)+threadIDy  
        k=(blockZ*blockIDz)+threadIDz 
        if(i<kernel_img_dimGPU[0] and j<kernel_img_dimGPU[1] and k<kernel_img_dimGPU[2]):
            kernel_ClassifiedGPU[i,j,k]=kernel_ClassifiedGPU[i,j,k]/255
    '''
    
         
    #@cuda.jit('void(float64[:,:,:],int16[:],int16[:],float64[:,:],float64[:,:,:],float64[:])')    Have to use cuda.jit() because imgBands and classBands not getting updated
    @cuda.jit()
    def Classify(kernel_ClassifiedGPU, kernel_img_dimGPU, kernel_classImg_dimGPU, kernel_MuGPU, kernel_covMatrixGPU, kernel_classDetGPU):
        #Thread initialization
        threadIDx=cuda.threadIdx.x
        threadIDy=cuda.threadIdx.y
        blockIDx=cuda.blockIdx.x
        blockIDy=cuda.blockIdx.y
        blockX=cuda.blockDim.x
        blockY=cuda.blockDim.y
        i=(blockX*blockIDx)+threadIDx
        j=(blockY*blockIDy)+threadIDy
        
        
        if(i<kernel_img_dimGPU[0] and j<kernel_img_dimGPU[1]):               #Defining boundaries to perform computation
            if(i==700 and j==800):
                print('Pixel being examined : ')
                print(i,j)
                print(imgBands, classBands)
                
            pixelValue=cuda.local.array(shape=imgBands,dtype=float64)        #Creating a copy of Pixel values at i,j
            for q in range(kernel_img_dimGPU[-1]):
                pixelValue[q]=kernel_ClassifiedGPU[i,j,q]
                
            classProb=cuda.local.array(shape=(classBands),dtype=float64)     #Creating an array to hold class probability values, initializing to zero
            for q in range(len(classProb)):
                classProb[q]=0
            
            muDiff=cuda.local.array(shape=(imgBands),dtype=float64)          #Creating an array to hold difference between pixel and mean
            temp_0=cuda.local.array(shape=(imgBands),dtype=float64)          #Creating an array to hold the values for first matrix multiplication
            for k in range(kernel_classImg_dimGPU[-1]):
                for q in range(len(muDiff)):                                              #Initializing these values
                    muDiff[q]=0 
                    temp_0[q]=0

                
                if(i==700 and j==800):
                    print('\nPixel values:')
                    for qq in pixelValue:
                        print(qq)
                for p in range(len(pixelValue)):
                    muDiff[p]=pixelValue[p]-kernel_MuGPU[k,p]
                if(i==700 and j==800):
                    print('\nPixel values:')
                    for qq in pixelValue:
                        print(qq)
                if(i==700 and j==800):
                    print('\nCluster number : ',k)
                    for qq in muDiff:
                        print("    ",qq)  
                          
                #MULTIPLICATION SEGMENT STARTS        
                for column_ in range(kernel_img_dimGPU[-1]):
                    temp_1=0
                    for k_ in range(kernel_img_dimGPU[-1]):
                        temp_1=temp_1+(muDiff[k_]*kernel_covMatrixGPU[k_,column_,k])
                    temp_0[column_]=temp_1
                temp_1=0
                for k__ in range(len(muDiff)):
                    temp_1=temp_1+temp_0[k__]*muDiff[k__]  
                #MULTIPLICATION SEGMENT ENDS
                if(i==700 and j==800):
                    print('\nPixel values after multiplication:')
                    for qq in pixelValue:
                        print(qq)
                        
                if(i==700 and j==800):
                    print('Result of Multiplication :')
                    print(temp_1) 
                classProb[k]=(math.exp(((temp_1)/2)*-1))/((math.pow(2*math.pi,(kernel_img_dimGPU[-1]/2)))*kernel_classDetGPU[k])
            for k___ in range(len(classProb)):
                kernel_ClassifiedGPU[i,j,k___]=classProb[k___]
                 
        if(i==700 and j==800):
            print('Class probability values : ')
            for bc in kernel_ClassifiedGPU[i,j]:
                print(bc),
        cuda.syncthreads()                                      
        
    #Run on CPU    
    def HardDecision(self,CLASSimg):
        self.Hard_=np.zeros(np.shape(self.image_)[:-1]+(3,))
        for i in range(np.shape(self.image_)[0]):
            for j in range(np.shape(self.image_)[1]): 
                self.a=list(CLASSimg[i,j])
                self.Hard_[i,j]=classColours[self.a.index(max(self.a))]
                del self.a
        del i,j  
        return self.Hard_          
         
    #Run on CPU  
    def Compute(self):
        global imgBands
        global classBands
        self.Covariance()   
        self.Mean()
        self.ThreadBlockClassify()
        #self.ThreadBlockNormalize()
         
        self.dim3=max(len(self.Cluster),img_dim[-1])                                          #To avoid having one array for result and one for image
        self.Classified_=np.zeros(shape=(img_dim[:-1]+[self.dim3]),dtype=np.float64)
        self.Classified_[:,:,:img_dim[-1]]=self.image_[:,:,:]
        
        #Debugging statements
        
        print("Mean : "+str(np.shape(self.Mu)))
        print(self.Mu)
        print("Inverse Covariance : "+str(np.shape(self.covMatrix)))
        print(self.covMatrix)
        print("Det : "+str(np.shape(self.classDet)))
        print(self.classDet)
        print('Classified : '+str(np.shape(self.Classified_)))
        print('Image dimension of cluster : '+str(classImg_dim))
        
        
        Classified_GPU=cuda.to_device(self.Classified_)
        img_dimGPU=cuda.to_device(np.array(img_dim,dtype=np.int16))
        classImg_dimGPU=cuda.to_device(np.array(classImg_dim,dtype=np.int16))
        MuGPU=cuda.to_device(self.Mu)
        covMatrixGPU=cuda.to_device(self.covMatrix)
        classDetGPU=cuda.to_device(self.classDet)
        
        #self.Normalize[self.BlocksPerGridNormalize,self.ThreadsPerBlockNormalize](Classified_GPU, img_dimGPU)
        self.Classify[self.BlocksPerGridClassify,self.ThreadsPerBlockClassify](Classified_GPU, img_dimGPU, classImg_dimGPU, MuGPU, covMatrixGPU, classDetGPU) 
        self.GPUClass=Classified_GPU.copy_to_host() 
        return self.HardDecision(self.GPUClass[:,:,:]),Classified_GPU
               
        
    def getMean(self):
        return self.Mu[:,:]
    
    def getCovariance(self):                          
        return self.covMatrix[:,:,:]
        
    def getClassified(self):
        return self.GPUClass[:,:,:]
        
################################################################## RELAXATION LABELLING #######################################################
       
                     
class RelaxationLabelling(object):
    def __init__(self,CompatMatrix):
        self.CompatMatrix=CompatMatrix
        
    #Run on CPU
    def ThreadBlockRelax(self):
        self.ThreadsPerBlockRelax=(16,16)                                                           #Hard coded, should not let the user choose this, TPB = 256 
        self.bpgX=(img_dim[0]+self.ThreadsPerBlockRelax[0]-1)//self.ThreadsPerBlockRelax[0]
        self.bpgY=(img_dim[1]+self.ThreadsPerBlockRelax[1]-1)//self.ThreadsPerBlockRelax[1]
        self.BlocksPerGridRelax=(self.bpgX,self.bpgY)
        del self.bpgX,self.bpgY 
        
           
    #@cuda.jit('void(float64[:,:,:],float64[:,:,:],float64[:,:],int16[:])')
    @cuda.jit()
    def relaxLabel(kernel_RelaxImage_GPU, kernel_ClassImage_GPU, CompatMatrix_GPU,kernel_class_dimGPU):   
        #Thread initialization
        threadIDx=cuda.threadIdx.x
        threadIDy=cuda.threadIdx.y
        blockIDx=cuda.blockIdx.x
        blockIDy=cuda.blockIdx.y
        blockX=cuda.blockDim.x
        blockY=cuda.blockDim.y
        i=(blockX*blockIDx)+threadIDx
        j=(blockY*blockIDy)+threadIDy
        
        
        #if(i>0 and j>0):
        if(True):
            if(i<(kernel_class_dimGPU[0]-1) and j<(kernel_class_dimGPU[1]-1)):                                 #Defining boundaries to perform computation
                temp=0
                for k in range(kernel_class_dimGPU[2]):
                    Qi=0
                    for p in (range(2)):                                             #Window size fixed at 3
                        for q in (range(2)):
                            if(p==1 and q==1):
                                pass
                            else:
                                Qij=0
                                for k_ in range(len(kernel_ClassImage_GPU[i,j])):
                                    Qij+=kernel_ClassImage_GPU[i-p-1,j-q-1,k_]*CompatMatrix_GPU[k,k_]
                                Qi+=Qij/8
                    temp+=Qi*kernel_ClassImage_GPU[i,j,k]
                    kernel_RelaxImage_GPU[i,j,k]=Qi*kernel_ClassImage_GPU[i,j,k]
                for _ in range(len(kernel_RelaxImage_GPU[i,j])):
                    kernel_RelaxImage_GPU[i,j,_]/=temp    
                
                                                  
                        
            
        
        
    def Compute(self,ClassImage_GPU):
        self.ThreadBlockRelax()
        RelaxImage_GPU=cuda.device_array_like(ClassImage_GPU)
        CompatMatrix_GPU=cuda.to_device(self.CompatMatrix)
        class_dimGPU=cuda.to_device(np.array(classImg_dim,dtype=np.int16))
        self.relaxLabel[self.BlocksPerGridRelax,self.ThreadsPerBlockRelax](RelaxImage_GPU,ClassImage_GPU,CompatMatrix_GPU,class_dimGPU)
        self.GPURelax=RelaxImage_GPU.copy_to_host()
        return self.GPURelax,RelaxImage_GPU
                  
