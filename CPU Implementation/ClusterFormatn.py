import pandas as pd
import numpy as np

def ClusterFormation(Img):
    iD=[]
    ID=1
    #Water_blue : 0 Water_grey : 0, Vegetation Trees : 1, Vegetation Grassland : 2 Barren Land : 3, Built-up_1 : 4
    #Band=[[]]*np.shape(Img)[-1] lead to all n lists pointing to the same location in memory
    Band=[[]]
    for k in range(np.shape(Img)[-1]-1):
        Band+=[[]]
    Class=[]
                    
    #Water_blue
    for j in range(244,253):
        for i in range(527,535):
            iD.append(ID)
            ID+=1
            for k in range(np.shape(Img)[-1]):
                Band[k].append(Img[i,j][k])
            Class.append(0)
        
       
    #Water_grey        
    for j in range(330,401):
        for i in range(680,731):
            iD.append(ID)
            ID+=1
            for k in range(np.shape(Img)[-1]):
                Band[k].append(Img[i,j][k])
            Class.append(0)   
   
    #Vegetation Trees
    for j in range(560,621):
        for i in range(640,701):
            iD.append(ID)
            ID+=1
            for k in range(np.shape(Img)[-1]):
                Band[k].append(Img[i,j][k])
            Class.append(1)    
        
                        
    #Vegetation Grassland
    for j in range(800,861):
        for i in range(660,701):
            iD.append(ID)
            ID+=1
            for k in range(np.shape(Img)[-1]):
                Band[k].append(Img[i,j][k])
            Class.append(2)    
        
                                
    #Barren Land
    for j in range(330,361):
        for i in range(40,61):
            iD.append(ID)
            ID+=1
            for k in range(np.shape(Img)[-1]):
                Band[k].append(Img[i,j][k])
            Class.append(3)
    for j in range(664,681):
        for i in range(80,106):
            iD.append(ID)
            ID+=1
            for k in range(np.shape(Img)[-1]):
                Band[k].append(Img[i,j][k])
            Class.append(3)
        
                                              
    #Builtup
    for j in range(50,101):
        for i in range(345,366):
            iD.append(ID)
            ID+=1
            for k in range(np.shape(Img)[-1]):
                Band[k].append(Img[i,j][k])
            Class.append(4)
    for j in range(110,151):
        for i in range(652,661):
            iD.append(ID)
            ID+=1
            for k in range(np.shape(Img)[-1]):
                Band[k].append(Img[i,j][k])
            Class.append(4)
                                                                                                             
        
        
    #Creating Dict
    cluster={'ID':iD,'Class':Class} 
    for k in range(np.shape(Img)[-1]):
        label='Band'+str(k)
        cluster[label]=Band[k]   
    cluster=pd.DataFrame(cluster)
    cluster.set_index('ID',inplace=True)
    cluster.to_csv('E:/IIT Bombay/IITB/Sem2/ASIP/Cluster.csv')
      
                                  
        
        
    


