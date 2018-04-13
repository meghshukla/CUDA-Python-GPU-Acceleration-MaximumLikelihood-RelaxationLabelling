import numpy as np
import pickle

CompatMatrix=np.zeros([5,5],dtype=np.float64)


CompatMatrix[0,:]=[0.7,0.4,0.55,0.1,0.25]              #Water
CompatMatrix[1,:]=[0.4,0.7,0.55,0.4,0.4]              #Forest
CompatMatrix[2,:]=[0.4,0.55,0.7,0.1,0.1]               #Grassland
CompatMatrix[3,:]=[0.2,0.45,0.2,0.7,0.6]               #Barren
CompatMatrix[4,:]=[0.05,0.4,0.15,0.55,0.7]               #Urban

pickle_Matrix=open('E:/IIT Bombay/IITB/Sem2/ASIP/Matrix.pkl','wb')
pickle.dump(CompatMatrix,pickle_Matrix)
pickle_Matrix.close()
