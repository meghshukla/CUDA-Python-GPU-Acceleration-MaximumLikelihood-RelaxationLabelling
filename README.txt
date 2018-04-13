CUDA Python GPU Accelerated implementation of Image Processing Technique
Author : Megh Shukla, MTech IIT Bombay

Dependency:
 1. NVIDIA cudatoolkit available from: https://developer.nvidia.com/cuda-downloads

How to use:
Run the MaxLikeRelaxLabel.exe in Classifier_Executable folder

Other files:
ClusterFormatn.py : Creating ground truth values of classes in csv format
Matrix.py : Generating Compatibility Matrix for Relaxation Labelling
ProjectCUDA.py: CUDA kernel Python implementation of Maximum Likelihood and Relaxation Labelling
MaxLikeRelaxLabel.py : GUI file making calls to ProjectCUDA, ClusterFormatn and Matrix.py

*.pkl : Python pickle files for storing objects


Executable can run only on Windows systems with cudatoolkit installed
Make sure to set system variable Path to location of Toolkit if not set by installer

GPU and CPU implementation attached, however it is highly recommmended to use GPU implementation,
 1. Executable comes with GUI which implements GPU code
 2. CPU code is highly time consuming, GPU implementation is extremely fast due to parallelized nature of algorithms
    CPU (i5-8250u): ~670 seconds for ONE relaxation labelling iteration
    GPU (NVIDIA GeForce MX 150): ~ 9 seconds for ONE relaxation labelling iteration
    
    NOTE : GPU becomes highly efficient if multiple iterations performed, since cost of CPU --> GPU and GPU --> CPU is performed
	   only once, and is amortized over all the iterations

Implementation is done as a part of Course Project: 
Advanced Satellite Image Processing, GNR 602
Centre of Studies in Resource Engineering
Indian Institute of Technology Bombay

General purpose Maximum Likelihood Classification of given Image
Relaxation Labelling is performed using initial probabilities from Maximum Likelihood Classification

GPU JIT compiler: Numba
GUI: PyQt4
Executable: PyInstaller

More information and acknowledgements can be found in docx and pptx file attached, help button of GUI

Input: Any image, example Satellite Image provided, Powai-ikonos
       Ground Truth samples
       Compatibility Matrix for Relaxation Labelling
	
Output: Classified output: Maximum Likelihood and Relaxation Labelling

