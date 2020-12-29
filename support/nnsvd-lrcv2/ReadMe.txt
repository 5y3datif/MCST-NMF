This code contains the SVD-based initialization for NMF, referred to as nonnegative SVD with low-rank correction (NNSVD-LRC), described in the paper 

Improved SVD-based Initialization for Nonnegative Matrix Factorization using Low-Rank Correction, Syed Muhammad Atif, Sameer Qazi, Nicolas Gillis, July 2018.


You can run test_cbcl (resp. test_classic) to compare this initialization with two existing SVD-based initializations, namely NNDSVD [1] and SVD-NMF [2], and with random initialization on the dense CBCL face image data set (resp. the sparse classic document data set). 

Version 2 has some bugs corrected, and the use of mySVD (available from http://www.cad.zju.edu.cn/home/dengcai/Data/code/mySVD.m) that runs faster than svds for dense matrices. 


[1] Boutsidis, C., Gallopoulos, E., 2008. SVD based initialization: A head start for nonnegative matrix factorization. Pattern Recognition 41, 1350–1362.
[2] Qiao, H., 2015. New SVD based initialization strategy for non-negative matrix factorization. Pattern Recognition Letters 63, 71–77. 