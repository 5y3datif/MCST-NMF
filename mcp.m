% Adding paths of support files.
addpath('.\support\a-hals');
%% Add paths for support files
addpath('.\datasets\geant_raw');
% Load Geant dataset.
TM=load('Geant_TM_with_miss_data.mat');
X=TM.TM_with_miss_data;
% logical indices of data missing in TM.
P=(X<0);
% rank of factorization for low rank matrix factorization.
r=20;
[n,T]=size(X);
% initialize W and H
W0=rand(n,r);
H0=rand(r,T);
% computing low rank approximation using HALS.
X_nn = X.*(X>0);
[W,H] = HALSacc(X_nn,W0,H0);
% reconstruct X
reconstruct_X=W*H;
temp1 = reconstruct_X.*P;
temp2 = X.*(X>=0);
TM_with_filled_miss_data=X.*(X>=0)+reconstruct_X.*P;
% save results.