clear all;
%% Add paths for support files
addpath('.\support\nnsvd-lrcv2');
%% Add paths for support files
addpath('.\datasets\abilene12by12');
%% Load Geant dataset.
TM = load('Abilene.mat');
R = dlmread('A-matrix.txt', '\t');
%% Initialing setup.
% Training and testing data
% number of link flows.
m = 54;
% number of OD flows.
n = 144;
% number of nodes.
nodes = sqrt(n);
% number of timestamp.
TT = 0*288;
T = 11*288;
Ttrain = 7*288;
Ttest = 4*288;
% m*n routing matrix.
A = R;
% n*t OD flows matrices.
X = [TM.X1',TM.X2'];
Xtrain = X(:,TT+1:TT+Ttrain); 
Xtest = X(:,TT+Ttrain+1:TT+T);
% m*t link flows matrices.
Y = A*X;
Ytrain = A*Xtrain;
Ytest = A*Xtest;
% Timestamps in training and testing sets.
Ttrain = size(Xtrain,2);
Ttest = size(Xtest,2);
% rank of factorization.
k = 20;
% Define the lag set.
L = [1, 2, 3, 12, 24, 96, 102, 108, 288];
%% Initializition of free parameter.
[W0,H0,Omega0,lambdaA,lambdaAR,einit,timeinit] = mcst_init(Xtrain,A,L,k);
%% Train model using iterative algorithm
options = [];
options.init.W = W0;
options.init.H = H0;
options.init.O = Omega0;
options.lambdas.lambdaA = lambdaA;
options.lambdas.lambdaAR = lambdaAR;
[W,H,O,cnt,etrain,timetrain] = mcst_training(Xtrain,A,L,k,options);
%% Estimation of OD flows.
Xtesthat = X(:,Ttrain+1:T);
for t=1:1:T
    yt = Ytest(:,t);
    Xtesthat(:,t) = mcst_estimation(W,A,yt);
end
