clear all;
%% Add paths for support files
addpath('.\support\nnsvd-lrcv2');
%% Add paths for support files
addpath('.\datasets\geant_preprocessed');
%% Load Geant dataset.
TM = load('Geant_TM_with_filled_miss_data.mat');
R = open('Geant_routing_table.mat');
%% Initialing setup.
% Training and testing data
% number of link flows.
m = 120;
% number of OD flows.
n = 529;
% number of nodes.
nodes = sqrt(n);
% number of timestamp.
TT = 11*96;
T = 11*96;
Ttrain = 7*96;
Ttest = 4*96;
% m*n routing matrix.
A = R.A;
% n*t OD flows matrices.
X = TM.TM_with_filled_miss_data(:,TT+1:TT+T);
Xtrain = X(:,1:Ttrain); 
Xtest = X(:,Ttrain+1:T);
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
L = [1, 4, 8, 32, 34, 36, 96];
%% Initializition of free parameter.
options = [];
options.betas.betaA = 0.6;
options.betas.betaAR = 0.2;
[W0,H0,Omega0,lambdaA,lambdaAR,einit,timeinit] = mcst_init(Xtrain,A,L,k,options);
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
