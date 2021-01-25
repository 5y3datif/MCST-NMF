clear all;
%% Add paths for support files
addpath('.\support\nnsvd-lrcv2');
%% Add paths for support files
addpath('.\datasets\geant_preprocessed');
addpath('.\datasets\geant_raw');
%% Load Geant dataset.
TM = load('Geant_TM_with_miss_data.mat');
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
TT = 0*96;
T = 11*96;
Ttrain = 7*96;
Ttest = 4*96;
% m*n routing matrix.
A = R.A;
% n*t OD flows matrices.
X = TM.TM_with_miss_data(:,TT+1:TT+T);
X = (X > 0).*X;
Xtrain = X(:,1:Ttrain); 
Xtest = X(:,Ttrain+1:T);
% m*t link flows matrices.
Y = A*X;
Ytrain = A*Xtrain;
Ytest = A*Xtest;
% rank of factorization.
k = 20;
% Define the lag set.
L = [1, 4, 8, 32, 34, 36, 96];
%% Initializition of free parameter.
Xtrain = (Xtrain > 0).*Xtrain;
options = [];
options.betas.betaA = 0.1;
options.betas.betaAR = 0.1;
[W0,H0,Omega0,lambdaA,lambdaAR,einit,timeinit] = mcst_init(Xtrain,A,L,k,options);
%% Train model using iterative algorithm
options = [];
options.M = (Xtrain > 0);
options.init.W = W0;
options.init.H = H0;
options.init.O = Omega0;
options.lambdas.lambdaA = lambdaA;
options.lambdas.lambdaAR = lambdaAR;
[W,H,O,cnt,etrain,timetrain] = mcst_nmc_training(Xtrain,A,L,k,options);
%% Estimation of OD flows.
Xtesthat = X(:,Ttrain+1:T);
for t=1:1:Ttest
    yt = Ytest(:,t);
    Xtesthat(:,t) = mcst_estimation(W,A,yt);
end
%% Analyse results.
addpath('.\datasets\geant_raw');
TM = load('Geant_TM_with_miss_data.mat');
X = TM.TM_with_miss_data(:,TT+1:TT+T);
Xtrain = X(:,1:Ttrain); 
Xtest = X(:,Ttrain+1:T);
err = Xtesthat-(Xtest > 0).*Xtest;
sq_X = Xtest.^2;
sq_err = err.^2;
nrm_err_TRE = sqrt(sum(sq_err));
nrm_err_SRE = sqrt(sum(sq_err,2));
nrm_X_TRE = sqrt(sum(sq_X));
nrm_X_TRE = (nrm_X_TRE > 0).*nrm_X_TRE + ...
   (nrm_X_TRE <= 0).*(ones(size(nrm_X_TRE))); % Avoid diviion by zero
nrm_X_SRE = sqrt(sum(sq_X,2));
nrm_X_SRE = (nrm_X_SRE > 0).*nrm_X_SRE + ...
  (nrm_X_SRE <= 0).*(ones(size(nrm_X_SRE))); % Avoid diviion by zero
TRE = nrm_err_TRE./nrm_X_TRE;
SRE = nrm_err_SRE./nrm_X_SRE;
pd_SRE = fitdist(SRE,'Kernel');
sort_SRE = 0:0.01:1.8;
CDF_SRE = cdf(pd_SRE,sort_SRE);
pd_TRE = fitdist(TRE','Kernel');
sort_TRE = 0:0.01:0.4;
CDF_TRE = cdf(pd_TRE,sort_TRE);
[mean_od,mean_od_ind]=sort(mean(X,2));
bias_od=(1/Ttest)*sum(err,2);
bias_od=bias_od(mean_od_ind);
sd_od=sqrt((1/(Ttest-1))*sum(((err-diag(bias_od)*ones(n,Ttest)).^2),2));
sd_od=sd_od(mean_od_ind);
%% CDF SRE TRE
figure
plot(sort_SRE,CDF_SRE,'-b','LineWidth',2);
title('CDF of SRE');
xlabel('SRE (from lowest to highest)');
ylabel('F(SRE)');
grid minor;
axis([0.2 1.2 0.1 0.95]);
set(gca,'XMinorTick','on','YMinorTick','on','XScale','log','YScale','log');

figure
plot(sort_TRE,CDF_TRE,'-b','LineWidth',2);
title('CDF of TRE');
xlabel('TRE (from lowest to highest)');
ylabel('F(TRE)');
grid minor;
axis([0.09 0.24 0.1 0.98]);
set(gca,'XMinorTick','on','YMinorTick','on','XScale','log','YScale','log');
%% Bias and Standard Deviation
figure
scatter(mean_od,bias_od,'MarkerFaceColor','b','MarkerEdgeColor','b',...
        'LineWidth',2);
title('Bias vs Means of OD flows (sorted from lowest to highest)');
xlabel('Mean of OD flows');
ylabel('Bias');
grid minor;
xlim([1,10e5]);
set(gca,'XMinorTick','on','YMinorTick','on','XAxisLocation','origin',...
    'XScale','log',...
    'Layer','top','Box', 'on');

figure
scatter(sd_od,bias_od,'MarkerFaceColor','b','MarkerEdgeColor','b',...
        'LineWidth',2);
title('Bias vs Standard Deviation of OD flows (sorted from lowest to highest)');
xlabel('Standard Deviation of OD flows');
ylabel('Bias');
grid minor;
xlim([1,10e5]);
set(gca,'XMinorTick','on','YMinorTick','on','XAxisLocation','origin',...
    'XScale','log',...
    'Layer','top','Box', 'on');
