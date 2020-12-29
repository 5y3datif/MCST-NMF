%% Initialization algorithm for Structured Nonnegative Matrix Factorization 
%  for Traffic Flow Estimation
% Reference: S M Atif, S Qazi, N Gillis, I Naseem. "Structured Nonnegative 
% Matrix Factorization for Traffic Flow Estimation of Large Cloud Network".
%
% Written by Syed Muhammad Atif, Karachi Institute of Economics and 
% Technology, Karachi, Pakistan.
% Latest update December 2020
%
% Use NNSVD-LRC.
% Reference: S M Atif, S Qazi, N Gillis. "Improved SVD-based initialization
% for nonnegative matrix factorization using low-rank correction." Pattern
% Recognition Letters 122 (2019): 53-59.
%
% Input 
%   X: training OD flows matrix
%   A: routng matrix of the underlying network
%   L: lag set for autoregression regularization in an array
%   k: factorization rank
%   options: a structure including  
%            'betaAR' (the scaling parameter for computing penalty 
%                     parameter for autoregresion regularization term 
%                     lambdaAR. By default betaAR = 0.2.)
%            'betaA' (the scaling parameter for computing penalty 
%                     parameter for orthogonality regularization term 
%                     lambdaA. By default betaA = 0.2. 
%                     For Abiline and GEANT, betaA = 0.2 and betaA = 0.6 
%                     work fine respectively.)
% Output
%  (W,H,O): solution
%      lanbdaAR: penelty parameter for autoregression regularization term.
%      lanbdaA: penelty parameter for orthogonality regularization term.
%      e: relative fitting errors
%      eAR: relative fitting errors
%      t: corresponding running time

function [W,H,O,lambdaA,lambdaAR,e,t] = mcst_init(X,A,L,k,options)
    cputime0 = tic;
    % Arrange lag set chronologically.
    L = sort(L,'ascend');
    % scaling parameters for computing penelty parameter for 
    %  regularization terms.
    if nargin < 5 
        options = []; 
    end
    if ~isfield(options,'betas') 
        betaA = 0.2;
        betaAR = 0.2;
    else
        betaA = options.betas.betaA;
        betaAR = options.betas.betaAR;        
    end
    
    [W,H] = NNSVDLRC(X,k);
    T = size(X,2);
    % scale the innitial point 
    HHt = H*H'; 
    XHt = X*H'; 
    scaling = sum(sum(XHt.*W))/sum(sum( (W'*W).*(HHt) )); 
    W = W*scaling; 
    % compute k Hp matrices
    sz_L = size(L,2);
    Hp = zeros(k,sz_L,T);
    O = zeros(k, sz_L);
    for idx = 1:sz_L 
        Hp(:,idx,1+L(idx):T) = H(:,1:T-L(idx)); 
    end
    % compute Omega
    for p=1:k
        O(p,:) = max(H(p,:)*pinv(reshape(Hp(p,:,:),sz_L,T)),0);
    end
    % penalty parameters of regularization terms.
    sumSqX = sum((X.^2),'all'); WtW = W'*W; HHt = H*H'; XHt = X*H'; 
    err = sumSqX - 2*sum(W.*XHt,'all') + sum(WtW.*HHt,'all'); 
    Ak = A*W;
    eRegA = sum((Ak'*Ak-eye(k)).^2,'all');
    sz_L = size(L,2);
    eRegAR = 0;
    for p=1:k
        temp = H(p,:) - O(p,:)*reshape(Hp(p,:,:),sz_L,T);
        eRegAR = eRegAR + temp*temp';
    end
    lambdaAR = betaAR*err/eRegAR;
    lambdaA = betaA*err/eRegA;
    t = toc(cputime0);
    % relative error. 
    e = sqrt(err)/sumSqX;
end