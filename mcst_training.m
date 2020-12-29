%% Training algorithm for Structured Nonnegative Matrix Factorization 
%  for Traffic Flow Estimation
% Reference: S M Atif, S Qazi, N Gillis, I Naseem. "Structured Nonnegative 
% Matrix Factorization for Traffic Flow Estimation of Large Cloud Network".
%
% Written by Syed Muhammad Atif, Karachi Institute of Economics and 
% Technology, Karachi, Pakistan.
% Latest update December 2020
%
% Input 
%   X: training OD flows matrix
%   A: routng matrix of the underlying network
%   L: lag set for autoregression regularization in an array
%   k: factorization rank
%   options: a structure including  
%            'init.W', 'init.H', 'init.H' (initial points of U, V and O)
%            'maxitrs.qmax' (maximum number of iterations. 
%                          By default qmax = 50)
%            'maxitrs.qWmax', 'maxitrs.qHmax', 'maxitrs.qOmax' (maximum 
%                    number of iterations for repeating updates of 
%                    W, H or O (Omega).
%                    By default qWmax = 10, qHmax = 10, qOmax = 10))
%            'timemax' (maximum of running time)
%            'maxdeltas.delta' (the parameter to stop the outer loop of 
%                    updating W, H and O (Omega)). 
%            'maxdeltas.deltaW', 'maxdeltas.deltaH', 'maxdeltas.deltaO' 
%                   (the parameter to stop the inner loop of updating 
%                    W, H or O (Omega). 
%                    Criterion: if 0 < eprev - ecurr < estart then stop.
%                    By default deltaW = 1e-3, deltaH = 1e-3, deltaO = 1e-5)
%           'lambdas.lamdaAR, lambdas.lamdaA (penalty parameters of 
%                    autoregression and orthogonality regularization terms)
% Output
%  (W,H,O): solution
%        q: iterations before exit (q < qmax if an exit occur due to 
%           meeting stopping criteion for outer loop).
%        e: sequence of squared fitting errors
%        t: corresponding running time

function [W,H,O,q,e,t] = mcst_training(X,A,L,k,options)
    cputime0 = tic;
    [n,T] = size(X);
    sz_L = size(L,2);
    % Arrange lag set chronologically.
    L = sort(L,'ascend');
    %% Parameters of MCST-NMF training algorithm.
    if nargin < 5
        options = [];
    end
    if ~isfield(options,'init')
        printf('Use initial free parameters provided by mcst_init.m'); exit;
    else
        W0 = options.init.W; 
        H0 = options.init.H;
        Omega0 = options.init.O; 
    end
    if ~isfield(options,'maxiters')
        qmax = 50;
        qWmax = 10;
        qHmax = 10;
        qOmax = 10;
    else
        qmax = options.maxiter.qmax;
        qWmax = options.maxiter.qWmax;
        qHmax = options.maxiter.qHmax;
        qOmax = options.maxiter.qOmax;        
    end
    if ~isfield(options,'maxdeltas')
        delta = 1e-9;
        deltaW = 1e-3;
        deltaH = 1e-3;
        deltaO = 1e-5;
    else
        delta = options.maxdelata.delta;
        deltaW = options.maxdelata.deltaW;
        deltaH = options.maxiter.deltaH;
        deltaO = options.maxiter.deltaO;
    end
    if ~isfield(options,'lambdas')
        printf('Use lambdas provided by mcst_init.m'); exit;
    else
        lambdaAR = options.lambdas.lambdaAR;
        lambdaA = options.lambdas.lambdaA;
    end
    if ~isfield(options,'timemax')
        timemax = Inf;
    else 
        timemax = options.timemax;
    end
    if (sz_L ~= size(Omega0,2))
        printf('Error! No of column of Omega0 are not equal to that of lag set L'); 
        exit;
    end
    %% Initialization.
    W = W0; H = H0; O = Omega0; 
    WtW = W'*W; HHt = H*H'; XHt = X*H'; sumSqX = sum(X.^2,'all');
    time1=tic;
    e = zeros(1, qmax); t = zeros(1, qmax);
    e(1) = sumSqX - 2*sum(W.*XHt,'all') + sum(WtW.*HHt,'all');
    time_err=toc(time1); t(1) = toc(cputime0)-time_err;
    Hlag = zeros(k,sz_L,T);
    HpHpt = zeros(k,sz_L,sz_L);
    Lo = zeros(k,1);
    q = 1; eps = 0; eprev = e(1); epsmin = delta*eprev;
    while(((q == 1) && (t(q) < timemax)) || ...
         ((q <= qmax) && (t(q) < timemax) && (( eps < 0) || ( eps >= epsmin))))
        %% Prepare for update of W
        Lw = norm(HHt); alphaprevW = 1; Wcurr = W; U = W;
        qW = 1; epsinnmin = deltaW*e(q); 
        %% Update W
        while((qW == 1) || ((qW <= qWmax) && ...
                (( epsinn < 0) || ...
                ( epsinn >= epsinnmin))))
            Ak = A*U;
            GradW = 2*(U*HHt - XHt) + 4*lambdaA*(A'*Ak)*(Ak'*Ak - eye(k));
            W = max((U - (1/Lw)*GradW),0);
            alphaW = (1 + sqrt( 4*alphaprevW + 1 ))/2;
            U = W + ((alphaprevW - 1)/(alphaW))*(W - Wcurr);
            UtU = U'*U;
            ecurr = sumSqX - 2*sum(U.*XHt,'all') + sum(UtU.*HHt,'all');
            epsinn = eprev - ecurr; 
            if(epsinn < 0)
                U = W; alphaW = 1; ecurr = eprev;
            end
            Wcurr = W; alphaprevW = alphaW; eprev = ecurr; qW = qW + 1;
        end
        %% Prepare for update of H
        WtW = W'*W; WtX = W'*X;
        [Lg, nrm_Lg] = graph_laplacian(O,L,T,k);
        Lh = norm(WtW) + lambdaAR*sum(nrm_Lg,'all');
        eprev = sumSqX - 2*sum(W.*XHt,'all') + sum(WtW.*HHt,'all');
        alphaprevH = 1; qH = 1; epsinnmin = deltaH*eprev; Hcurr = H; V = H; 
        %% Update H
        while((qH == 1) || ...
                (( qH <= qHmax) && ... 
                (( epsinn < 0) || ...
                ( epsinn >= epsinnmin))))
            GradH = 2*(WtW*V - WtX);
%             for p=1:1:k
%                 Lgp = reshape(Lg(p,:,:),T,T);
%                 GradH(p,:) = GradH(p,:) + lamAR*V(p,:)*(Lgp' + Lgp);
%             end
            GradH = GradH + lambdaAR*(sum(reshape(V,k,1,T).*(permute(Lg, [1,3,2]) + Lg),3));
            H = max((V - (1/Lh)*GradH),0);
            alphaH = (1 + sqrt( 4*alphaprevH + 1 ))/2;
            V = H + ((alphaprevH - 1)/(alphaH))*(H - Hcurr);
            VVt = V*V';
            ecurr = sumSqX - 2*sum(V.*WtX,'all') + sum(WtW.*VVt,'all');
            epsinn = eprev - ecurr;
            if(epsinn < 0)
                V = H; alphaH = 1; ecurr = eprev;
            end
            Hcurr = H; alphaprevH = alphaH; eprev = ecurr; qH = qH + 1;
        end       
        %% Prepare for update of O (Omega)
        HHt = H*H'; XHt = X*H';
        for idx = 1:sz_L 
            Hlag(:,idx,1+L(idx):T) = H(:,1:T-L(idx)); 
        end
        for p=1:1:k
            Hp = reshape(Hlag(p,:,:),sz_L,T);
            temp = Hp*Hp';
            HpHpt(p,:,:) = Hp*Hp';
            Lo(p,1) = norm(temp);
        end
%         e_prev = 0;
%         for p=1:1:k
%             e_prev = e_prev ...
%                 + (H(p,:) - O(p,:)*(reshape(Hlag(p,:,:),sz_L,T)))*(H(p,:) ...
%                 - O(p,:)*(reshape(Hlag(p,:,:),sz_L,T)))';
%             e_prev = e_prev + H(p,:)*H(p,:)' + ...
%                     -2*sum(H(p,:).*(O(p,:)*reshape(Hlag(p,:,:),sz_L,T)),'all') ...
%                     + sum((O(p,:)'*O(p,:)).*reshape(HpHpt(p,:,:),sz_L,sz_L),'all'); 
%         end
        sum_sq_H = sum(H.^2,'all');
        eprev = sum_sq_H ... 
              - 2*sum(reshape(H,k,1,T).*(sum(reshape(O, k, sz_L, 1).*Hlag,2)),'all')...
              + sum((reshape(O, k, 1 , sz_L).*reshape(O, k, sz_L, 1)).*HpHpt,'all');
        alphaprevO = 1; qO = 1; epsinnmin = deltaO*eprev; O_curr = O; P = O;
        %% Update Omega 
        while((qO == 1) || ...
                (( qO <= qOmax) && ... 
                (( epsinn < 0) || ...
                ( epsinn >= epsinnmin))))
%             for p=1:1:k
%                 temp3(p,:) = 2*((P(p,:)*(reshape(HpHpt(p,:,:),sz_L,sz_L))) - H(p,:)*(reshape(Hlag(p,:,:),sz_L,T))');
%                 Gradw = 2*((P(p,:)*(reshape(HpHpt(p,:,:),sz_L,sz_L))) - H(p,:)*(reshape(Hlag(p,:,:),sz_L,T))');
%                 O(p,:) = max((P(p,:) - (1/Lo(p,1))*Gradw),0);
%             end           
            GradO = 2*(sum(reshape(P,k,1,sz_L).*HpHpt,3) - sum(reshape(H,k,1,T).*Hlag,3));
            O = max((P - (Lo.^-1).*GradO),0);
            alphaO = (1 + sqrt( 4*alphaprevO + 1 ))/2;
            P = O + ((alphaprevO - 1)/(alphaO))*(O - O_curr);
            ecurr = sum_sq_H ... 
              - 2*sum(reshape(H,k,1,T).*(sum(reshape(P, k, sz_L, 1).*Hlag,2)),'all')...
              + sum((reshape(P, k, 1 , sz_L).*reshape(P, k, sz_L, 1)).*HpHpt,'all');
            epsinn = eprev - ecurr;           
            if(epsinn < 0)
                P = O; alphaO = 1; ecurr = eprev;
            end
            O_curr = O; alphaprevO = alphaO; eprev = ecurr; qO = qO+1;
        end
        %% Compute error
        time1=tic;
        e(q+1) = sumSqX - 2*sum(W.*XHt,'all') + sum(WtW.*HHt,'all');
        eps = e(q) - e(q+1);
        time_err=time_err + toc(time1); t(q+1) = toc(cputime0)-time_err;
        %% Prepare for the next iteration
        q = q+1;        
    end    
end