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
%          W: n by k matrix
%          A: routng matrix of underlying network
%         yt: new observed link flow
%       qmax: maximum number of iterations for repeating updates of ht using
%             fast gradient step.
%       rmax: maximum number of iterations for repeating updates of xt using
%             expectation maximization iteration (EMI) step.
%    deltagd: the parameter to stop iteration of fast gradient step. By
%             default: deltagd = 1-e3 
%   deltaemi: the parameter to stop iteration of expectation maximization 
%             iteration step. By default: deltaemi = 1-e9
% Output
%   xhatt: final estimated OD flow
%       q: fast gradient iterations before exit (q < qmax if an exit occur  
%          due tomeeting stopping criteion for outer loop).
%       r: expectation maximation iterations before exit (r < rmax if an  
%          exit occur due to meeting stopping criteion for outer loop).

function [xhatt,q,r] = mcst_estimation(W,A,yt,qmax,rmax,deltagd,deltaemi)
    if nargin < 4, qmax = 200; end
    if nargin < 5, rmax = 200; end
    if nargin < 6, deltagd = 1e-3; end
    if nargin < 7, deltaemi = 1e-9; end
    n = size(W,1);
    sum_col_A_inv = (sum(A)).^(-1);
    cmptA = A*W; cmptAtcmptA = cmptA'*cmptA; L = norm(cmptAtcmptA);
    y = yt; ht = cmptA'*y;
    cmptAty = cmptA'*y;
    temp = y - cmptA*ht;
    eprev = (temp'*temp);   
    % Fast gradient step.
    q = 1; alphaprev  =1; vt = ht; htcurr = ht; 
    epsgd = 0; epsgdmin = deltagd*(y'*y);
    while((q == 1) || ((q <= qmax) && ((epsgd < 0) || (epsgd >= epsgdmin))))
        % Update ht
        Gradh = 2*(cmptAtcmptA*vt - cmptAty);
        ht = max((vt - (1/L)*Gradh),0);
        alpha = (1 + sqrt( 4*alphaprev + 1 ))/2;
        vt = ht + ((alphaprev - 1)/(alpha))*(ht - htcurr);
        % compute error
        temp = y - cmptA*vt; ecurr = (temp'*temp); epsgd = eprev - ecurr;
        % Restart with reduce learning rate.
        if epsgd < 0
            vt = ht; alpha = 1; ecurr = eprev;
        end         
        % Prepare for the next iteration
        eprev = ecurr; htcurr = ht; alphaprev = alpha; q = q+1;
    end
    
    % Expectation maximization iteration step.
    x = W*ht; r = 1; epsemi = 0; epsemimin = deltaemi*(x'*x);
    while((r == 1) || ...
            ((r <= rmax) && ((epsemi < 0) || (epsemi >= epsemimin))))
        x_prev = x;
        % Replacing zero entries with small numbers
        x = x.*(x > 1e-5) + 1e-5*(x <= 1e-5); 
        % Replacing zero entries with small numbers
        y = y.*(y > 1e-5) + 1e-5*(y <= 1e-5);    
        
        
        
        for idx = 1:1:n
            temp1 = A(:,idx).*y; A_x_inv = (A*x).^(-1);
            x(idx) = sum_col_A_inv(idx)*x(idx)*sum(A_x_inv.*temp1);
        end
        epsemi = (x-x_prev)'*(x-x_prev); r = r + 1;
    end
    xhatt = x;
end
