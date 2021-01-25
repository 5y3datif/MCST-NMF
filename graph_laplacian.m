%% Compute k Graph Laplacian.
% Reference: S M Atif, S Qazi, N Gillis, I Naseem. "Structured Nonnegative 
% Matrix Factorization for Traffic Flow Estimation of Large Cloud Network".
%
% Written by Syed Muhammad Atif, Karachi Institute of Economics and 
% Technology, Karachi, Pakistan.
% Latest update December 2020
%
% Input:
%     O: Omega matrix
%     L: lag set
%     k: factorization rank
%     T: number of training timestamps
%   eta: a small number between 0 and 1 (exclusive). By default: eta = 1e-3
% Output: 
%      Lg: k T by T graph laplacian matrices.
%   nrmLg: corresponding Frobenius norm. normest is used faster computation
%          as graph laplacian matrices are sparse. 
function [Lg,nrmLg] = graph_laplacian(O,L,T,k,eta)
    if nargin < 5, eta = 1e-3; end
    szL = size(L,2);
    maxL = max(L);
    Lbar = zeros(1,(szL+1));
    Lbar(2:szL+1) = L;
    Oext = zeros(k,(szL+1));
    Oext(:,1) = -1*ones(k,1);
    Oext(:,2:szL+1) = O;
    sumOext = sum(Oext,2);
    GARidx = zeros(floor(0.1*T*T),2);
    GARval = zeros(floor(0.1*T*T),k);
    Didx = zeros(T,2);
    Dval = zeros(T,k);
    Lg = cell(k,1);
    nrmLg = zeros(1,k);
    % Similarity and diagonal matrices.
    tic;
    cnt1 = 1;
    cnt2 = 1;
    for t=1:1:T
        ptt_l = Lbar(1,((Lbar > (maxL-t)) & (Lbar <= (T-t))));
        for d=1:1:maxL
            t1 = t+d;
            if(t1<=T)
                ptt_l_minus_d = ptt_l-d;
                [l_minus_d,l_minus_d_idx,~] = intersect(Lbar,ptt_l_minus_d);
                if(~isempty(l_minus_d))
                    l = l_minus_d + d;
                    [~,l_idx,~] = intersect(Lbar,l);
                    GARidx(cnt1,:) = [t,t1];
                    GARval(cnt1,:) = -1*sum(Oext(:,l_idx).*Oext(:,l_minus_d_idx),2);
                    cnt1 = cnt1 + 1;
                end
            end
        end
        l = Lbar(1,((Lbar > (maxL-t)) & (Lbar <= (T-t))));
        [~,l_idx,~] = intersect(Lbar,l);
        Didx(cnt2,:) = [t,t];
        Dval(cnt2,:) = sumOext.*sum(Oext(:,l_idx),2);
        cnt2 = cnt2 + 1;
    end
    % Graph laplacian matrices and their corresponding Frobenius norm
    for p=1:1:k
        temp2 = sparse(GARidx(1:(cnt1-1),1), GARidx(1:(cnt1-1),2), GARval(1:(cnt1-1),p),T,T);
        temp3 = sparse(Didx(1:(cnt2-1),1), Didx(1:(cnt2-1),2), Dval(1:(cnt2-1),p),T,T);
        temp4 = diag(sum(temp2,2)) -1*(temp2 - diag(diag(temp2))) + ...
                    temp3 + eta*speye(T,T);
        nrmLg(1,p) = normest(temp4,1e-2);
        Lg{p} = temp4;
    end
end
