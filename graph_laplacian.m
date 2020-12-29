%% Compute k Graph Laplacian.
% Reference: S M Atif, S Qazi, N Gillis, I Naseem. "Structured Nonnegative 
% Matrix Factorization for Traffic Flow Estimation of Large Cloud Network".
%
% Written by Syed Muhammad Atif, Karachi Institute of Economics and 
% Technology, Karachi, Pakistan.
% Latest update December 2020
%
% Graph laplacian matrices are sparse. This is not a very efficient 
% implementation. One may consider an alternative implementation by
% considering sparsity in the matrices. 
%
% Input
%     O: Omega matrix
%     L: lag set
%     k: factorization rank
%     T: number of training timestamps
%   eta: a small number between 0 and 1 (exclusive). By default: eta = 1e-3
% Output: 
%      Lg: k T by T graph laplacian matrices.
%   nrmLg: corresponding Frobenius norm. normest is used faster computation
%          as graph laplacian matrices are sparse. 
function [Lg,nrm_Lg] = graph_laplacian(O,L,T,k,eta)
    if nargin < 5, eta = 1e-3; end
    sz_L = size(L,2);
    max_L = max(L);
    L_bar = zeros(1,(sz_L+1));
    L_bar(2:sz_L+1) = L;
    O_ext = zeros(k,(sz_L+1));
    O_ext(:,1) = -1*ones(k,1);
    O_ext(:,2:sz_L+1) = O;
    sum_O_ext = sum(O_ext,2);
    G_AR = zeros(k,T,T);
    D = zeros(k,T,T);
    Lg = zeros(k,T,T);
    nrm_Lg =zeros(1,k);
    % Similarity and diagonal matrices.
    for t=1:1:T
        ptt_l = L_bar(1,((L_bar > (max_L-t)) & (L_bar <= (T-t))));
        for d=1:1:max_L
            t1 = t+d;
            if(t1<=T)
                ptt_l_minus_d = ptt_l-d;
                [l_minus_d,l_minus_d_idx,~] = intersect(L_bar,ptt_l_minus_d);
                if(~isempty(l_minus_d))
                    l = l_minus_d + d;
                    [~,l_idx,~] = intersect(L_bar,l);
                    G_AR(:,t,t1) = G_AR(:,t,t1) - ...
                        sum(O_ext(:,l_idx).*O_ext(:,l_minus_d_idx),2);
                end
            end
        end
        l = L_bar(1,((L_bar > (max_L-t)) & (L_bar <= (T-t))));
        [~,l_idx,~] = intersect(L_bar,l);
        D(:,t,t) = D(:,t,t) + sum_O_ext.*sum(O_ext(:,l_idx),2);
    end
    % Graph laplacian matrices and their corresponding Frobenius norm
    for p=1:1:k
        temp = reshape(G_AR(p,:,:),T,T);
        temp1 = diag(sum(temp,2)) -1*(temp - diag(diag(temp))) + ...
            reshape(D(p,:,:),T,T) + eta*eye(T,T);
        nrm_Lg(1,p) = normest(temp1,1e-2);
        Lg(p,:,:) = temp1;
    end
end