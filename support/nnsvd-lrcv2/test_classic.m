% Example on the classic document data set; see 
% Zhong, S., Ghosh, J.:  Generative model-based document clustering:  a 
% comparative study.  Knowledge andInformation Systems8 (3), 374–384, 2005.  

close all; clear all; clc; 

load classic; 
r = 25; 
[m,n] = size(X); 
maxiter = 10; 

% Compare three SVD-based NMF initializations and random initialization 
% 1) NNDSVD
fprintf('Running NNDSVD...'), 
tic; 
[W1,H1] = NNDSVD(X,r,1);
fprintf(' Done. Computational time = %2.2f s.\n', toc); 
% 2) SVD-NMF
fprintf('Running SVD-NMF...'), 
tic; 
[W2,H2] = SVDNMF(X,r); 
fprintf(' Done. Computational time = %2.2f s.\n', toc); 
% 3) NNSVDLRC
fprintf('Running NNSVD-LRC...'), 
tic; 
[W3,H3] = NNSVDLRC(X,r); 
fprintf(' Done. Computational time = %2.2f s.\n', toc); 
% 4) Random initialization
W4 = rand(m,r); 
H4 = rand(r,n); 

% Display evolution of A-HALS error for the 3 initializations 
[W1n,H1n,e1n] = HALSacc(X,W1,H1,0.5,0.01,maxiter); 
[W2n,H2n,e2n] = HALSacc(X,W2,H2,0.5,0.01,maxiter); 
[W3n,H3n,e3n] = HALSacc(X,W3,H3,0.5,0.01,maxiter); 
[W4n,H4n,e4n] = HALSacc(X,W4,H4,0.5,0.01,maxiter); 

figure; 
set(0, 'DefaultAxesFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);

nX = norm(X,'fro'); 
plot(e1n/nX,'bo-'); hold on; 
plot(e2n/nX,'kd-');
plot(e3n/nX,'rs-'); 
plot(e4n/nX,'m*-'); 
legend('NNDSVD', 'SVD-NMF', 'NNSVD-LRC', 'random'); 
ylabel('relative error ||X-WH||_F/||X||_F'); 
xlabel('iterations'); 
title('A-HALS with different initializations'); 
axis([0 10 0.88 0.98]); 