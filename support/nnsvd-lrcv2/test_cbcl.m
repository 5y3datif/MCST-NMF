% Example on the CBCL face data set 
% Source: 
% http://cbcl.mit.edu/software-datasets/FaceData2.html 

close all; clear all; clc; 

load cbclim; 
r = 49; % Same value as in the seminal Lee and Seung paper in Nature (1999)
[m,n] = size(X); 
maxiter = 50;

% Compare three SVD-based NMF initializations 
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

% Display the initial basis elements --columns of W
affichage(W1,7,19,19); title('Basis elements NNDSVD'); 
affichage(W2,7,19,19); title('Basis elements SVD-NMF'); 
affichage(W3,7,19,19); title('Basis elements NNSVD-LRC'); 

% DRun A-HALS for the 3 initializations 
[W1n,H1n,e1n] = HALSacc(X,W1,H1,0.5,0.01,100); 
[W2n,H2n,e2n] = HALSacc(X,W2,H2,0.5,0.01,100); 
[W3n,H3n,e3n] = HALSacc(X,W3,H3,0.5,0.01,100); 
[W4n,H4n,e4n] = HALSacc(X,W4,H4,0.5,0.01,100); 

% Display the evolution of the error 
set(0, 'DefaultAxesFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);
figure; 
nX = norm(X,'fro');
semilogx(e1n/nX,'bo-'); hold on; 
semilogx(e2n/nX,'kd-');
semilogx(e3n/nX,'rs-'); 
semilogx(e4n/nX,'m*-'); 
legend('NNDSVD', 'SVD-NMF', 'NNSVD-LRC', 'random'); 
ylabel('relative error ||X-WH||_F/||X||_F'); 
xlabel('iterations'); 
title('A-HALS with different initializations'); 
axis([0 100 0.08 0.2]); 