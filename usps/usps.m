% WASSERSTEIN KERNELS (MNIST)
% Author: HENRI DE PLAEN
% Copyright: KU LEUVEN
% Date: January 2019

%% PRELIMINARIES
disp('STARTING EXPERIMENT') ;

clear all ; close all ; clc ;
warning('off','all') ;
reset(gpuDevice) ;

% parameters
lambda = 2.5e+0 ;
p_single = false ;
p_gpu = false ;
grid_s = 15 ;
grid_g = 10 ;

Ntr = 7000 ;
Nte = 2000 ;
%compute_dists_usps ;

n_iter = 5 ;
rng('shuffle') ;

% size of the experiments
Nval = 2000 ;
N1 = 1000 ;
N2 = 1500 ;
assert(length(N1)==length(N2),'N1 and N2 not consistent') ;

save_val = cell(9,length(N1),n_iter) ;

for idx_iter = 1:n_iter
    for idx_n = 1:length(N1)
        %% DATA
        n_tr1 = N1(idx_n) ; n_tr2 = N2(idx_n) ; n_val = Nval ;
        idx_tot = randperm(Ntr,n_tr1+n_tr2+n_val) ;
        idx_tr1 = idx_tot(1:n_tr1) ;
        idx_tr2 = idx_tot(n_tr1+1:n_tr1+n_tr2) ;
        idx_val = idx_tot(n_tr1+n_tr2+1:n_tr1+n_tr2+n_val) ;
        idx_tr = [idx_tr1, idx_tr2] ;
        
        % vals
        indices1.idx_tr1 = idx_tr1 ;
        indices1.idx_tr2 = idx_tr2 ;
        indices1.idx_val = idx_val ;
        
        indices2.idx_tr1 = idx_tr ;
        indices2.idx_tr2 = [] ;
        indices2.idx_val = idx_val ;
        
        rangesW.sigma = 10.^linspace(-1,1.5,grid_s) ; % 25
        rangesW.gamma = 10.^linspace(0,8,grid_g) ; % 10

        rangesWi.sigma = 10.^linspace(-1,1.5,grid_s) ; % 25
        rangesWi.gamma = 10.^linspace(-2,6,grid_g) ; % 10
        
        rangesL2.sigma = 10.^linspace(-.5,2,grid_s) ; % 25
        rangesL2.gamma = 10.^linspace(0,9,grid_g) ; % 10
        
        matricesW.D = 'DW' ;
        matricesW.Dt = 'DtW' ;
        matricesW.Y = 'Y' ;
        
        matricesW.N = 'N2' ;
        matricesW.Nt = 'Nt2' ;
        
        matricesWi = matricesW ;
        matricesWi.N = 'NW' ;
        matricesWi.Nt = 'NtW' ;
        
        matricesL2.D = 'D2' ;
        matricesL2.Dt = 'Dt2' ;
        matricesL2.Y = 'Y' ;
        
        matricesL2.N = 'N2' ;
        matricesL2.Nt = 'Nt2' ;
        
        params_d.gpu_threshold = 3000 ;
        params_d.indef = false ;
        params_d.coding = 'code_OneVsOne' ;
        params_d.store_threshold = 1.5e+4 ;
        params_d.single = false ;
        
        params_i.gpu_threshold = 3000 ;
        params_i.indef = true ;
        params_i.coding = 'code_OneVsOne' ;
        params_i.store_threshold = 1.5e+4 ;
        params_i.single = false ;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% NO INK KERNELS
        
        % WASS (CORE+OOS)
        str = ['WASS (CORE+OOS) ', num2str(n_tr1), ' ', num2str(n_tr2)] ;
        diary on ; disp(str) ; diary off ;
        tic ;
        
        save_val{1,idx_n,idx_iter} = mlssvm_cv(matricesW,rangesW,indices1,params_d) ;
        score = save_val{1,idx_n,idx_iter}.cp ;
        diary on ; disp(['Execution time: ' num2str(toc/60) 'min']) ; diary off ;
        diary on ; disp(['Misclass. error: ' num2str((1-sum(diag(score.CountingMatrix))/sum(sum(score.CountingMatrix)))*100) '%']) ; diary off ;
        
        save('vals_mnist.mat','save_val','-v7.3') ;
        
        diary on ; disp(newline) ; diary off ;
        
        % WASS (CORE)
        diary on ;
        str = ['WASS (CORE) ', num2str(n_tr1+n_tr2), ' ', 0] ;
        diary on ; disp(str) ; diary off ;
        
        tic ;
        save_val{2,idx_n,idx_iter} = mlssvm_cv(matricesW,rangesW,indices2,params_d) ;
        score = save_val{2,idx_n,idx_iter}.cp ;
        diary on ; disp(['Execution time: ' num2str(toc/60) 'min']) ; diary off ;
        diary on ; disp(['Misclass. error: ' num2str((1-sum(diag(score.CountingMatrix))/sum(sum(score.CountingMatrix)))*100) '%']) ; diary off ;
        
        save('vals_mnist.mat','save_val','-v7.3') ;
        
        diary on ; disp(newline) ; diary off ;
        
        % WASS (INDEF)
        str = ['WASS (INDEF LSSVM) ', num2str(n_tr1+n_tr2), ' ', 0] ;
        diary on ; disp(str) ; diary off ;
        tic ;
        save_val{3,idx_n,idx_iter} = mlssvm_cv(matricesW,rangesW,indices2,params_i) ;
        score = save_val{3,idx_n,idx_iter}.cp ;
        diary on ; disp(['Execution time: ' num2str(toc/60) 'min']) ; diary off ;
        diary on ; disp(['Misclass. error: ' num2str((1-sum(diag(score.CountingMatrix))/sum(sum(score.CountingMatrix)))*100) '%']) ; diary off ;
        
        save('vals_mnist.mat','save_val','-v7.3') ;
        
        diary on ; disp(newline) ; diary off ;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% INK KERNELS
        
        % WASS (CORE+OOS)
        str = ['WASS (CORE+OOS) with ink ', num2str(n_tr1), ' ', num2str(n_tr2)] ;
        diary on ; disp(str) ; diary off ;
        tic ;
        
        save_val{7,idx_n,idx_iter} = mlssvm_cv(matricesWi,rangesWi,indices1,params_d) ;
        score = save_val{7,idx_n,idx_iter}.cp ;
        diary on ; disp(['Execution time: ' num2str(toc/60) 'min']) ; diary off ;
        diary on ; disp(['Misclass. error: ' num2str((1-sum(diag(score.CountingMatrix))/sum(sum(score.CountingMatrix)))*100) '%']) ; diary off ;
        
        save('vals_mnist.mat','save_val','-v7.3') ;
        
        diary on ; disp(newline) ; diary off ;
        
        % WASS (CORE)
        str = ['WASS (CORE) with ink ', num2str(n_tr1+n_tr2), ' ', 0] ;
        diary on ; disp(str) ; diary off ;
        
        tic ;
        save_val{8,idx_n,idx_iter} = mlssvm_cv(matricesWi,rangesWi,indices2,params_d) ;
        score = save_val{8,idx_n,idx_iter}.cp ;
        diary on ; disp(['Execution time: ' num2str(toc/60) 'min']) ; diary off ;
        diary on ; disp(['Misclass. error: ' num2str((1-sum(diag(score.CountingMatrix))/sum(sum(score.CountingMatrix)))*100) '%']) ; diary off ;
        
        save('vals_mnist.mat','save_val','-v7.3') ;
        
        diary on ; disp(newline) ; diary off ;
        
        % WASS (INDEF)
        str = ['WASS (INDEF LSSVM) with ink ', num2str(n_tr1+n_tr2), ' ', 0] ;
        diary on ; disp(str) ; diary off ;
        tic ;
        save_val{9,idx_n,idx_iter} = mlssvm_cv(matricesWi,rangesWi,indices2,params_i) ;
        score = save_val{9,idx_n,idx_iter}.cp ;
        diary on ; disp(['Execution time: ' num2str(toc/60) 'min']) ; diary off ;
        diary on ; disp(['Misclass. error: ' num2str((1-sum(diag(score.CountingMatrix))/sum(sum(score.CountingMatrix)))*100) '%']) ; diary off ;
        
        save('vals_mnist.mat','save_val','-v7.3') ;
        
        diary on ; disp(newline) ; diary off ;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% BENCHMARK METHODS
        
        % WASS (kNN)
        str = ['WASS (kNN) ', num2str(n_tr1+n_tr2), ' ', 0] ;
        diary on ; disp(str) ; diary off ;
        tic ;
        
        save_val{4,idx_n,idx_iter} = knn_cv(matricesW,indices2,params_i) ;
        score = save_val{4,idx_n,idx_iter}.cp ;
        diary on ; disp(['Execution time: ' num2str(toc/60) 'min']) ; diary off ;
        diary on ; disp(['Misclass. error: ' num2str((1-sum(diag(score.CountingMatrix))/sum(sum(score.CountingMatrix)))*100) '%']) ; diary off ;
        
        save('vals_mnist.mat','save_val','-v7.3') ;
        
        diary on ; disp(newline) ; diary off ;
        
        % L2
        str = ['L2 (LSSVM) ', num2str(n_tr1+n_tr2), ' ', 0] ;
        diary on ; disp(str) ; diary off ;
        tic ;
        
        save_val{5,idx_n,idx_iter} = mlssvm_cv(matricesL2,rangesL2,indices2,params_i) ;
        score = save_val{5,idx_n,idx_iter}.cp ;
        diary on ; disp(['Execution time: ' num2str(toc/60) 'min']) ; diary off ;
        diary on ; disp(['Misclass. error: ' num2str((1-sum(diag(score.CountingMatrix))/sum(sum(score.CountingMatrix)))*100) '%']) ; diary off ;
        
        save('vals_mnist.mat','save_val','-v7.3') ;
        
        diary on ; disp(newline) ; diary off ;

        % L2 (kNN)
        str = ['L2 (kNN) ', num2str(n_tr1+n_tr2), ' ', 0] ;
        diary on ; disp(str) ; diary off
        tic ;
        
        save_val{6,idx_n,idx_iter} = knn_cv(matricesL2, indices2, params_i) ;
        score = save_val{6,idx_n,idx_iter}.cp ;
        diary on ; disp(['Execution time: ' num2str(toc/60) 'min']) ; diary off ;
        diary on ; disp(['Misclass. error: ' num2str((1-sum(diag(score.CountingMatrix))/sum(sum(score.CountingMatrix)))*100) '%']) ; diary off ;
        
        save('vals_mnist.mat','save_val','-v7.3') ;
        
        diary on ; disp(newline) ; diary off ;

        diary on ; disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%') ; diary off ;
    end
end
