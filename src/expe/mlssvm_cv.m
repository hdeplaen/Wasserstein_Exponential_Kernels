function [score] = mlssvm_cv(matrices, ranges, indices,param)
% INPUT
% matrices:         structure containing the precomputed distance and label matrices
% ranges:           structure containing the sigma and gamma ranges
% indices:          structure containing the indices
% param:            structure containing different parameters

% COMPUTE LS-SVM

%% PRELIMINARIES
reset(gpuDevice) ;
perform_single = param.single ;

% indices
idx_tr1 = indices.idx_tr1 ;
idx_tr2 = indices.idx_tr2 ;
idx_val = indices.idx_val ;

% ranges
sigma_range = ranges.sigma ;
gamma_range = ranges.gamma ;
n_sigma = length(sigma_range) ;
n_gamma = length(gamma_range) ;

%% COMPUTE DISTANCE MATRIX
% load data
data1 = load(matrices.D) ;
eval(['D = data1.' matrices.D ' ;']) ;
clear data1 ;

data3 = load(matrices.Y) ;
trY = data3.trY ;
teY = data3.teY ;
clear data3 ;

data4 = load(matrices.N) ;
eval(['N = data4.' matrices.N ' ;']) ;
clear data4 ;

n_tr1 = length(idx_tr1) ;
n_tr2 = length(idx_tr2) ;
n_val = length(idx_val) ;

%construct matrices
if perform_single
    D_tr1 = single(D(idx_tr1,idx_tr1)) ;
    D_tr2 = single(D(idx_tr1,idx_tr2)) ;
    D_val = single(D(idx_tr1, idx_val)) ;
    
    N_tr1 = single(N(idx_tr1)) ;
    N_tr2 = single(N(idx_tr2)) ;
    N_val = single(N(idx_val)) ;
else
    D_tr1 = D(idx_tr1,idx_tr1) ;
    D_tr2 = D(idx_tr1,idx_tr2) ;
    D_val = D(idx_tr1, idx_val) ;
    
    N_tr1 = N(idx_tr1) ;
    N_tr2 = N(idx_tr2) ;
    N_val = N(idx_val) ;
end

% gpu
if n_tr1+n_tr2 <= param.gpu_threshold
    D_tr1 = gpuArray(D_tr1) ;
    D_tr2 = gpuArray(D_tr2) ;
    D_val = gpuArray(D_val) ;
    
    N_tr1 = gpuArray(N_tr1) ;
    N_tr2 = gpuArray(N_tr2) ;
    N_val = gpuArray(N_val) ;
end

if n_tr1+n_tr2 > param.store_threshold
    delete _temp.mat
    save('_temp.mat','D_tr1','D_tr2','D_val','-v7.3') ;
    clear D_tr1 D_tr2 D_val
end

clear D Dt

% correcting for the curse of dimensionality
gamma_range = gamma_range./(n_tr1+n_tr2) ;

% labels & encoding
Y_tr1_decoded = trY(idx_tr1) ;
Y_tr2_decoded = trY(idx_tr2) ;
Y_val_decoded = trY(idx_val) ;

Y_decoded = [Y_tr1_decoded ; Y_tr2_decoded] ;
[Y_encoded, codebook, old_codebook] = code(Y_decoded,param.coding) ;
Y_val_encoded = code(Y_val_decoded, codebook) ;
n_problems = size(Y_encoded,2) ;

%% TUNE param
best_value = ones(n_problems,1) ;
best_gamma = zeros(n_problems,1) ;
best_sigma = zeros(n_problems,1) ;
best_model = cell(n_problems,1) ;
best_VW = cell(n_problems,1) ;
best_lw = cell(n_problems,1) ;

if n_tr1+n_tr2 > param.store_threshold
    save('_model.mat','best_model','best_gamma','best_sigma','best_VW','best_lw','-v7.3') ;
    clear best_model best_VW best_lw ;
end

%ppm = ParforProgressbar(n_problems*n_sigma*n_gamma,'title','LS-SVM Tuning', 'progressBarUpdatePeriod', 5);
parfor_progress(n_problems*n_sigma*n_gamma) ;

for idx_gamma = 1:n_gamma
    gamma = gamma_range(idx_gamma) ;
    for idx_sigma = 1:n_sigma
        sigma2 = sigma_range(idx_sigma) ;
        for idx_problem = 1:n_problems
            %% KERNEL
            if param.indef
                if n_tr1+n_tr2 > param.store_threshold
                    data = load('_temp.mat','D_tr1') ;
                    D_tr1 = data.D_tr1 ;
                    clear data ;
                end
                K_tr = (N_tr1*N_tr1').*exp(-D_tr1.^2/sigma2) ;
                if n_tr1+n_tr2 > param.store_threshold
                    clear D_tr1 ;
                end
                
                if n_tr1+n_tr2 <= param.gpu_threshold
                    model = compute_mlssvmD_gpu(K_tr, gamma, gpuArray(Y_encoded(:,idx_problem))) ;
                else
                    model = compute_mlssvmD_cpu(K_tr, gamma, Y_encoded(:,idx_problem)) ;
                end
            else
                % truncate
                if n_tr1+n_tr2 > param.store_threshold
                    data = load('_temp.mat','D_tr1') ;
                    D_tr1 = data.D_tr1 ;
                    clear data ;
                end
                K_tr1 = (N_tr1*N_tr1').*exp(-D_tr1.^2/sigma2) ;
                if n_tr1+n_tr2 > param.store_threshold
                    clear D_tr1 ;
                end
                [VW,LW] = eig(K_tr1) ;
                lw = diag(LW) ;
                idx_lw_pos = lw > 1e-6 ;
                lw = lw(idx_lw_pos) ;
                VW = VW(:,idx_lw_pos) ;
                VlW_tr1 = (VW'*K_tr1)./repmat(sqrt(lw),1,n_tr1) ;
                if n_tr1+n_tr2 > param.store_threshold
                    clear LW K_tr1 ;
                end
                
                % kernel
                if n_tr1+n_tr2 > param.store_threshold
                    data = load('_temp.mat','D_tr2') ;
                    D_tr2 = data.D_tr2 ;
                    clear data ;
                end
                
                if length(N_tr2) == 0
                    K_tr2 = exp(-D_tr2.^2/sigma2) ;
                else
                    K_tr2 = (N_tr1*N_tr2').*exp(-D_tr2.^2/sigma2) ;
                end
                
                if n_tr1+n_tr2 > param.store_threshold
                    clear D_tr2 ;
                end
                VlW_tr2 = (VW'*K_tr2)./repmat(sqrt(lw),1,n_tr2) ;
                if n_tr1+n_tr2 > param.store_threshold
                    clear K_tr2
                end
                VlW = [VlW_tr1, VlW_tr2]' ;
                if n_tr1+n_tr2 > param.store_threshold
                    clear VlW_tr1 VlW_tr2 ;
                end
                
                % train
                if n_tr1+n_tr2 <= param.gpu_threshold
                    model = compute_mlssvmP_gpu(VlW, gamma, gpuArray(Y_encoded(:,idx_problem))) ;
                else
                    model = compute_mlssvmP_cpu(VlW, gamma, Y_encoded(:,idx_problem)) ;
                end
                
                if n_tr1+n_tr2 > param.store_threshold
                    clear VlW ;
                end
            end
            
            %% VAL
            if param.indef
                % evaluate
                if n_tr1+n_tr2 > param.store_threshold
                    data = load('_temp.mat','D_val') ;
                    D_val = data.D_val ;
                    clear data ;
                end
                K_val = (N_tr1*N_val').*exp(-D_val.^2/sigma2) ;
                if n_tr1+n_tr2 > param.store_threshold
                    clear D_val ;
                end
                if n_tr1+n_tr2 <= param.gpu_threshold
                    Y_val_lssvm_encoded = gather(eval_mlssvmD_gpu(model, K_val)) ;
                else
                    Y_val_lssvm_encoded = eval_mlssvmD_cpu(model, K_val) ;
                end
                
                if n_tr1+n_tr2 > param.store_threshold
                    clear K_val ;
                end
                
                % scores
                err_val = 1-sum((Y_val_lssvm_encoded-Y_val_encoded(:,idx_problem))==0)/length(Y_val_decoded) ;
                if err_val < best_value(idx_problem)
                    best_value(idx_problem) = err_val ;
                    best_sigma(idx_problem) = sigma2 ;
                    best_gamma(idx_problem) = gamma ;
                    
                    if n_tr1+n_tr2 > param.store_threshold
                        data = load('_model.mat') ;
                        best_model = data.best_model ;
                        clear data ;
                    end
                    
                    best_model{idx_problem} = model ;
                    
                    if n_tr1+n_tr2 > param.store_threshold
                        save('_model.mat','best_model','best_sigma','best_gamma','-v7.3') ;
                        clear best_model best_VW best_lw ;
                    end
                end
            else
                % evaluate
                if n_tr1+n_tr2 > param.store_threshold
                    data = load('_temp.mat','D_val') ;
                    D_val = data.D_val ;
                    clear data ;
                end
                K_val = (N_tr1*N_val').*exp(-D_val.^2/sigma2) ;
                if n_tr1+n_tr2 > param.store_threshold
                    clear D_val ;
                end
                V_val = (K_val'*VW)./repmat(sqrt(lw)',n_val,1) ;
                if n_tr1+n_tr2 > param.store_threshold
                    clear K_val
                end
                Y_val_lssvm_encoded = gather(eval_mlssvmP_gpu(model, V_val)) ;
                if n_tr1+n_tr2 > param.store_threshold
                    clear V_val
                end
                
                % scores
                err_val = 1-sum((Y_val_lssvm_encoded-Y_val_encoded(:,idx_problem))==0)/length(Y_val_decoded) ;
                if err_val < best_value(idx_problem)
                    best_value(idx_problem) = err_val ;
                    best_sigma(idx_problem) = sigma2 ;
                    best_gamma(idx_problem) = gamma ;
                    
                    if n_tr1+n_tr2 > param.store_threshold
                        data = load('_model.mat') ;
                        best_model = data.best_model ;
                        best_VW = data.best_VW ;
                        best_lw = data.best_lw ;
                        clear data ;
                    end
                    
                    best_model{idx_problem} = model ;
                    best_VW{idx_problem} = VW ;
                    best_lw{idx_problem} = lw ;
                    
                    if n_tr1+n_tr2 > param.store_threshold
                        save('_model.mat','best_model','best_sigma','best_gamma','best_VW','best_lw','-v7.3') ;
                        clear best_model best_VW best_lw ;
                    end
                end
                
                if n_tr1+n_tr2 > param.store_threshold
                    clear VW lw ;
                end
            end
            
            if n_tr1+n_tr2 > param.store_threshold
                clear model
            end
            
            %disp('hello') ;
            %ppm.increment() ;
            parfor_progress;
        end
    end
end
parfor_progress(0) ;



%% BEST SCORES
%DATA
data2 = load(matrices.Dt) ;
eval(['Dt = data2.' matrices.Dt ' ;']) ;
clear data2 ;

data5 = load(matrices.Nt) ;
eval(['Nt = data5.' matrices.Nt ' ;']) ;
clear data5 ;

n_test = size(Dt,2) ;

%construct matrices
if perform_single
    D_test = single(Dt(idx_tr1, :)) ;
    N_test = single(Nt) ;
else
    D_test = Dt(idx_tr1, :) ;
    N_test = Nt ;
end

% gpu
if n_tr1+n_tr2 <= param.gpu_threshold
    D_test = gpuArray(D_test) ;
    N_test = gpuArray(N_test) ;
end

% test
if n_tr1+n_tr2 > param.store_threshold
    data = load('_model.mat') ;
    best_model = data.best_model ;
    if ~param.indef
        best_VW = data.best_VW ;
        best_lw = data.best_lw ;
    end
    best_sigma = data.best_sigma ;
    best_gamma = data.best_gamma ;
    clear data ;
end

Y_test_decoded = teY ;
Y_test_lssvm_encoded = zeros(n_test,n_problems) ;

% COMPUTATION
for idx_problem = 1:n_problems
    sigma2 = best_sigma(idx_problem) ;
    if param.indef
        % evaluate
        if n_tr1+n_tr2 > param.store_threshold
            data = load('_temp.mat','D_test') ;
            D_test = data.D_test ;
            clear data ;
        end
        K_test = (N_tr1*N_test').*exp(-D_test.^2/sigma2) ;
        if n_tr1+n_tr2 > param.store_threshold
            clear D_test ;
        end
        if n_tr1+n_tr2 <= param.gpu_threshold
            Y_test_lssvm_encoded(:,idx_problem) = gather(eval_mlssvmD_gpu(best_model{idx_problem}, K_test)) ;
        else
            Y_test_lssvm_encoded(:,idx_problem) = eval_mlssvmD_cpu(best_model{idx_problem}, K_test) ;
        end
        if n_tr1+n_tr2 > param.store_threshold
            clear K_test ;
        end
    else
        % evaluate
        if n_tr1+n_tr2 > param.store_threshold
            data = load('_temp.mat','D_test') ;
            D_test = data.D_test ;
            clear data ;
        end
        K_test = (N_tr1*N_test').*exp(-D_test.^2/sigma2) ;
        if n_tr1+n_tr2 > param.store_threshold
            clear D_test ;
        end
        VW = best_VW{idx_problem} ;
        lw = best_lw{idx_problem} ;
        V_test = (K_test'*VW)./repmat(sqrt(lw)',n_test,1) ;
        if n_tr1+n_tr2 > param.store_threshold
            clear K_test
        end
        
        if n_tr1+n_tr2 <= param.gpu_threshold
            Y_test_lssvm_encoded(:,idx_problem) = gather(eval_mlssvmP_gpu(best_model{idx_problem}, V_test)) ;
        else
            Y_test_lssvm_encoded(:,idx_problem) = eval_mlssvmP_cpu(best_model{idx_problem}, V_test) ;
        end
        if n_tr1+n_tr2 > param.store_threshold
            clear V_test
            clear VW lw ;
        end
    end
end

Y_test_lssvm_decoded = code(gather(Y_test_lssvm_encoded),old_codebook,[],codebook,'codedist_hamming') ;
Y_test_lssvm_decoded(Y_test_lssvm_decoded==-Inf) = NaN ;

% output
cp = classperf(Y_test_decoded,Y_test_lssvm_decoded) ;

score.cp = cp ;
score.best_sigmas = best_sigma ;
score.best_gammas = best_gamma*(n_tr1+n_tr2) ;
end

