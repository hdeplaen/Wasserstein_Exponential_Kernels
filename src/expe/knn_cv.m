function score = knn_cv(matrices, indices, param)
% INPUT
% matrices:         structure containing the precomputed distance and label matrices
% ranges:           structure containing the sigma and gamma ranges
% indices:          structure containing the indices
% param:            structure containing different parameters

% COMPUTE KNN

%% PRELIMINARIES
k_max = 10 ;

% indices
idx_tr1 = indices.idx_tr1 ;
idx_tr2 = indices.idx_tr2 ;
idx_val = indices.idx_val ;

n_max = length(idx_tr1) + length(idx_tr2) ;

%% COMPUTE DISTANCE MATRIX
% load data
data1 = load(matrices.D) ;
eval(['D = data1.' matrices.D ' ;']) ;
clear data1 ;

data2 = load(matrices.Dt) ;
eval(['Dt = data2.' matrices.Dt ' ;']) ;
clear data2

% construct matrices
if n_max <= param.gpu_threshold
%     warning('GPU kNN not implemented. Using CPU.') ;
    D_val = D(idx_tr1, idx_val) ;
    D_test = Dt(idx_tr1, :) ;
%     D_val = gpuArray(D(idx_tr1, idx_val)) ;
%     D_test = gpuArray(Dt(idx_tr1, :)) ;
else
    D_val = D(idx_tr1, idx_val) ;
    D_test = Dt(idx_tr1, :) ;
end

clear D Dt

%%%%%%%%%%%%%%%%%%%%

data3 = load(matrices.Y) ;
trY = data3.trY ;
teY = data3.teY ;
clear data3 ;

Y_tr = trY(idx_tr1) ;
Y_val = trY(idx_val) ;
Y_test = teY ;
clear teY trY ;

%% GRID
k_max = 10 ;
k_range = 1:1:k_max ;

n_k = length(k_range) ;
err_val = ones(n_k,1) ;
err_test = ones(n_k,1) ;

parfor idx_k = 1:n_k
    k_loc = k_range(idx_k) ;
    [~,~,err_val(idx_k)] = KNN_(k_loc,Y_tr,Y_val,D_val) ;
    [y(:,idx_k),~,~] = KNN_(k_loc,Y_tr,Y_test,D_test) ;
end

%% VALUES
err_val = 1-err_val ;

min_err = min(min(err_val)) ;
[idx_k] = find(err_val==min_err) ;
idx_k = idx_k(1) ;

% output
cp = classperf(Y_test,y(:,idx_k(1))) ;
best_k = k_range(idx_k) ;

score.cp = cp ;
score.best_k = best_k ;
end
