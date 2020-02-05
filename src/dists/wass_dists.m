function varargout = wass_dists(varargin)
%Computes the Wasserstein distance matrix with a 2 dimensional euclidean
%distance as cost function. The distances are computed between X1 and X2
%and between X1 and itself if X2 is not specified. An additional parameter,
%lambda, controls the precision of the final Wasserstein distance. A high
%value of lambda leads to a better approximation, but takes more time and
%can lead to numerical instabilities. Please refer to the docummentation of
%the Sinkhorn algorithm parfor further details [Cuturi et al., 2013].
%
%D = WASS_DISTS(X1,X2,lambda,single,gpu)
%
%INPUT
%   X1:         ni1 images of size ns*ns (size = ns,ns,ni1);
%   X2:         ni2 images of size ns*ns (size = ns,ns,ni2), [] if X1;
%   lambda:     entropic regularization parfor the Sinkhorn algorithm;
%   equal_mass: boolean if both distributions ought to be of equal mass
%               (default 1);
%   single:     true if all computations ought to be computed with single
%               precision (default is false, so double precision);
%   gpu:        perform computation on GPU;
%   torus:      boolean is the topology is toroidal (default: false);
%   tol:        tolerance for the sinkhorn algorithm (default: 5e-2).
%
%OUTPUT
%   D:          ni1*ni2 symmetric matrix of distances.
%
%Author: HENRI DE PLAEN
%Date: March 2019
%Copyright: KU Leuven


%% PRLIMINARIES
assert((nargin>=3)&&(nargin<=8),  'Wrong number of input arguments') ;
assert(nargout==1, 'Wrong number of output arguments') ;

% input arguments
X1 = varargin{1} ;
X2 = varargin{2} ;
lambda = varargin{3} ;

if nargin<4
    equal_mass = 1 ;
else
    equal_mass = varargin{4} ;
end

if nargin<5
    perform_single = false ;
else
    perform_single = varargin{5} ;
end

if nargin<6
    perform_gpu = true ;
else
    perform_gpu = varargin{6} ;
end

if nargin<7
    torus = false ;
else
    torus = varargin{7} ;
end

if nargin<8
    tol = 5e-2 ;
else
    tol = varargin{8} ;
end

if perform_single
    t_precision = 'single' ;
    X1 = single(X1) ;
    X2 = single(X2) ;
else
    t_precision = 'double' ;
    X1 = double(X1) ;
    X2 = double(X2) ;
end

% sizes
[nsx, nsy, ni1] = size(X1) ;
[~, ~, ni2] = size(X2) ;
nt = nsx*nsy ;

%preprocess images (mapping on probability simplex, ...
% simple normalization of each image)
if equal_mass
    X1_norm = zeros(nsx, nsy, ni1, t_precision) ; % prealloc
    parfor idx_i = 1:ni1
        X_loc = X1(:,:,idx_i) ;
        s = sum(sum(X_loc)) ;
        X1_norm(:,:,idx_i) = X_loc/s ;
    end
    
    if ~isempty(X2)
        X2_norm = zeros(nsx, nsy, ni2, t_precision) ; % prealloc
        parfor idx_i = 1:ni2
            X_loc = X2(:,:,idx_i) ;
            s = sum(sum(X_loc)) ;
            X2_norm(:,:,idx_i) = X_loc/s ;
        end
    end
else
    X1_norm = X1 ;
    if ~isempty(X2)
        X2_norm = X2 ;
    end
end

%% COST MATRIX
if torus
    px = repmat([1:round(nsx/2) (floor(nsx/2)):-1:1],nsy,1) ;
    py = repmat([1:round(nsy/2) (floor(nsy/2)):-1:1]',1,nsx) ;
else
    px = repmat(1:nsx,nsy,1) ;
    py = repmat((1:nsy)',1,nsx) ;
end

px = px(:) ;
py = py(:) ;

px = repmat(px,1,nt) ;
py = repmat(py,1,nt) ;

M = sqrt((px-px').^2 + (py-py').^2) ; % gaussian kernel

%% SINKHORN DISTANCES
if perform_single
    K = single(exp(-lambda*M)) ;
    U = single(K.*M) ;
else
    K = double(exp(-lambda*M)) ;
    U = double(K.*M) ;
end

MOD = 20 ;
verb = false ;

if perform_gpu
    %% GPU
    gpudev = gpuDevice(1) ;
    reset(gpudev) ;
    
    if isempty(X2)
        parfor_progress(ni1,MOD);
        D = zeros(ni1,ni1, t_precision) ;
        for idx_i = 1:ni1
            a = X1_norm(:,:,idx_i) ;
            a = a(:) ;
            
            b = zeros(nt,ni1-idx_i, t_precision) ;
            for idx_i2 = 1:(ni1-idx_i)
                b_loc = X1_norm(:,:,idx_i+idx_i2) ;
                b_loc = b_loc(:) ;
                
                b(:,idx_i2) = b_loc ;
            end
            %reset(gpudev) ;
            a_gpu = gpuArray(a) ;
            b_gpu = gpuArray(b) ;
            K_gpu = gpuArray(K) ;
            U_gpu = gpuArray(U) ;
            D_loc = sinkhornTransport(a_gpu,b_gpu,K_gpu,U_gpu,lambda, ...
                [],[],tol,[],verb) ;
            D(idx_i,:) = [zeros(1,idx_i) gather(D_loc)] ;
            
            if mod(idx_i,MOD)==0
                parfor_progress ;
            end
        end
        D = D+D' ;
        
        parfor_progress(0) ;
    else
        parfor_progress(ni1,MOD);
        D = zeros(ni1,ni2, t_precision) ;
        
        b = zeros(nt,ni2, t_precision) ;
        for idx_i2 = 1:ni2
            b_loc = X2_norm(:,:,idx_i2) ;
            b_loc = b_loc(:) ;
            b(:,idx_i2) = b_loc ;
        end
        for idx_i = 1:ni1
            a = X1_norm(:,:,idx_i) ;
            a = a(:) ;
            
            %reset(gpudev) ;
            a_gpu = gpuArray(a) ;
            b_gpu = gpuArray(b) ;
            K_gpu = gpuArray(K) ;
            U_gpu = gpuArray(U) ;
            D_loc = sinkhornTransport(a_gpu,b_gpu,K_gpu,U_gpu,lambda, ...
                [],[],tol,[],verb) ;
            D(idx_i,:) = gather(D_loc) ;
            
            if mod(idx_i,MOD)==0
                parfor_progress ;
            end
        end
    end
else
    %% CPU
    if isempty(X2)
        parfor_progress(ni1,MOD);
        D = zeros(ni1,ni1, t_precision) ;
        parfor idx_i = 1:ni1
            a = X1_norm(:,:,idx_i) ;
            a = a(:) ;
            
            b = zeros(nt,ni1-idx_i, t_precision) ;
            for idx_i2 = 1:(ni1-idx_i)
                b_loc = X1_norm(:,:,idx_i+idx_i2) ;
                b_loc = b_loc(:) ;
                
                b(:,idx_i2) = b_loc ;
            end
            
            D_loc = sinkhornTransport(a,b,K,U,lambda, ...
                [],[],tol,[],verb) ;
            D(idx_i,:) = [zeros(1,idx_i) D_loc] ;
            
            if mod(idx_i,MOD)==0
                parfor_progress ;
            end
        end
        D = D+D' ;
        
        parfor_progress(0) ;
    else
        parfor_progress(ni1,MOD);
        D = zeros(ni1,ni2, t_precision) ;
        
        b = zeros(nt,ni2, t_precision) ;
        parfor idx_i2 = 1:ni2
            b_loc = X2_norm(:,:,idx_i2) ;
            b_loc = b_loc(:) ;
            b(:,idx_i2) = b_loc ;
        end
        parfor idx_i = 1:ni1
            a = X1_norm(:,:,idx_i) ;
            a = a(:) ;
            
            D_loc = sinkhornTransport(a,b,K,U,lambda, ...
                [],[],tol,[],verb) ;
            D(idx_i,:) = D_loc ;
            
            if mod(idx_i,MOD)==0
                parfor_progress ;
            end
        end
    end
end

%% OUTPUT
varargout{1} = D ;

end

