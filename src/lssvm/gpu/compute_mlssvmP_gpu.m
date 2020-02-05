function varargout = compute_mlssvmP_gpu(varargin)
%Computes a multi-class LSSVM (primal with feature maps)
%
%model = COMPUTE_MULTI_LSSVM(V, gamma, Y)
%
%INPUT
%   V:      feature vectors (n*dim)
%   gamma:  regularization term
%   Y: encoded labels of each elements of the kernel matrix
%
%OUTPUT
%   model:  LSSVM multi-class model (structure)
%
%Author: HENRI DE PLAEN
%Date: March 2019
%Copyright: KU Leuven

%% PRELIMINARIES

% I/O
assert(nargin==3,  'Wrong number of uinput arguments') ;
assert(nargout==1, 'Wrong number of output arguments') ;

V = varargin{1} ;
gamma = varargin{2} ;
Y = varargin{3} ;


idx = isnan(Y) ;
nan_rmv = sum(idx)~=0 ;
if nan_rmv
    Y = Y(~idx) ;
    V = V(~idx,:) ;
end

%sizes and consistency
[nv,dimv] = size(V) ;
nl = size(Y,1) ;

assert(nv==nl, 'Number of features and labels should be consistent') ;

%% LSSVM SYSTEM

%Construct system (AX = B)
A = [V'*V + 1/gamma*eye(dimv,'single','gpuArray') , sum(V,1)' ; sum(V,1), nv] ;
B = [V'*Y ; sum(Y,1)] ;

clear V gamma Y

%Solve system
X = A\B ;
clear A B

W = X(1:dimv,:) ;
bT = X(dimv+1:end,:) ;

clear X

%% OUTPUT
model.W = W ;
model.bT = bT ;

varargout{1} = model ;

end

