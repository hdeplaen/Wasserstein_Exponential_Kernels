function varargout = compute_mlssvmD_cpu(varargin)
%Computes a multi-class LSSVM (dual with kernel) in the one-against-all
%framework.
%
%model = COMPUTE_MULTI_LSSVM(K, gamma, labels)
%
%INPUT
%   K:      kernel matrix
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

K = varargin{1} ;
gamma = varargin{2} ;
Y = varargin{3} ;

idx = ~isnan(Y) ;
Y = Y(idx) ;
K = K(idx,idx) ;

%sizes and consistency
[nx,ny] = size(K) ;
nl = size(Y,2) ;

assert(nx==ny, 'Kernel matrix should be square') ;

%% LSSVM SYSTEM
A = [K + 1/gamma*eye(nx,'single'), ones(nx,1,'single') ; ones(1,ny,'single'), 0] ;
B = [Y; zeros(1,nl,'single')] ;
clear K Y ;

X = A\B ;
clear A B ;

HT = X(1:nx,:) ;
bT = X(nx+1:end,:) ;

%% OUTPUT
model.HT = HT ;
model.bT = bT ;
model.idx = idx ;

varargout{1} = model ;

end

% %% OLD
% %Construct system (Ax = b)
% Y_temp = cell(nl,1) ;     % prealloc
% W_temp = cell(nl,1);      % prealloc
% for idx_l = 1:nl
%     Y_temp{idx_l} = sparse(Y_label(:,idx_l)) ;
%     W_temp{idx_l} = sparse((Y_temp{idx_l}*Y_temp{idx_l}').*K + (1/gamma)*eye(nix,nix)) ;
% end
% Y = blkdiag(Y_temp{:}) ;   % block diagram matrix
% W = blkdiag(W_temp{:}) ;   % block diagram matrix
% Z = sparse(nl,nl) ;         % prealloc
% clear W_temp
% 
% A = [Z, Y'; Y, W] ;
% clear Z Y W
% b = [zeros(nl,1) ; ones(nl*nix,1)] ;
% 
% %Solve system
% x = A\b ;
% clear A b
% 
% intercept = zeros(nl,1) ;  % prealloc
% alpha = zeros(nix,nl) ;    % prealloc
% 
% for idx_l = 1:nl
%     intercept(idx_l) = x(idx_l) ;
%     alpha(:,idx_l) = x((nl+(idx_l-1)*nix)+(1:nix)) ;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %% PRELIMINARIES
% 
% % I/O
% assert(nargin==3,  'Wrong number of uinput arguments') ;
% assert(nargout==1, 'Wrong number of output arguments') ;
% 
% K = gather(varargin{1}) ;
% gamma = varargin{2} ;
% Y = gather(varargin{3}) ;
% 
% %sizes and consistency
% [nx,ny] = size(K) ;
% nl = size(Y,2) ;
% 
% assert(nx==ny, 'Kernel matrix should be square') ;
% 
% %% LSSVM SYSTEM
% A = [K + 1/gamma*eye(nx), ones(nx,1) ; ones(1,ny), 0] ;
% B = [Y; zeros(1,nl)] ;
% clear K Y ;
% 
% X = A\B ;
% clear A B ;
% 
% HT = X(1:nx,:) ;
% bT = X(nx+1:end,:) ;
% 
% %% OUTPUT
% model.HT = HT ;
% model.bT = bT ;
% 
% varargout{1} = model ;