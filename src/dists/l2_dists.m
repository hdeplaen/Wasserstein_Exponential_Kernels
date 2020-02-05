function varargout = l2_dists(varargin)
%Computes the L2 distance matrix with or without normalization
%pre-processing.
%
%D = L2_DISTS(X1,X2,nrm)
%
%INPUT
%   X1:         ni1 images of size ns*ns (size = ns,ns,ni1)
%   X2:         ni2 images of size ns*ns (size = ns,ns,ni2), [] if X1
%   nrm:        normalization (boolean)
%
%OUTPUT
%   D:          ni1*ni2 symmetric matrix of distances
%
%Author: HENRI DE PLAEN
%Date: March 2019
%Copyright: KU Leuven


%% PRLIMINARIES
assert((nargin>=3)&&(nargin<=4),  'Wrong number of input arguments') ;
assert(nargout==1, 'Wrong number of output arguments') ;

if nargin<4
    perform_single = false ;
else
    perform_single = varargin{4} ;
end

X1 = varargin{1} ;
X2 = varargin{2} ;
nrm = varargin{3} ;

[nsx, nsy, ni1] = size(X1) ;
[~, ~, ni2] = size(X2) ;
% assert(nsx==nsy, 'Only implemented for square images now') ;
nt = nsx*nsy ;

%preprocess images (mapping on probability simplex, ...
% simple normalization of each image)
if nrm
    X1_norm = zeros(nsx*nsy, ni1) ; % prealloc
    for idx_i = 1:ni1
        X_loc = X1(:,:,idx_i) ;
        s = sum(sum(X_loc)) ;
        X1_norm(:,idx_i) = X_loc(:)/s ;
    end
    
    if ~isempty(X2)
        X2_norm = zeros(nsx*nsy, ni2) ; % prealloc
        for idx_i = 1:ni2
            X_loc = X2(:,:,idx_i) ;
            s = sum(sum(X_loc)) ;
            X2_norm(:,idx_i) = X_loc(:)/s ;
        end
    end
else
    X1_norm = zeros(nsx*nsy, ni1) ; % prealloc
    for idx_i = 1:ni1
        X_loc = X1(:,:,idx_i) ;
        X1_norm(:,idx_i) = X_loc(:) ;
    end
    
    if ~isempty(X2)
        X2_norm = zeros(nsx*nsy, ni2) ; % prealloc
        for idx_i = 1:ni2
            X_loc = X2(:,:,idx_i) ;
            X2_norm(:,idx_i) = X_loc(:) ;
        end
    end
end

%% L2 DISTANCES
if ~isempty(X2)
    D = pdist2(X1_norm',X2_norm','euclidean') ;
else
    P = pdist(X1_norm','euclidean') ;
    D = squareform(P) ;
end

%% OUTPUT
if perform_single
    varargout{1} = single(D) ;
else
    varargout{1} = double(D) ;
end


end

