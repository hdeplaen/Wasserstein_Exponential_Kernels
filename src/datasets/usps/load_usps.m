function varargout = load_usps(varargin)
%United States Postal Service
%Digits dataset
%
%   [X, Y, class]  = LOAD_USPS(size, shuffle)
%
%   INPUT
%   size_tr:       size of training set (default: all);
%   size_te:       size of test set (default: all);
%   shuffle:       randomization (default: false).
%
%   OUTPUT
%   X:             training images;
%   Y:             training labels;
%   Xt:            test images;
%   Yt:            test labels.
%
%Author: HENRI DE PLAEN
%Copyright: KULeuven
%Date: Jan 2020

%% PRELIMINARIES
assert(nargin<=2,'Too many input arguments') ;
assert(nargout<=4,'Too many output arguments') ;

% load
load('data_usps.mat') ;
tr_size = size(X,3) ;
te_size = size(Xt,3) ;

% arguments

if nargin < 1
    wanted_tr_size = tr_size ;
else
    wanted_tr_size = varargin{1} ;
end

if nargin < 2
    wanted_te_size = te_size ;
else
    wanted_te_size = varargin{2} ;
end

if nargin < 3
    shuffle = false ;
else
    shuffle = varargin{3} ;
end

%% LOAD MNIST 2
if shuffle
    shuffle_idx_tr = randperm(tr_size,wanted_tr_size) ;
    shuffle_idx_te = randperm(te_size,wanted_te_size) ;
else
    shuffle_idx_tr = 1:wanted_tr_size ;
    shuffle_idx_te = 1:wanted_te_size ;
end

X = X(:,:,shuffle_idx_tr) ;
Y = Y(shuffle_idx_tr) ;

Xt = Xt(:,:,shuffle_idx_te) ;
Yt = Yt(shuffle_idx_te) ;

%% OUTPUT
varargout{1} = X ;
varargout{2} = Y ;
varargout{3} = Xt ;
varargout{4} = Yt ;

end

