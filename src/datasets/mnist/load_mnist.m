function varargout = load_mnist(varargin)
%Loads the MNIST digits dataset (LeCun et al.)
%
%   [trX, trY, teX, teY]  = LOAD_MNIST(size_tr, size_te, shuffle)
%
%   INPUT
%   size_tr:       size of training set
%   size_te:       size of test set
%   shuffle:       randomization (default 0) NOT IMPLEMENTED YET
%
%   OUTPUT
%   trX:           training images
%   trY:           training labels
%   teX:           test images
%   teY:           test labels
%
%Author: HENRI DE PLAEN
%Copyright: KULeuven
%Date: March 2019

%% PRELIMINARIES
% parameters
assert(nargout==4, 'Wrong number of output parameters') ;
assert(nargin<4, 'Wrong number of input parameters') ;

size_tr = varargin{1} ;
size_te = varargin{2} ;
shuffle = 0 ;
if nargin==3
    shuffle = varargin{3} ;
end

% paths
folder_path = 'mnist/' ;
trX_path = 'mnist-tr-img' ;
trY_path = 'mnist-tr-labels' ;
teX_path = 'mnist-te-img' ;
teY_path = 'mnist-te-labels' ;

trX_path = [folder_path trX_path] ;
trY_path = [folder_path trY_path] ;
teX_path = [folder_path teX_path] ;
teY_path = [folder_path teY_path] ;

% %% LOAD MNIST 1
% offset = 0 ;
% [trX, trY] = readMNIST(trX_path, trY_path, size_tr, offset) ;
% [teX, teY] = readMNIST(teX_path, teY_path, size_te, offset) ;

%% LOAD MNIST 2
if shuffle
    tr_idx = randperm(6e+4,size_tr) ;
    te_idx = randperm(1e+4,size_te) ;
else
    tr_idx = 1:size_tr ;
    te_idx = 1:size_te ;
end

disp([newline 'TRAINING']) ;
trX = processMNISTimages(trX_path) ;
trY = processMNISTlabels(trY_path) ;

disp([newline 'TEST']) ;
teX = processMNISTimages(teX_path) ;
teY = processMNISTlabels(teY_path) ;

trX = squeeze(trX(:,:,:,tr_idx)) ;
teX = squeeze(teX(:,:,:,te_idx)) ;

trY = squeeze(trY(tr_idx,:)) ;
teY = squeeze(teY(te_idx,:)) ;

%% EVENTUAL SHUFFLE
% not implemented

%% OUTPUT
varargout{1} = trX ;
varargout{2} = trY ;
varargout{3} = teX ;
varargout{4} = teY ;

end

