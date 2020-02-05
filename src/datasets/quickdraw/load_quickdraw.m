function varargout = load_quickdraw(varargin)
%Loads the MNIST digits dataset (LeCun et al.)
%
%   [trX, trY, teX, teY]  = LOAD_QUICKDRAW(size_tr, size_te, shuffle)
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
shuffle = 1 ;
if nargin==3
    shuffle = varargin{3} ;
end


%% LOAD MNIST
size_tr = round(size_tr/8) ;
size_te = round(size_te/8) ;
size_tot = size_tr + size_te ;
[X,Y] = load_quickdraw_local(size_tot) ;

size_tot = 8*(size_tr + size_te) ;
idx_tot = randperm(size_tot) ;
idx_tr = idx_tot(1:8*size_tr) ;
idx_te = idx_tot(8*size_tr+1:end) ;

if ~shuffle
    idx_tr = sort(idx_tr) ;
    idx_te = sort(idx_te) ;
end

%output
trX = X(:,:,idx_tr) ;
teX = X(:,:,idx_te) ;
trY = Y(idx_tr) ;
teY = Y(idx_te) ;

%% OUTPUT
varargout{1} = trX ;
varargout{2} = trY ;
varargout{3} = teX ;
varargout{4} = teY ;

end

function [X,Y] = load_quickdraw_local(size_tr)

%% LOAD MNIST
quickdraw = load('quickdraw.mat') ;
airplane = quickdraw.airplane ;
crayon = quickdraw.crayon ;
calculator = quickdraw.calculator ;
clock = quickdraw.clock ;
moustache = quickdraw.moustache ;
pants = quickdraw.pants ;
rainbow = quickdraw.rainbow ;
triangle = quickdraw.triangle ;

d = sqrt(size(airplane,2)) ;

%select
n1 = size(airplane,1) ;
n2 = size(crayon,1) ;
n3 = size(calculator,1) ;
n4 = size(clock,1) ;
n5 = size(moustache,1) ;
n6 = size(pants,1) ;
n7 = size(rainbow,1) ;
n8 = size(triangle,1) ;

idx1 = randperm(n1,size_tr) ;
idx2 = randperm(n2,size_tr) ;
idx3 = randperm(n3,size_tr) ;
idx4 = randperm(n4,size_tr) ;
idx5 = randperm(n5,size_tr) ;
idx6 = randperm(n6,size_tr) ;
idx7 = randperm(n7,size_tr) ;
idx8 = randperm(n8,size_tr) ;

airplane = airplane(idx1,:) ;
crayon = crayon(idx2,:) ;
calculator = calculator(idx3,:) ;
clock = clock(idx4,:) ;
moustache = moustache(idx5,:) ;
pants = pants(idx6,:) ;
rainbow = rainbow(idx7,:) ;
triangle = triangle(idx8,:) ;

%% RESHAPE
%input
trX = zeros(d,d,8*size_tr) ;

for idx = 1:size_tr
    trX(:,:,idx) = vec2mat(airplane(idx,:),d) ;
    trX(:,:,idx+size_tr) = vec2mat(crayon(idx,:),d) ;
    trX(:,:,idx+2*size_tr) = vec2mat(calculator(idx,:),d) ;
    trX(:,:,idx+3*size_tr) = vec2mat(clock(idx,:),d) ;
    trX(:,:,idx+4*size_tr) = vec2mat(moustache(idx,:),d) ;
    trX(:,:,idx+5*size_tr) = vec2mat(pants(idx,:),d) ;
    trX(:,:,idx+6*size_tr) = vec2mat(rainbow(idx,:),d) ;
    trX(:,:,idx+7*size_tr) = vec2mat(triangle(idx,:),d) ;
end

%output
trY = zeros(8*size_tr,1) ;
trY(1:size_tr) = 1 ;
trY(1*size_tr+1:2*size_tr) = 2 ;
trY(2*size_tr+1:3*size_tr) = 3 ;
trY(3*size_tr+1:4*size_tr) = 4 ;
trY(4*size_tr+1:5*size_tr) = 5 ;
trY(5*size_tr+1:6*size_tr) = 6 ;
trY(6*size_tr+1:7*size_tr) = 7 ;
trY(7*size_tr+1:end) = 8 ;

%% OUTPUT
X = trX ;
Y = trY ;

end
