function varargout = group_data(varargin)
%Groups the different labels together.
%
%[X_out, labels_out] = GROUP_MNIST(X_in, labels_in)
%
%INPUT
%   X_in:           input of images
%   labels_in:      corresponding labels
%
%OUTPUT
%   X_out:          image output
%   labels_out:     corresponding labels
%
%Author: HENRI DE PLAEN
%Date: March 2019
%Copyright: KU Leuven


%% PRELIMINARIES
assert(nargin==2, 'Wrong number of input arguments') ;
assert(nargout==2, 'Wrong number of output arguments') ;

X_in = varargin{1} ;
labels_in = varargin{2} ;
labels_in = labels_in(:) ;

%% GROUPING
ulabs = unique(labels_in) ;
nl = length(ulabs) ;

X_temp = cell(nl,1) ;       % prealloc
labs_temp = cell(nl,1) ;    % prealloc
for idx_l = 1:nl
    loc_lab = ulabs(idx_l) ;
    loc_idx = labels_in==loc_lab ;
    X_temp{idx_l} = X_in(:,:,loc_idx) ;
    labs_temp{idx_l} = labels_in(loc_idx) ;
end

X_out = cat(3,X_temp{:}) ;
labels_out = cat(1,labs_temp{:}) ;

%% OUTPUT
varargout{1} = X_out ;
varargout{2} = labels_out ;

end

