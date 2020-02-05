function [predicted_labels,nn_index,accuracy] = KNN_(k,labels,t_labels,Dt)
%KNN_: classifying using k-nearest neighbors algorithm. The nearest neighbors
%search method is euclidean distance
%Usage:
%       [predicted_labels,nn_index,accuracy] = KNN_(3,training,training_labels,testing,testing_labels)
%       predicted_labels = KNN_(3,training,training_labels,testing)
%Input:
%       - k: number of nearest neighbors
%       - data: (NxD) training data; N is the number of samples and D is the
%       dimensionality of each data point
%       - labels: training labels 
%       - t_data: (MxD) testing data; M is the number of data points and D
%       is the dimensionality of each data point
%       - t_labels: testing labels (default = [])
%Output:
%       - predicted_labels: the predicted labels based on the k-NN
%       algorithm
%       - nn_index: the index of the nearest training data point for each training sample (Mx1).
%       - accuracy: if the testing labels are supported, the accuracy of
%       the classification is returned, otherwise it will be zero.
%Author: Mahmoud Afifi - York University 
%checks
assert(nargin == 4, 'Wrong number of input arguments') ;

n_tr = length(labels) ;
n_test = length(t_labels) ;

assert(n_test==size(Dt,2),'Dimensions not consistent') ;
assert(n_tr==size(Dt,1),'Dimensions not consistent') ;

% if mod(k,2)==0
%     error('to reduce the chance of ties, please choose odd k');
% end
%initialization
predicted_labels=zeros(n_test,1);
ind=zeros(n_tr,n_test); %corresponding indices (MxN)
k_nn=zeros(k,n_test); %k-nearest neighbors for testing sample (Mxk)
%calc euclidean distances between each testing data point and the training
%data samples
for test_point=1:n_test
    [Dt(:,test_point),ind(:,test_point)] = sort(Dt(:,test_point));
end
%find the nearest k for each data point of the testing data
k_nn=ind(1:k,:);
nn_index=k_nn(1,:);
%get the majority vote 
for i=1:size(k_nn,2)
    options=unique(labels(k_nn(:,i)'));
    max_count=0;
    max_label=0;
    for j=1:length(options)
        L=length(find(labels(k_nn(:,i)')==options(j)));
        if L>max_count
            max_label=options(j);
            max_count=L;
        end
    end
    predicted_labels(i)=max_label;
end
%calculate the classification accuracy
if isempty(t_labels)==0
    accuracy=sum(predicted_labels==t_labels)/n_test;
end

end