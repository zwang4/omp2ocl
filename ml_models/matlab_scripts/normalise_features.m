function [train_feature,test_feature] = normalise_features(train_feature,test_feature)
ts=length(train_feature);
data=[train_feature;test_feature];
data=scale_data(data);
train_feature=data(1:ts,1:end);
t=ts+1;
lt=length(data);
test_feature=data(t:lt,1:end);
end

function [data]=scale_data(data)
%data=(data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
data=data;
end