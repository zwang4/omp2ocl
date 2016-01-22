function [train_features,train_dec,test_features,test_dec] = load_features(app, path)
train_feature_f = [path,'\',app,'_train.csv'];
train_dec_f = [path,'\',app,'_train.dec.csv'];
test_feature_f = [path,'\',app,'_test.csv'];
test_dec_f = [path,'\',app,'_test.dec.csv'];

train_features=load_f(train_feature_f);
test_features = load_f(test_feature_f);
train_dec = load_dec(train_dec_f);
test_dec = load_dec(test_dec_f);

end

function [features] = load_f(file)
features=dlmread(file,',');
end

function [dec] = load_dec(file)
fid = fopen(file, 'r');
if (fid < 0)
    disp('open file error\n');
end

dec=cell(1);
i=1;
while ~feof(fid)
    d = fscanf(fid, '%s', 1);
    if ~isempty(d)
        dec{i}=d;
    end
    i = i + 1;
end

dec=dec';

fclose(fid);
end