function [avg,preds,pca,trees]=train_predict(apps,path)
l=length(apps);
trees=cell(l);
pca=zeros(l,1);
preds=cell(l,5);
err=0;
mn=0;
for i=1:length(apps)
    app = apps{i};
    [train_feature,train_dec,test_feature,test_dec]=load_features(app, path);
    [train_feature,test_feature] = normalise_features(train_feature,test_feature);
    %t=classregtree([train_feature;test_feature],[train_dec;test_dec]);
    t=classregtree(train_feature,train_dec);
    trees{i}=t;
    pred=t.eval(test_feature);
    for j=1:length(pred)
        preds(i,j)=pred(j);
    end
    pct=mean(strcmp(pred,test_dec));
    err = err + sum(strcmp(pred,test_dec));
    mn= mn +length(test_feature);
    pca(i)=pct;
end

avg=err/mn;

end