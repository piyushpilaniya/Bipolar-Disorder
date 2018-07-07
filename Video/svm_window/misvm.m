load('/home/intern/internship_avec/prinshu/output_Windows_feature.mat');
pp = final1(:,1);
pp = transpose(pp);
train = final1(:,[2:183771]);
for i = 1:104
train(i,:) = train(i,:)/mean(train(i,:));
end

M = svm.train(train,pp,'kernel_function','quadratic');
test = final(:,[2:183771]);
for i = 1:60
test(i,:) = test(i,:)/mean(test(i,:));
end
outVec = final(:,1);
outVec = transpose(outVec);
N = svm.predict(M,test);
Accuracy=mean(N==outVec)*100;
disp(Accuracy)