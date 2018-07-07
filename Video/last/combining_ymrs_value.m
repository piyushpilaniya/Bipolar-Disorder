load('/home/intern/internship_avec/Dev_geometric/window_3d/output_dev_ymrs_final.mat')
bb = ones(104*5,1);
%mm = ones(60*2,1);
for i = 1:104
bb((i-1)*5+1,1) = output(i,1)
bb((i-1)*5+2,1) = output(i,1)
bb((i-1)*5+3,1) = output(i,1)
bb((i-1)*5+4,1) = output(i,1)
bb((i-1)*5+5,1) = output(i,1)
end
%load('/home/intern/internship_avec/Dev_geometric/window_3d/output_train_ymrs_final.mat')

%for i = 1:60
%mm((i-1)*2+1,1) = outVec(i,1)
%mm((i-1)*2+2,1) = outVec(i,1)
%mm((i-1)*3+3,1) = outVec(i,1)
%end
%kk = [bb;mm]

