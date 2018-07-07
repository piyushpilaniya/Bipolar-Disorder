load('/home/intern/internship_avec/prinshu/output_Windows_feature.mat')
train = ones(104,799,230);
output = ones(104,1);
for i = 1:104
   for j = 1:799
      train(i,j,:) = final1(i,(j-1)*230 + 1);
   end
   output(i) = final1(i,1);
end

dev = ones(60,799,230);
outVec = ones(60,1);
for i = 1:60
   for j = 1:799
      dev(i,j,:) = final(i,(j-1)*230 + 1);
   end
   outVec(i) = final(i,1);
end

save('/home/intern/internship_avec/Dev_geometric/window_3d/3d_output_window')
