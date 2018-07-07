list2=dir('/home/intern/Downloads/AVEC2018-master/baseline_features/LLDs_audio_opensmile_MFCCs_turns/');
list = [];
c = 0;
for j = 1:length(list2)
x = regexp(list2(j).name,'.csv','match');

if length(x) > 0
        c = c+1;
s1 = '/home/intern/Downloads/AVEC2018-master/baseline_features/LLDs_audio_opensmile_MFCCs_turns/';
s2 = list2(j).name;
s=strcat(s1,s2);
M = importdata(s);
[row,col] = size(M.data);
% s3 = '/home/intern/Downloads/AVEC2018-master/baseline_features/features_audio_eGeMAPS_turns/';
% s4 = list2(j).name;
% s5=strcat(s3,s4);
% M1 = importdata(s5);
% [row1,col1] = size(M1.data);
matrix = zeros(16000,col);
if(row > 16000)
    factor = (row-11)/16000;
       value = fix(factor);
       offset = (row-11) - value*16000;
       rows = 1;
       finds = 1;
       current = 1;
   while (rows ~= 16001)
       if(current <= offset)
           matrix(rows,(1:col)) = M.data(finds,:);
           matrix(rows,(col+1:col+col1)) = M1.data(finds,:);
           current = current +1;
           finds = finds + value + 1;
       else
           matrix(rows,(1:col)) = M.data(finds,:);
           matrix(rows,(col+1:col+col1)) = M1.data(finds,:);
           finds = finds + value ;

       end
              rows = rows+1;
   end
end

if(row < 16000)
    factor = 16000/(row-10);
       value = fix(factor);
       offset = 16000 - value*(row-10);
       rows = 1;
       finds = 1;
       current = 1;
       
   while (rows~= 16001)
       x = 1;
       if(current <= offset)
         current = current +1;
           while (x <= value+2 && rows~= 16001)
              matrix(rows,(1:col)) = M.data(finds,:);
              matrix(rows,(1+col:col+col1)) = M1.data(finds,:);
              rows = rows + 1;
              x = x+1;
           end
       else
           while (x <= value+1 && rows~=16001)
              matrix(rows,(1:col)) = M.data(finds,:);
              matrix(rows,(1+col:col+col1)) = M1.data(finds,:);
              rows = rows + 1;
              x = x+1;
           end
       end
       finds = finds + 1;
   end
end

y1 = 'feature';
d1 = '/home/intern/internship_avec/piyush/output_combined_new/';
y1 = strcat(d1,y1);
p = int2str(c);
p = strcat(p,'.mat');
y = strcat(y1,p);
save(y);
end
end
