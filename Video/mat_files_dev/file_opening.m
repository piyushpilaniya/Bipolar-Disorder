list2=dir('/home/intern/internship_avec/Dev_geometric/baseline_features/LLDs_video_openFace_AUs/');
list = [];
c = 0;
for j = 1:length(list2)
x = regexp(list2(j).name,'.csv','match');

if length(x) > 0
        c = c+1;
s1 = '/home/intern/internship_avec/Dev_geometric/baseline_features/LLDs_video_openFace_AUs/';
s2 = list2(j).name;
s=strcat(s1,s2);
M = importdata(s);
[row,col] = size(M.data);
matrix = zeros(4000,col);
if(row > 4000)
    factor = (row-1)/4000;
       value = fix(factor);
       offset = (row-1) - value*4000;
       rows = 1;
       finds = 1;
       current = 1;
   while (rows ~= 4001)
       if(current <= offset)
           matrix(rows,:) = M.data(finds,:);
           current = current +1;
           finds = finds + value + 1;
       else
           matrix(rows,:) = M.data(finds,:);
           finds = finds + value ;

       end
              rows = rows+1;
   end
end

if(row < 4000)
    factor = 4000/row;
       value = fix(factor);
       offset = 4000 - value*row;
       rows = 1;
       finds = 1;
       current = 1;
       
   while (rows~= 4001)
       x = 1;
       if(current <= offset)
         current = current +1;
           while (x <= value+2 && rows~= 4001)
              matrix(rows,:) = M.data(finds,:);
              rows = rows + 1;
              x = x+1;
           end
       else
           while (x <= value+1 && rows~=4001)
              matrix(rows,:) = M.data(finds,:);
              rows = rows + 1;
              x = x+1;
           end
       end
       finds = finds + 1;
   end
end

y1 = 'feature';
d1 = '/home/intern/internship_avec/Dev_geometric/mat_files_dev/output/';
y1 = strcat(d1,y1);
p = int2str(c);
p = strcat(p,'.mat');
y = strcat(y1,p);
result = zeros(4000,23);
for i = 1:4000
    b = matrix(i,[317:365]);
    b = transpose(b);
    a = zeros(49,2);
    a(:,1) = b;
    b = matrix(i,[385:433]);
    b = transpose(b);
    a(:,2) = b;
    res = lms_to_geo(a);
    res = transpose(res);
    result(i,:) = res;
end
disp(result)
save(y);
end
end
