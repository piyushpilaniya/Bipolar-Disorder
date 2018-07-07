c = 0;
final = ones(60,231);
Mat = importdata('/home/intern/internship_avec/piyush/labels_metadata.csv');
xx = ones(22000,1);
        for k  = 1:22000
            xx(k) = k;
        end
for c = 1:60
    disp(c)
y1 = 'feature';
d1 = '/home/intern/internship_avec/piyush/output/';
y1 = strcat(d1,y1);
p = int2str(c);
p = strcat(p,'.mat');
y = strcat(y1,p);
load(y);
x = regexp(Mat(c+1),',','split');
y = x{1}{6};
final(c,1) = y;
final(c,1) = final(c,1) - 48;
%Feature1
matrix1 = ones(22000,23);
matrix1 = matrix(:,[2:24]);
for j = 1:23
    final(c,j+1) = mean(matrix1(:,j));
end
%feature 2
for j = 24:46
   final(c,j+1) = std(matrix1(:,j-23));
end
%feature 3 and 4
for j =47:69
        yy = matrix1(:,j-46);
        pp = polyfit(xx,yy,2);
        final(c,j+1) = pp(1);
end
for j = 70:92
	    yy = matrix1(:,j-69);
        pp = polyfit(xx,yy,1);
        final(c,j+1) = pp(1);
        final(c,j+23+1) = pp(2);
        [mini,ind] = min(matrix1(:,j-69));
        final(c,j+46+1) = mini;
        final(c,j+69+1) = ind;
        [maxim,ind] = max(matrix1(:,j-69));
        final(c,j+95+1) = maxim;
        final(c,j+1+115) = ind; 
        final(c,j+1+138) = maxim - mini;
end
end


c1 = 0;
final1 = ones(104,231);
Mat = importdata('/home/intern/internship_avec/piyush/labels_metadata.csv');
for c1 = 1:104
    disp(c1)
y1 = 'feature';
d1 = '/home/intern/internship_avec/piyush/output/';
y1 = strcat(d1,y1);
cc = c1 + 114;
p = int2str(cc);
p = strcat(p,'.mat');
y = strcat(y1,p);
load(y);
x = regexp(Mat(cc-54+1),',','split');
y = x{1}{6};
final1(c1,1) = y;
final1(c1,1) = final1(c1,1) - 48;
%Feature1
matrix2 = ones(22000,23);
matrix2 = matrix(:,[2:24]);
for j = 1:23
    final1(c1,j+1) = mean(matrix2(:,j));
end
%feature 2
for j = 24:46
   final1(c1,j+1) = std(matrix2(:,j-23));
end
%feature 3 and 4
for j =47:69
        yy = matrix2(:,j-46);
        pp = polyfit(xx,yy,2);
        final1(c1,j+1) = pp(1);
end
for j = 70:92
	    yy = matrix2(:,j-69);
        pp = polyfit(xx,yy,1);
        final1(c1,j+1) = pp(1);
        final1(c1,j+23+1) = pp(2);
        [mini,ind] = min(matrix2(:,j-69));
        final1(c1,j+46+1) = mini;
        final1(c1,j+69+1) = ind;
        [maxim,ind] = max(matrix2(:,j-69));
        final1(c1,j+92+1) = maxim;
        final1(c1,j+1+115) = ind; 
        final1(c1,j+1+138) = maxim - mini;
end
end
save('/home/intern/internship_avec/piyush/output_audio')
ELM(final1, final, 1, 2000, 'sig');
