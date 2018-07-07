final = ones(54,231);
Mat = importdata('/home/intern/internship_avec/prinshu/labels_metadata.csv');
t = 0;
for t = 1:54
y1 = 'feature';
d1 = '/home/intern/internship_avec/prinshu/output_geometric_feature/';
y1 = strcat(d1,y1);
cc1 = t + 60;

p = int2str(cc1);
p = strcat(p,'.mat');
y = strcat(y1,p);
load(y);
x = regexp(Mat(cc1+1),',','split');
yyy = x{1}{5};
Res = ones(10,24);
yyy = str2num(['uint8(',yyy,')']);
 final(t,1) = yyy;
 %final(t,1) = final(t,1);
%Feature1
for j = 1:23
    final(t,j+1) = mean(result(:,j));
end

%feature 2
for j = 24:46
   final(t,j+1) = std(result(:,j-23));
end
%feature 3 and 4
for j =47:69
        xx = ones(4000,1);
        for k  = 1:4000
            xx(k) = k;
        end
        yy = result(:,j-46);
        pp = polyfit(xx,yy,2);
        final(t,j+1) = pp(1);
end
for j = 70:92
        yy = result(:,j-69);
        pp = polyfit(xx,yy,1);
        final(t,j+1) = pp(1);
        final(t,j+23+1) = pp(2);
        [mini,ind] = min(result(:,j-69));
        final(t,j+46+1) = mini;
        final(t,j+69+1) = ind;
        [maxim,ind] = max(result(:,j-69));
        final(t,j+92+1) = maxim;
        final(t,j+115+1) = ind; 
        final(t,j+138+1) = maxim - mini;
end
end


c1 = 0;
final1 = ones(104,231);
Mat = importdata('/home/intern/internship_avec/prinshu/labels_metadata.csv');
for c1 = 1:104
y1 = 'feature';
d1 = '/home/intern/internship_avec/prinshu/output_geometric_feature/';
y1 = strcat(d1,y1);
cc = c1+114;
p = int2str(cc);
p = strcat(p,'.mat');
y = strcat(y1,p);
load(y);
x = regexp(Mat(cc-54+1),',','split');
yyy = x{1}{5};
Res = ones(10,24);
yyy = str2num(['uint8(',yyy,')']);

 final1(c1,1) = yyy;
%Feature1
for j = 1:23
    final1(c1,j+1) = mean(result(:,j));
end
%feature 2
for j = 24:46
   final1(c1,j+1) = std(result(:,j-23));
end
%feature 3 and 4
for j =47:69
        xx = ones(4000,1);
        for k  = 1:4000
            xx(k) = k;
        end
        yy = result(:,j-46);
        pp = polyfit(xx,yy,2);
        final1(c1,j+1) = pp(1);
end
for j = 70:92
        yy = result(:,j-69);
        pp = polyfit(xx,yy,1);
        final1(c1,j+1) = pp(1);
        final1(c1,j+23+1) = pp(2);
        [mini,ind] = min(result(:,j-69));
        final1(c1,j+46+1) = mini;
        final1(c1,j+69+1) = ind;
        [maxim,ind] = max(result(:,j-69));
        final1(c1,j+92+1) = maxim;
        final1(c1,j+1+115) = ind; 
        final1(c1,j+1+138) = maxim - mini;
end
end
%final = 1000*final;
%final1 = 1000*final1;
disp(final1)

save('/home/intern/internship_avec/prinshu/output_feature_test')
ELM(final1, final, 1, 20, 'sig');
