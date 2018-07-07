load('/home/intern/internship_avec/Dev_geometric/random_forest/3_window_of_a_video/(train+dev)->test/Dev.mat')

c1 = 0;
xx = ones(2400,1);
        for k  = 1:2400
            xx(k) = k;
        end
final1 = ones(54*2,231);
Mat = importdata('/home/intern/internship_avec/prinshu/labels_metadata.csv');
for c1 = 1:54
y1 = 'feature';
d1 = '/home/intern/internship_avec/prinshu/output_geometric_feature/';
y1 = strcat(d1,y1);
cc = c1+60;
p = int2str(cc);
p = strcat(p,'.mat');
y = strcat(y1,p);
load(y);
x = regexp(Mat(cc-54+1),',','split');
y = x{1}{6};
final1(c1,1) = y;
final1(c1,1) = final1(c1,1) - 48;

for jj = 1:2
    finalNew = result([(jj-1)*1600+1 : jj*2400-(jj-1)*800],:);
    final1((c1-1)*2 + jj,1) = y - 48;

%Feature1
for j = 1:23
    final1((c1-1)*3 + jj,j+1) = mean(finalNew(:,j));
end
%feature 2
for j = 24:46
   final1((c1-1)*3 + jj,j+1) = std(finalNew(:,j-23));
end
%feature 3 and 4
for j =47:69
        
        yy = finalNew(:,j-46);
        pp = polyfit(xx,yy,2);
        final1((c1-1)*3 + jj,j+1) = pp(1);
end
for j = 70:92
        yy = finalNew(:,j-69);
        pp = polyfit(xx,yy,1);
        final1((c1-1)*3 + jj,j+1) = pp(1);
        final1((c1-1)*3 + jj,j+23+1) = pp(2);
        [mini,ind] = min(finalNew(:,j-69));
        final1((c1-1)*3 + jj,j+46+1) = mini;
        final1((c1-1)*3 + jj,j+69+1) = ind;
        [maxim,ind] = max(finalNew(:,j-69));
        final1((c1-1)*3 + jj,j+92+1) = maxim;
        final1((c1-1)*3 + jj,j+1+115) = ind; 
        final1((c1-1)*3 + jj,j+1+138) = maxim - mini;
end
end
end