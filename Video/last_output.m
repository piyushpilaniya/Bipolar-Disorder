
c = 0;
final = ones(300,231);
xx = ones(1200,1);
        for k  = 1:1200
            xx(k) = k;
        end
Mat = importdata('/home/intern/internship_avec/prinshu/labels_metadata.csv');
for c = 1:60
    if (c == 7)
        continue;
    end
y1 = 'feature';
d1 = '/home/intern/internship_avec/prinshu/output_geometric_feature/';
y1 = strcat(d1,y1);
p = int2str(c);
p = strcat(p,'.mat');
y = strcat(y1,p);
load(y);
x = regexp(Mat(c+1),',','split');
y = x{1}{6};
Res = ones(10,24);
final(c,1) = y;
final(c,1) = final(c,1) - 48;
for jj = 1:5
    finalNew = result([(jj-1)*700+1 : jj*1200-(jj-1)*500],:);
    final((c-1)*5 + jj,1) = y - 48;

%Feature1
for j = 1:23
    final((c-1)*5 + jj,j+1) = mean(finalNew(:,j));
end
%feature 2
for j = 24:46
   final((c-1)*5 + jj,j+1) = std(finalNew(:,j-23));
end
%feature 3 and 4
for j =47:69
        yy = finalNew(:,j-46);
        pp = polyfit(xx,yy,2);
        final((c-1)*5 + jj,j+1) = pp(1);
end
for j = 70:92
        yy = finalNew(:,j-69);
        pp = polyfit(xx,yy,1);
        final((c-1)*5 + jj,j+1) = pp(1);
        final((c-1)*5 + jj,j+23+1) = pp(2);
        [mini,ind] = min(finalNew(:,j-69));
        final((c-1)*5 + jj,j+46+1) = mini;
        final((c-1)*5 + jj,j+69+1) = ind;
        [maxim,ind] = max(finalNew(:,j-69));
        final((c-1)*5 + jj,j+92+1) = maxim;
        final((c-1)*5 + jj,j+1+115) = ind; 
        final((c-1)*5 + jj,j+1+138) = maxim - mini;
end
end
end

c1 = 0;
final1 = ones(520,231);
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
y = x{1}{6};
final1(c1,1) = y;
final1(c1,1) = final1(c1,1) - 48;

for jj = 1:5
    finalNew = result([(jj-1)*700+1 : jj*1200-(jj-1)*500],:);
    final1((c1-1)*5 + jj,1) = y - 48;

%Feature1
for j = 1:23
    final1((c1-1)*5 + jj,j+1) = mean(finalNew(:,j));
end
%feature 2
for j = 24:46
   final1((c1-1)*5 + jj,j+1) = std(finalNew(:,j-23));
end
%feature 3 and 4
for j =47:69
        
        yy = finalNew(:,j-46);
        pp = polyfit(xx,yy,2);
        final1((c1-1)*5 + jj,j+1) = pp(1);
end
for j = 70:92
        yy = finalNew(:,j-69);
        pp = polyfit(xx,yy,1);
        final1((c1-1)*5 + jj,j+1) = pp(1);
        final1((c1-1)*5 + jj,j+23+1) = pp(2);
        [mini,ind] = min(finalNew(:,j-69));
        final1((c1-1)*5 + jj,j+46+1) = mini;
        final1((c1-1)*5 + jj,j+69+1) = ind;
        [maxim,ind] = max(finalNew(:,j-69));
        final1((c1-1)*5 + jj,j+92+1) = maxim;
        final1((c1-1)*5 + jj,j+1+115) = ind; 
        final1((c1-1)*5 + jj,j+1+138) = maxim - mini;
end
end
end
%final = 1000*final;
%final1 = 1000*final1;
disp(final1)
save('/home/intern/internship_avec/prinshu/output_feature_threeWindow')
ELM(final1, final, 1, 20, 'sig');
