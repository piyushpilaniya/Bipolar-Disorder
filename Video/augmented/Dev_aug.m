load('/home/intern/internship_avec/data/devdata_ymrs(60_231)/Dev.mat')
aug_mat = ones(294,231);
current = 1;
count1 = 0;
count2 = 0;
count3 = 0;
for i = 1:60
   if(final3(i,1) <= 7)
     for j = 1:7
       aug_mat(current,:) = final3(i,:);
       current = current + 1;
     end
     count1 = count1 + 1;
   end
   if(final3(i,1)<20 && final3(i,1) > 7)
        for j = 1:2
           aug_mat(current,:) = final3(i,:);
       current = current + 1;
        end
        count2 = count2 + 1;
   end
      if(final3(i,1) >=20)
          for j = 1:6
              aug_mat(current,:) = final3(i,:);
       current = current + 1;
          end
          count3 = count3 + 1;
       end
end
disp(count1)
disp(count2)
disp(count3)

%save('/home/intern/internship_avec/Dev_geometric/augmented/output_augmented')

