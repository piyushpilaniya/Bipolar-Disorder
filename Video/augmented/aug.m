load('/home/intern/internship_avec/piyush/Dev/pls/mfcc/Train.mat')

aug_mat = ones(638,391);
current = 1;
count1 = 0;
count2 = 0;
count3 = 0;
for i = 1:104
   if(final1(i,1) <= 7)
     for j = 1:11
       aug_mat(current,:) = final1(i,:);
       current = current + 1;
     end
     count1 = count1 + 1;
   end
   if(final1(i,1)<20 && final1(i,1) > 7)
        for j = 1:2
           aug_mat(current,:) = final1(i,:);
       current = current + 1;
        end
        count2 = count2 + 1;
   end
      if(final1(i,1) >=20)
          for j = 1:7
              aug_mat(current,:) = final1(i,:);
       current = current + 1;
          end
          count3 = count3 + 1;
       end
end
disp(count1)
disp(count2)
disp(count3)

