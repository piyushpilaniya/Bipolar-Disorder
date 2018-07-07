function g = lms_to_geo(lms)
% -----------------------------------------------------------------------
% Copyright (2015): Furkan Gurpinar
% E-mail: furkan.gurpinar@boun.edu.tr
%
% This software is distributed under the terms
% of the GNU General Public License Version 3
% 
% Permission to use, copy, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose.
%
% If you use this code, please refer to our ICMI paper:
% Heysem Kaya, Furkan Gurpinar, Sadaf Afshar, Albert Ali Salah, "Contrasting and Combining Least Squares Based Learners for Emotion Recognition in the Wild", in Proc. ACM ICMI 2015
% ----------------------------------------------------------------------
% Input: lms [49,2] landmarks
% Output g: geometric feature vector
% Feature 1 : eye height/eye width = eye aspect ratio
% Feature 2 : Mouth aspect ratio

g = zeros(2,1);



% if length(lms)<49
%     error('No face supplied.');
%     %return
% end

% ------------------------------------------------
% Feature 1 : Eye Aspect Ratio : 
height_eye_left  = (norm(lms(21,:)-lms(24,:)) + norm(lms(22,:)-lms(25,:))) / 2;
height_eye_right = (norm(lms(27,:)-lms(30,:)) + norm(lms(28,:)-lms(31,:))) / 2;
height_eye = mean([height_eye_left , height_eye_right]);

width_eye_left  = norm(lms(20,:) - lms(23,:));
width_eye_right = norm(lms(26,:) - lms(29,:));
width_eye = mean([width_eye_left , width_eye_right]);

g(1) = width_eye / height_eye;
% ------------------------------------------------
% Feature 2 : Mouth Aspect Ratio :
width_mouth  = norm(lms(32,:) - lms(38,:));
height_mouth = norm(lms(35,:) - lms(41,:));

g(2) = width_mouth / height_mouth;
% ------------------------------------------------
% Feature 3 --NOT and 4-- : Lip Angles :
left_mouth = lms(32,:);
upper_mouth = lms(35,:);
right_mouth = lms(38,:);
left_lip_vector = left_mouth - upper_mouth;
left_lip_vector = left_lip_vector / norm(left_lip_vector);

right_lip_vector = right_mouth - upper_mouth;
right_lip_vector = right_lip_vector / norm(right_lip_vector);

left_lip_angle  = rad2deg(acos(dot([-1 0] , left_lip_vector)));
right_lip_angle = rad2deg(acos(dot([1 0] , right_lip_vector)));

%g(3) = left_lip_angle;
%g(4) = right_lip_angle;
g(3) = (left_lip_angle + right_lip_angle) / 2;

% ------------------------------------------------
% Feature 4 : Nose angles : 
nose_tip = lms(17,:);


left_nose_vector = left_mouth - nose_tip; 
left_nose_vector = left_nose_vector/norm(left_nose_vector);
left_nose_angle = rad2deg(acos(dot(left_nose_vector , [0,1])));

right_nose_vector = right_mouth - nose_tip; 
right_nose_vector = right_nose_vector/norm(right_nose_vector);
right_nose_angle = rad2deg(acos(dot(right_nose_vector , [0,1])));

g(4) = mean([left_nose_angle , right_nose_angle]);

% ------------------------------------------------
% Feature 5 : Lip angle in bottom-sides
% i.e. angles between nodes 32-42 (left) and 38-40 (right)

left_bottom_lip2 = lms(42,:);
left_lip2_vector = left_mouth - left_bottom_lip2;
left_lip2_vector = left_lip2_vector / norm(left_lip2_vector);
left_lip2_angle = rad2deg(acos(dot([-1 0] , left_lip2_vector)));

right_bottom_lip2 = lms(40,:);
right_lip2_vector = right_mouth - right_bottom_lip2;
right_lip2_vector = right_lip2_vector / norm(right_lip2_vector);
right_lip2_angle = rad2deg(acos(dot([1 0] , right_lip2_vector)));

g(5) = mean([left_lip2_angle , right_lip2_angle]);

% ------------------------------------------------
% Feature 6 : Eyebrow angles :

left_outer_eyebrow = lms(1,:);
left_inner_eyebrow = lms(5,:);
right_outer_eyebrow = lms(10,:);
right_inner_eyebrow = lms(6,:);

left_eyebrow_vector = left_outer_eyebrow - left_inner_eyebrow;
left_eyebrow_vector = left_eyebrow_vector / norm(left_eyebrow_vector);
left_eyebrow_angle = rad2deg(acos(dot([-1 0] , left_eyebrow_vector)));

right_eyebrow_vector = right_outer_eyebrow - right_inner_eyebrow;
right_eyebrow_vector = right_eyebrow_vector / norm(right_eyebrow_vector);
right_eyebrow_angle = rad2deg(acos(dot([1 0] , right_eyebrow_vector)));

g(6) = mean([left_eyebrow_angle , right_eyebrow_angle]);
% ------------------------------------------------
% Feature 7 : Average -NOT-normalized intensity in cheek region :

% % % left_nose = lms(15,:);
% % % right_nose = lms(19,:);
% % % 
% % % left_cheek_x = [round(left_mouth(:,1) - abs(left_mouth(:,1) - left_nose(:,1))) : round(left_nose(:,1))];
% % % left_cheek_y = [round(left_nose(:,2) - abs(left_nose(:,2) - left_mouth(:,2))) : round(left_mouth(:,2))];

% Visualize the cheek region : 

% % % line([left_cheek_x(1);left_cheek_x(1)] , [left_cheek_y(1);left_cheek_y(length(left_cheek_y))]); % LEFT VERTICAL
% % % line([left_cheek_x(length(left_cheek_x));left_cheek_x(length(left_cheek_x))] , [left_cheek_y(1);left_cheek_y(length(left_cheek_y))]); % RIGHT VERTICAL
% % % line([left_cheek_x(1);left_cheek_x(length(left_cheek_x))] , [left_cheek_y(1);left_cheek_y(1)]); % UP Horizontal
% % % line([left_cheek_x(1);left_cheek_x(length(left_cheek_x))] , [left_cheek_y(length(left_cheek_y));left_cheek_y(length(left_cheek_y))]); % Bottom Horizontal

% ------------------------------------------------
% g(7)=0;
% g(8)=0;
% ------------------------------------------------
% Feature 7 and 8 : Lower Eye angles (9: Outer, 10: inner) : 

% Left Eye :
left_eye_out = lms(20,:);
left_eye_out_low = lms(25,:);
left_eye_in = lms(23,:);
left_eye_in_low = lms(24,:);

lower_eye_left_out_vector = left_eye_out_low - left_eye_out;
lower_eye_left_out_vector = lower_eye_left_out_vector/norm(lower_eye_left_out_vector);
left_outer_angle = rad2deg(acos(dot([1 0] , lower_eye_left_out_vector)));

lower_eye_left_in_vector = left_eye_in - left_eye_in_low;
lower_eye_left_in_vector = lower_eye_left_in_vector / norm(lower_eye_left_in_vector);
left_inner_angle = rad2deg(acos(dot([1 0] , lower_eye_left_in_vector)));


% Right Eye :
right_eye_out = lms(29,:);
right_eye_out_low = lms(30,:);
right_eye_in = lms(26,:);
right_eye_in_low = lms(31,:);

lower_eye_right_out_vector = right_eye_out_low - right_eye_out;
lower_eye_right_out_vector = lower_eye_right_out_vector/norm(lower_eye_right_out_vector);
right_outer_angle = rad2deg(acos(dot([-1 0] , lower_eye_right_out_vector)));

lower_eye_right_in_vector = right_eye_in - right_eye_in_low;
lower_eye_right_in_vector = lower_eye_right_in_vector / norm(lower_eye_right_in_vector);
right_inner_angle = rad2deg(acos(dot([-1 0] , lower_eye_right_in_vector)));

%g(9) = mean([left_outer_angle, left_inner_angle , right_outer_angle , right_inner_angle]);
g(7) = mean([left_outer_angle, right_outer_angle]);
g(8) = mean([left_inner_angle, right_inner_angle]);


% ------------------------------------------------
% Feature 9 : Mouth corner - mouth bottom angle : 
bottom_mouth = lms(41,:);

left_mouth_vector = bottom_mouth - left_mouth;
left_mouth_vector = left_mouth_vector / norm(left_mouth_vector);
left_mouth_bottom_angle = rad2deg(acos(dot([1 0] , left_mouth_vector)));

right_mouth_vector = bottom_mouth - right_mouth;
right_mouth_vector = right_mouth_vector / norm(right_mouth_vector);
right_mouth_bottom_angle = rad2deg(acos(dot([-1 0] , right_mouth_vector)));

g(9) = mean([right_mouth_bottom_angle, left_mouth_bottom_angle]);

% ------------------------------------------------
% Feature 10 : Mouth corner - mouth up (2nds) angle : 
mouth_up_2nd_right = lms(36,:);
mouth_up_2nd_left = lms(34,:);

left_mouth_vector = mouth_up_2nd_left - left_mouth;
left_mouth_vector = left_mouth_vector / norm(left_mouth_vector);
left_mouth_bottom_angle = rad2deg(acos(dot([1 0] , left_mouth_vector)));

right_mouth_vector = mouth_up_2nd_right - right_mouth;
right_mouth_vector = right_mouth_vector / norm(right_mouth_vector);
right_mouth_bottom_angle = rad2deg(acos(dot([-1 0] , right_mouth_vector)));

g(10) = mean([right_mouth_bottom_angle, left_mouth_bottom_angle]);

% ------------------------------------------------
% Feature 11 : Menger Curvature of lower-outer lips : 
left_mouth_1st = lms(43,:);
right_mouth_1st = lms(39,:);
x = left_mouth_1st;
y = left_bottom_lip2;
z = left_mouth;
xyz = [x;y;z];
menger_curvature_left = 4*polyarea(xyz(:,1) , xyz(:,2)) / (norm(x-y)*norm(y-z)*norm(x-z));
menger_curvature_left = menger_curvature_left*1000;

x = right_mouth_1st;
y = right_bottom_lip2;
z = right_mouth;
xyz = [x;y;z];
menger_curvature_right = 4*polyarea(xyz(:,1) , xyz(:,2)) / (norm(x-y)*norm(y-z)*norm(x-z));
menger_curvature_right = menger_curvature_right*1000;

g(11) = mean([menger_curvature_left,menger_curvature_right]);



% ------------------------------------------------
% Feature 12 : Menger Curvature of lips lower- bottom2: 
x = bottom_mouth;
y = left_bottom_lip2;
z = left_mouth;
xyz = [x;y;z];
menger_curvature_left = 4*polyarea(xyz(:,1) , xyz(:,2)) / (norm(x-y)*norm(y-z)*norm(x-z));
menger_curvature_left = menger_curvature_left*1000;

x = bottom_mouth;
y = right_bottom_lip2;
z = right_mouth;
xyz = [x;y;z];
menger_curvature_right = 4*polyarea(xyz(:,1) , xyz(:,2)) / (norm(x-y)*norm(y-z)*norm(x-z));
menger_curvature_right = menger_curvature_right*1000;

g(12) = mean([menger_curvature_left,menger_curvature_right]);


% ------------------------------------------------
% Feature 13 : Menger Curvature of lips lower- bottom2: 
x = bottom_mouth;
y = right_mouth;
z = left_mouth;
xyz = [x;y;z];
menger_curvature_wholelow = 4*polyarea(xyz(:,1) , xyz(:,2)) / (norm(x-y)*norm(y-z)*norm(x-z));
menger_curvature_wholelow = menger_curvature_wholelow*1000;


g(13) = menger_curvature_wholelow;

% ------------------------------------------------
% Feature 14 : Mouth Opening

d1 = norm(lms(49,:)-lms(44,:));
d2 = norm(lms(48,:)-lms(45,:));
d3 = norm(lms(46,:)-lms(47,:));

mouthWidth = norm(lms(32,:)-lms(38,:));

g(14) = mean([d1,d2,d3])/mouthWidth; % make this Dimensionless ? ok, divided by mouthWidth


% ------------------------------------------------
% Feature 15 : Lower Mouth / Upper Mouth Ratio

d1 = norm(lms(45,:)-lms(41,:));
d2 = norm(lms(45,:)-lms(35,:));

g(15) = d1/d2;
% here is a little cheat for feature 15.
% Just 2 values are greater than 20. one is 25, the other is 162 ???
% so just threshold them.
if g(15)>25
    g(15) = 5;
end



% ------------------------------------------------
% Feature 16 : Eyebrow-Eye distance (normalized to eyebrow / lower mouth distance)

left_eyebrow_mid = (left_outer_eyebrow + left_inner_eyebrow) / 2;
right_eyebrow_mid = (right_outer_eyebrow + right_inner_eyebrow) / 2;
left_eye_mid = (left_eye_out + left_eye_in) / 2;
right_eye_mid = (right_eye_out + right_eye_in) / 2;

d1 = mean([norm(left_eyebrow_mid - left_eye_mid), norm(right_eyebrow_mid - right_eye_mid)]);
d2 = mean([norm(left_eyebrow_mid - bottom_mouth), norm(right_eyebrow_mid - bottom_mouth)]);
g(16) = d1/d2;

% ------------------------------------------------
% Feature 17 : INNER Eyebrow-Eye distance (normalized to eyebrow / lower mouth distance)


left_eye_mid = (left_eye_out + left_eye_in) / 2;
right_eye_mid = (right_eye_out + right_eye_in) / 2;

d1 = mean([norm(left_inner_eyebrow - left_eye_mid), norm(right_inner_eyebrow - right_eye_mid)]);
d2 = mean([norm(left_eyebrow_mid - bottom_mouth), norm(right_eyebrow_mid - bottom_mouth)]);
g(17) = d1/d2;


% ------------------------------------------------
% Some new features from the paper :
% % @inproceedings{saeed2012effective,
% %   title={Effective geometric features for human emotion recognition},
% %   author={Saeed, Anwar and Al-Hamadi, Ayoub and Niese, Robert and Elzobi, Moftah},
% %   booktitle={Signal Processing (ICSP), 2012 IEEE 11th International Conference on},
% %   volume={1},
% %   pages={623--627},
% %   year={2012},
% %   organization={IEEE}
% % }

normalize_by = (norm(lms(14,:)-lms(5,:))+norm(lms(14,:)-lms(6,:)))/2; % nose tip - inner eyebrow distance.
% ------------------------------------------------
% Feature 1 : 
f = mean([norm(lms(23,:)-lms(3,:)) , norm(lms(26,:)-lms(8,:)) ]);
g(18) = f/normalize_by;
% ------------------------------------------------
% Feature 2 : 
f = mean([norm(lms(23,:)-lms(35,:)) , norm(lms(26,:)-lms(35,:)) ]);
g(19) = f/normalize_by;
% ------------------------------------------------
% Feature 3 : 
f = norm(lms(32,:)-lms(38,:));
g(20) = f/normalize_by;
% ------------------------------------------------
% Feature 4 : 
f = norm(lms(35,:)-lms(41,:));
g(21) = f/normalize_by;
    % Features 5 and 6 require the mouth center : 
     mouth_center = mean([lms(45,:);lms(48,:)]); 
% ------------------------------------------------
% Feature 5 : 
f = norm(mouth_center - lms(35,:));
g(22) = f/normalize_by;
% ------------------------------------------------
% Feature 6 : 
f = norm(mouth_center - lms(41,:));
g(23) = f/normalize_by;

end

