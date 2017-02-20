function dh_baxter(q1,q2,q3,q4,q5,q6,q7)

% DH parameters = [theta, d, a, alpha]
DH(1,:) = [q1, 0.27035, 0.069, -pi/2]; % frame 0 to 1
DH(2,:) = [q2 + pi/2, 0, 0, pi/2]; % frame 1 to 2
DH(3,:) = [q3, 0.3640, 0.069, -pi/2]; % frame 2 to 3
DH(4,:) = [q4, 0, 0, pi/2]; % frame 3 to 4
DH(5,:) = [q5, 0.371 0.01, -pi/2]; % frame 4 to 5
DH(6,:) = [q6, 0, 0, pi/2]; % frame 5 to 6
DH(7,:) = [q7, 0.28, 0, 0]; % frame 6 to 7
T=eye(4);
for i = 1:7
     T=T*dh2mat(DH(i,3),DH(i,4),DH(i,2),DH(i,1));
end
origin_base = [0;0;0;1];

for i = 1:2
T(i).torso = translation(0.064,-0.258,0.121)*rpy2tran(0,0,-pi/4);
%T(i).base = translation(0,0,0.3)*rpy2tran(0,0,0);
end

EE_pose = T(1).torso*T(1).right*T(2).right*T(3).right*T(4).right*T(5).right*T(6).right*T(7).right

end