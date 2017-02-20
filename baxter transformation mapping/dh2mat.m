function T=dh2mat(a, alpha,d, theta)
    R=[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha); 
        sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha);
        0,          sin(alpha),         cos(alpha)];
    t=  [a*cos(theta); 
         a*sin(theta);
         d];
    T=[[R,           t];
       zeros(1,3),   1];
end