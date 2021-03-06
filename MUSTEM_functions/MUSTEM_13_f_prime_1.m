function out1 = MUSTEM_13_f_prime_1(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_13_F_PRIME_1
%    OUT1 = MUSTEM_13_F_PRIME_1(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:16:36

t2 = -Theta1;
t3 = -Theta2;
t4 = -Theta3;
t5 = exp(t2);
t6 = exp(t3);
t7 = exp(t4);
t8 = Theta1+t3;
t9 = Theta1+t4;
t10 = Theta2+t4;
t11 = 1.0./t8;
t13 = 1.0./t9;
t15 = 1.0./t10;
t12 = t11.^2;
t14 = t13.^2;
t16 = Theta1.*Theta2.*t5.*t11.*t13;
out1 = -(t16+Theta2.*t6.*t11.*t15+t3.*t5.*t11.*t13+t3.*t7.*t13.*t15+Theta1.*Theta2.*t5.*t11.*t14+Theta1.*Theta2.*t5.*t12.*t13+Theta1.*Theta2.*t7.*t14.*t15+Theta2.*t2.*t6.*t12.*t15)./(t16+Theta1.*Theta2.*t7.*t13.*t15+Theta2.*t2.*t6.*t11.*t15);
