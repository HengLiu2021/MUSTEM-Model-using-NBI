function out1 = MUSTEM_47_f_prime_4(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_47_F_PRIME_4
%    OUT1 = MUSTEM_47_F_PRIME_4(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:27:30

t2 = -Theta4;
t3 = -Theta5;
t4 = -Theta6;
t5 = exp(t2);
t6 = exp(t3);
t7 = exp(t4);
t8 = Theta4+t3;
t9 = Theta4+t4;
t10 = Theta5+t4;
t11 = 1.0./t8;
t13 = 1.0./t9;
t15 = 1.0./t10;
t12 = t11.^2;
t14 = t13.^2;
t16 = Theta4.*t5.*t11;
t17 = t2.*t5.*t11;
t18 = Theta5.*t13.*t16;
out1 = -(t5+t17+t18+t5.*t11-t6.*t11+Theta4.*t6.*t12+Theta5.*t14.*t16+t2.*t5.*t12+Theta5.*t6.*t11.*t15+t3.*t5.*t11.*t13+t3.*t7.*t13.*t15+Theta4.*Theta5.*t5.*t12.*t13+Theta4.*Theta5.*t7.*t14.*t15+Theta5.*t2.*t6.*t12.*t15)./(t5+t17+t18+Theta4.*t6.*t11+Theta4.*Theta5.*t7.*t13.*t15+Theta5.*t2.*t6.*t11.*t15-1.0);
