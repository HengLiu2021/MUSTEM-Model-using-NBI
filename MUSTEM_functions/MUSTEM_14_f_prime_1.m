function out1 = MUSTEM_14_f_prime_1(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_14_F_PRIME_1
%    OUT1 = MUSTEM_14_F_PRIME_1(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:16:44

t2 = -Theta1;
t3 = -Theta2;
t4 = -Theta3;
t5 = -Theta4;
t6 = exp(t2);
t7 = exp(t3);
t8 = exp(t4);
t9 = exp(t5);
t10 = Theta1+t3;
t11 = Theta1+t4;
t12 = Theta1+t5;
t13 = Theta2+t4;
t14 = Theta2+t5;
t15 = Theta3+t5;
t16 = 1.0./t10;
t18 = 1.0./t11;
t20 = 1.0./t12;
t21 = 1.0./t13;
t23 = 1.0./t14;
t24 = 1.0./t15;
t17 = t16.^2;
t19 = t18.^2;
t22 = t20.^2;
t25 = Theta1.*Theta2.*Theta3.*t6.*t16.*t18.*t20;
out1 = -(t25+Theta2.*Theta3.*t7.*t16.*t21.*t23+Theta2.*Theta3.*t9.*t20.*t23.*t24+Theta3.*t3.*t6.*t16.*t18.*t20+Theta3.*t3.*t8.*t18.*t21.*t24+Theta1.*Theta2.*Theta3.*t6.*t16.*t19.*t20+Theta1.*Theta2.*Theta3.*t6.*t17.*t18.*t20+Theta1.*Theta2.*Theta3.*t6.*t16.*t18.*t22+Theta1.*Theta2.*Theta3.*t8.*t19.*t21.*t24+Theta2.*Theta3.*t2.*t7.*t17.*t21.*t23+Theta2.*Theta3.*t2.*t9.*t22.*t23.*t24)./(t25+Theta1.*Theta2.*Theta3.*t8.*t18.*t21.*t24+Theta2.*Theta3.*t2.*t7.*t16.*t21.*t23+Theta2.*Theta3.*t2.*t9.*t20.*t23.*t24);
