function out1 = MUSTEM_14_f_prime_2(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_14_F_PRIME_2
%    OUT1 = MUSTEM_14_F_PRIME_2(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

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
t19 = 1.0./t12;
t20 = 1.0./t13;
t22 = 1.0./t14;
t24 = 1.0./t15;
t17 = t16.^2;
t21 = t20.^2;
t23 = t22.^2;
t25 = Theta1.*Theta2.*Theta3.*t7.*t16.*t20.*t22;
out1 = (t25+Theta1.*Theta3.*t6.*t16.*t18.*t19+Theta1.*Theta3.*t8.*t18.*t20.*t24+Theta3.*t2.*t7.*t16.*t20.*t22+Theta3.*t2.*t9.*t19.*t22.*t24+Theta1.*Theta2.*Theta3.*t6.*t17.*t18.*t19+Theta1.*Theta2.*Theta3.*t7.*t16.*t20.*t23+Theta1.*Theta2.*Theta3.*t7.*t16.*t21.*t22+Theta1.*Theta2.*Theta3.*t9.*t19.*t23.*t24+Theta2.*Theta3.*t2.*t7.*t17.*t20.*t22+Theta2.*Theta3.*t2.*t8.*t18.*t21.*t24)./(Theta1.*Theta2.*Theta3.*t6.*t16.*t18.*t19+Theta1.*Theta2.*Theta3.*t8.*t18.*t20.*t24+Theta2.*Theta3.*t2.*t7.*t16.*t20.*t22+Theta2.*Theta3.*t2.*t9.*t19.*t22.*t24);
