function out1 = MUSTEM_37_f_prime_4(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_37_F_PRIME_4
%    OUT1 = MUSTEM_37_F_PRIME_4(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:26:47

t2 = -Theta3;
t3 = -Theta4;
t4 = -Theta5;
t5 = -Theta6;
t6 = exp(t2);
t7 = exp(t3);
t8 = exp(t4);
t9 = exp(t5);
t10 = Theta3+t3;
t11 = Theta3+t4;
t12 = Theta3+t5;
t13 = Theta4+t4;
t14 = Theta4+t5;
t15 = Theta5+t5;
t16 = 1.0./t10;
t18 = 1.0./t11;
t19 = 1.0./t12;
t20 = 1.0./t13;
t22 = 1.0./t14;
t24 = 1.0./t15;
t17 = t16.^2;
t21 = t20.^2;
t23 = t22.^2;
t25 = Theta3.*t7.*t16;
t27 = Theta4.*t2.*t7.*t16.*t20;
t26 = Theta4.*t20.*t25;
t28 = Theta5.*t22.*t26;
out1 = -(t25+t27+t28+t20.*t25+Theta3.*t6.*t17+Theta5.*t23.*t26+t2.*t7.*t17+t2.*t6.*t16.*t18+t2.*t8.*t18.*t20+Theta3.*Theta4.*t7.*t17.*t20+Theta3.*Theta4.*t8.*t18.*t21+Theta4.*Theta5.*t21.*t22.*t25+Theta4.*t2.*t6.*t17.*t18+Theta4.*t2.*t7.*t16.*t21+Theta3.*Theta5.*t6.*t16.*t18.*t19+Theta3.*Theta5.*t8.*t18.*t20.*t24+Theta5.*t2.*t7.*t16.*t20.*t22+Theta5.*t2.*t9.*t19.*t22.*t24+Theta3.*Theta4.*Theta5.*t6.*t17.*t18.*t19+Theta3.*Theta4.*Theta5.*t9.*t19.*t23.*t24+Theta4.*Theta5.*t2.*t7.*t17.*t20.*t22+Theta4.*Theta5.*t2.*t8.*t18.*t21.*t24)./(t6+t25+t27+t28+t2.*t6.*t16+Theta3.*Theta4.*t6.*t16.*t18+Theta3.*Theta4.*t8.*t18.*t20+Theta3.*Theta4.*Theta5.*t9.*t19.*t22.*t24+Theta4.*Theta5.*t2.*t6.*t16.*t18.*t19+Theta4.*Theta5.*t2.*t8.*t18.*t20.*t24-1.0);
