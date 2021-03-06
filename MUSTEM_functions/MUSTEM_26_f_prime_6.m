function out1 = MUSTEM_26_f_prime_6(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_26_F_PRIME_6
%    OUT1 = MUSTEM_26_F_PRIME_6(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:24:06

t2 = -Theta2;
t3 = -Theta3;
t4 = -Theta4;
t5 = -Theta5;
t6 = -Theta6;
t7 = exp(t2);
t8 = exp(t3);
t9 = exp(t4);
t10 = exp(t5);
t11 = exp(t6);
t12 = Theta2+t3;
t13 = Theta2+t4;
t14 = Theta2+t5;
t15 = Theta3+t4;
t16 = Theta2+t6;
t17 = Theta3+t5;
t18 = Theta3+t6;
t19 = Theta4+t5;
t20 = Theta4+t6;
t21 = Theta5+t6;
t22 = 1.0./t12;
t23 = 1.0./t13;
t24 = 1.0./t14;
t25 = 1.0./t15;
t26 = 1.0./t16;
t27 = 1.0./t17;
t29 = 1.0./t18;
t30 = 1.0./t19;
t32 = 1.0./t20;
t34 = 1.0./t21;
t28 = t26.^2;
t31 = t29.^2;
t33 = t32.^2;
t35 = t34.^2;
t36 = Theta2.*Theta3.*Theta4.*Theta5.*t11.*t26.*t29.*t32.*t34;
out1 = (Theta3.*Theta4.*Theta5.*t2.*t8.*t22.*t25.*t27.*t31+Theta3.*Theta4.*Theta5.*t2.*t10.*t24.*t27.*t30.*t35+Theta3.*Theta4.*Theta5.*t2.*t11.*t26.*t29.*t32.*t34+Theta2.*Theta3.*Theta4.*Theta5.*t7.*t22.*t23.*t24.*t28+Theta2.*Theta3.*Theta4.*Theta5.*t9.*t23.*t25.*t30.*t33+Theta2.*Theta3.*Theta4.*Theta5.*t11.*t26.*t29.*t32.*t35+Theta2.*Theta3.*Theta4.*Theta5.*t11.*t26.*t29.*t33.*t34+Theta2.*Theta3.*Theta4.*Theta5.*t11.*t26.*t31.*t32.*t34+Theta2.*Theta3.*Theta4.*Theta5.*t11.*t28.*t29.*t32.*t34)./(t36+Theta3.*Theta4.*Theta5.*t2.*t8.*t22.*t25.*t27.*t29+Theta3.*Theta4.*Theta5.*t2.*t10.*t24.*t27.*t30.*t34+Theta2.*Theta3.*Theta4.*Theta5.*t7.*t22.*t23.*t24.*t26+Theta2.*Theta3.*Theta4.*Theta5.*t9.*t23.*t25.*t30.*t32);
