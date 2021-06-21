function Pij = MUSTEM_P16(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_P16
%    PIJ = MUSTEM_P16(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:17:47

t2 = -Theta2;
t3 = -Theta3;
t4 = -Theta4;
t5 = -Theta5;
t6 = -Theta6;
t7 = Theta1+t2;
t8 = Theta1+t3;
t9 = Theta1+t4;
t10 = Theta2+t3;
t11 = Theta1+t5;
t12 = Theta2+t4;
t13 = Theta1+t6;
t14 = Theta2+t5;
t15 = Theta3+t4;
t16 = Theta2+t6;
t17 = Theta3+t5;
t18 = Theta3+t6;
t19 = Theta4+t5;
t20 = Theta4+t6;
t21 = Theta5+t6;
t22 = 1.0./t7;
t23 = 1.0./t8;
t24 = 1.0./t9;
t25 = 1.0./t10;
t26 = 1.0./t11;
t27 = 1.0./t12;
t28 = 1.0./t13;
t29 = 1.0./t14;
t30 = 1.0./t15;
t31 = 1.0./t16;
t32 = 1.0./t17;
t33 = 1.0./t18;
t34 = 1.0./t19;
t35 = 1.0./t20;
t36 = 1.0./t21;
Pij = Theta1.*Theta2.*Theta3.*Theta4.*Theta5.*t22.*t25.*t27.*t29.*t31.*exp(t2)+Theta1.*Theta2.*Theta3.*Theta4.*Theta5.*t24.*t27.*t30.*t34.*t35.*exp(t4)+Theta1.*Theta2.*Theta3.*Theta4.*Theta5.*t28.*t31.*t33.*t35.*t36.*exp(t6)+Theta1.*Theta3.*Theta4.*Theta5.*t2.*t23.*t25.*t30.*t32.*t33.*exp(t3)+Theta1.*Theta3.*Theta4.*Theta5.*t2.*t26.*t29.*t32.*t34.*t36.*exp(t5)+Theta1.*Theta3.*Theta4.*Theta5.*t2.*t22.*t23.*t24.*t26.*t28.*exp(-Theta1);
