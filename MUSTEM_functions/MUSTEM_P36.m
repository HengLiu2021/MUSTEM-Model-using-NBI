function Pij = MUSTEM_P36(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_P36
%    PIJ = MUSTEM_P36(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:26:28

t2 = -Theta4;
t3 = -Theta5;
t4 = -Theta6;
t5 = Theta3+t2;
t6 = Theta3+t3;
t7 = Theta3+t4;
t8 = Theta4+t3;
t9 = Theta4+t4;
t10 = Theta5+t4;
t11 = 1.0./t5;
t12 = 1.0./t6;
t13 = 1.0./t7;
t14 = 1.0./t8;
t15 = 1.0./t9;
t16 = 1.0./t10;
Pij = Theta3.*Theta4.*Theta5.*t11.*t14.*t15.*exp(t2)+Theta3.*Theta4.*Theta5.*t13.*t15.*t16.*exp(t4)+Theta3.*Theta5.*t2.*t12.*t14.*t16.*exp(t3)+Theta3.*Theta5.*t2.*t11.*t12.*t13.*exp(-Theta3);