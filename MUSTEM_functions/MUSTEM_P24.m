function Pij = MUSTEM_P24(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_P24
%    PIJ = MUSTEM_P24(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:23:38

t2 = -Theta3;
t3 = -Theta4;
t4 = Theta2+t2;
t5 = Theta2+t3;
t6 = Theta3+t3;
t7 = 1.0./t4;
t8 = 1.0./t5;
t9 = 1.0./t6;
Pij = Theta2.*Theta3.*t7.*t8.*exp(-Theta2)+Theta2.*Theta3.*t8.*t9.*exp(t3)+Theta2.*t2.*t7.*t9.*exp(t2);