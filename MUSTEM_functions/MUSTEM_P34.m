function Pij = MUSTEM_P34(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_P34
%    PIJ = MUSTEM_P34(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:26:14

t2 = -Theta4;
t3 = Theta3+t2;
t4 = 1.0./t3;
Pij = Theta3.*t4.*exp(t2)-Theta3.*t4.*exp(-Theta3);
