function Pij = MUSTEM_P47(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_P47
%    PIJ = MUSTEM_P47(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:27:29

t2 = -Theta4;
t3 = -Theta5;
t4 = -Theta6;
t5 = exp(t2);
t6 = exp(t3);
t7 = Theta4+t3;
t8 = Theta4+t4;
t9 = Theta5+t4;
t10 = 1.0./t7;
t11 = 1.0./t8;
t12 = 1.0./t9;
Pij = -t5+Theta4.*t5.*t10+t2.*t6.*t10+Theta4.*Theta5.*t6.*t10.*t12+Theta5.*t2.*t5.*t10.*t11+Theta5.*t2.*t11.*t12.*exp(t4)+1.0;