function out1 = MUSTEM_23_f_prime_3(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_23_F_PRIME_3
%    OUT1 = MUSTEM_23_F_PRIME_3(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:23:33

t2 = -Theta2;
t3 = -Theta3;
t4 = exp(t2);
t5 = exp(t3);
t6 = Theta2+t3;
t7 = 1.0./t6;
t8 = t7.^2;
t9 = Theta2.*t5.*t7;
out1 = (t9+Theta2.*t4.*t8+t2.*t5.*t8)./(Theta2.*t4.*t7+t2.*t5.*t7);
