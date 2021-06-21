function out1 = MUSTEM_57_f_prime_6(Theta1,Theta2,Theta3,Theta4,Theta5,Theta6)
%MUSTEM_57_F_PRIME_6
%    OUT1 = MUSTEM_57_F_PRIME_6(THETA1,THETA2,THETA3,THETA4,THETA5,THETA6)

%    This function was generated by the Symbolic Math Toolbox version 8.4.
%    21-Apr-2021 10:27:49

t2 = -Theta5;
t3 = -Theta6;
t4 = exp(t2);
t5 = exp(t3);
t6 = Theta5+t3;
t7 = 1.0./t6;
t8 = t7.^2;
t9 = Theta5.*t5.*t7;
out1 = -(t9+Theta5.*t4.*t8+t2.*t5.*t8)./(t4+t9+t2.*t4.*t7-1.0);
