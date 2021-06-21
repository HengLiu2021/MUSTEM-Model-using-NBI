function [Loss,Grad,Hess] = MUSTEM_Loss_wGradHess(Beta_iter, X_norm, I, J)
    % reverse condition rating based on the paper
    % CR 9 = 1
    % CR 8 = 2
    % CR 7 = 3
    % CR 6 = 4
    % CR 5 = 5
    % CR 4 = 6
    % CR 3 = 7
    I = 1 + 9 - I;
    J = 1 + 9 - J;

   
    
    Loss = 0;
    [n,m] = size(Beta_iter);
    Grad = zeros(n,m);
    Hess = zeros(n*m,n*m);
    num_skip = 0;
    %%%%%%%%%%%%%
    i = 1;
    j = 1;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_11_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_11_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_11_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_11_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_11_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_11_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_11_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_11_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_11_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_11_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_11_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_11_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_11_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_11_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_11_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_11_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_11_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_11_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_11_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_11_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_11_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_11_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_11_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_11_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_11_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_11_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_11_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_11_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_11_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_11_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_11_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_11_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_11_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_11_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_11_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_11_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_11_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_11_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_11_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_11_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_11_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_11_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 1;
    j = 2;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_12_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_12_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_12_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_12_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_12_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_12_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_12_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_12_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_12_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_12_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_12_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_12_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_12_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_12_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_12_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_12_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_12_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_12_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_12_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_12_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_12_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_12_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_12_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_12_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_12_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_12_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_12_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_12_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_12_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_12_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_12_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_12_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_12_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_12_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_12_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_12_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_12_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_12_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_12_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_12_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_12_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_12_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 1;
    j = 3;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_13_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_13_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_13_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_13_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_13_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_13_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_13_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_13_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_13_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_13_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_13_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_13_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_13_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_13_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_13_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_13_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_13_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_13_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_13_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_13_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_13_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_13_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_13_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_13_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_13_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_13_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_13_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_13_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_13_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_13_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_13_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_13_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_13_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_13_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_13_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_13_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_13_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_13_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_13_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_13_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_13_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_13_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)  
    %%%%%%%%%%%%%
    i = 1;
    j = 4;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_14_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_14_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_14_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_14_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_14_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_14_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_14_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_14_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_14_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_14_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_14_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_14_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_14_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_14_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_14_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_14_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_14_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_14_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_14_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_14_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_14_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_14_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_14_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_14_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_14_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_14_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_14_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_14_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_14_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_14_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_14_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_14_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_14_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_14_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_14_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_14_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_14_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_14_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_14_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_14_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_14_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_14_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 1;
    j = 5;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_15_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_15_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_15_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_15_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_15_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_15_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_15_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_15_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_15_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_15_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_15_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_15_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_15_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_15_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_15_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_15_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_15_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_15_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_15_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_15_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_15_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_15_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_15_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_15_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_15_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_15_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_15_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_15_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_15_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_15_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_15_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_15_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_15_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_15_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_15_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_15_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_15_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_15_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_15_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_15_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_15_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_15_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 1;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_16_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_16_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_16_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_16_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_16_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_16_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_16_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_16_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_16_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_16_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_16_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_16_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_16_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_16_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_16_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_16_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_16_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_16_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_16_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_16_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_16_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_16_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_16_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_16_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_16_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_16_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_16_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_16_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_16_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_16_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_16_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_16_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_16_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_16_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_16_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_16_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_16_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_16_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_16_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_16_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_16_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_16_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 1;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_17_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_17_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_17_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_17_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_17_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_17_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_17_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_17_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_17_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_17_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_17_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_17_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_17_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_17_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_17_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_17_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_17_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_17_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_17_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_17_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_17_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_17_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_17_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_17_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_17_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_17_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_17_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_17_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_17_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_17_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_17_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_17_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_17_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_17_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_17_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_17_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_17_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_17_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_17_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_17_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_17_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_17_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 2;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_22_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_22_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_22_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_22_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_22_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_22_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_22_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_22_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_22_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_22_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_22_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_22_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_22_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_22_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_22_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_22_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_22_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_22_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_22_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_22_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_22_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_22_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_22_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_22_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_22_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_22_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_22_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_22_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_22_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_22_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_22_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_22_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_22_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_22_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_22_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_22_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_22_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_22_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_22_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_22_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_22_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_22_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 2;
    j = 3;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_23_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_23_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_23_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_23_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_23_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_23_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_23_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_23_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_23_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_23_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_23_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_23_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_23_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_23_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_23_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_23_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_23_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_23_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_23_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_23_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_23_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_23_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_23_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_23_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_23_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_23_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_23_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_23_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_23_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_23_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_23_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_23_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_23_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_23_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_23_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_23_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_23_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_23_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_23_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_23_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_23_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_23_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 4;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_24_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_24_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_24_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_24_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_24_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_24_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_24_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_24_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_24_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_24_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_24_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_24_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_24_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_24_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_24_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_24_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_24_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_24_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_24_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_24_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_24_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_24_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_24_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_24_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_24_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_24_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_24_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_24_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_24_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_24_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_24_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_24_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_24_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_24_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_24_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_24_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_24_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_24_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_24_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_24_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_24_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_24_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 5;
     %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_25_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_25_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_25_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_25_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_25_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_25_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_25_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_25_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_25_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_25_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_25_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_25_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_25_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_25_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_25_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_25_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_25_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_25_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_25_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_25_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_25_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_25_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_25_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_25_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_25_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_25_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_25_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_25_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_25_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_25_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_25_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_25_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_25_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_25_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_25_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_25_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_25_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_25_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_25_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_25_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_25_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_25_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_26_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_26_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_26_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_26_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_26_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_26_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_26_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_26_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_26_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_26_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_26_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_26_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_26_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_26_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_26_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_26_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_26_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_26_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_26_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_26_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_26_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_26_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_26_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_26_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_26_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_26_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_26_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_26_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_26_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_26_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_26_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_26_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_26_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_26_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_26_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_26_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_26_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_26_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_26_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_26_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_26_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_26_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_27_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_27_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_27_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_27_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_27_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_27_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_27_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_27_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_27_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_27_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_27_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_27_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_27_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_27_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_27_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_27_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_27_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_27_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_27_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_27_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_27_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_27_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_27_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_27_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_27_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_27_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_27_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_27_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_27_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_27_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_27_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_27_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_27_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_27_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_27_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_27_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_27_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_27_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_27_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_27_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_27_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_27_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 3;
    j = 3;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_33_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_33_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_33_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_33_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_33_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_33_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_33_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_33_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_33_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_33_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_33_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_33_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_33_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_33_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_33_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_33_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_33_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_33_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_33_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_33_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_33_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_33_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_33_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_33_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_33_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_33_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_33_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_33_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_33_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_33_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_33_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_33_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_33_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_33_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_33_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_33_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_33_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_33_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_33_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_33_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_33_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_33_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 3;
    j = 4;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_34_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_34_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_34_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_34_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_34_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_34_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_34_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_34_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_34_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_34_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_34_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_34_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_34_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_34_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_34_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_34_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_34_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_34_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_34_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_34_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_34_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_34_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_34_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_34_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_34_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_34_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_34_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_34_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_34_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_34_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_34_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_34_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_34_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_34_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_34_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_34_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_34_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_34_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_34_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_34_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_34_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_34_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 3;
    j = 5;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_35_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_35_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_35_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_35_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_35_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_35_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_35_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_35_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_35_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_35_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_35_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_35_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_35_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_35_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_35_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_35_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_35_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_35_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_35_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_35_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_35_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_35_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_35_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_35_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_35_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_35_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_35_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_35_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_35_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_35_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_35_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_35_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_35_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_35_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_35_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_35_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_35_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_35_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_35_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_35_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_35_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_35_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 3;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_36_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_36_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_36_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_36_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_36_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_36_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_36_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_36_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_36_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_36_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_36_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_36_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_36_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_36_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_36_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_36_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_36_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_36_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_36_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_36_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_36_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_36_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_36_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_36_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_36_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_36_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_36_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_36_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_36_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_36_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_36_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_36_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_36_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_36_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_36_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_36_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_36_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_36_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_36_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_36_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_36_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_36_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 3;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_37_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_37_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_37_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_37_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_37_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_37_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_37_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_37_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_37_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_37_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_37_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_37_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_37_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_37_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_37_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_37_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_37_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_37_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_37_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_37_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_37_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_37_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_37_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_37_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_37_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_37_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_37_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_37_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_37_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_37_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_37_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_37_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_37_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_37_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_37_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_37_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_37_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_37_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_37_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_37_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_37_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_37_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 4;
    j = 4;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_44_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_44_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_44_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_44_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_44_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_44_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_44_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_44_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_44_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_44_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_44_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_44_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_44_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_44_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_44_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_44_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_44_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_44_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_44_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_44_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_44_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_44_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_44_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_44_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_44_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_44_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_44_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_44_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_44_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_44_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_44_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_44_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_44_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_44_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_44_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_44_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_44_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_44_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_44_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_44_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_44_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_44_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 4;
    j = 5;
   %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_45_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_45_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_45_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_45_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_45_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_45_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_45_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_45_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_45_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_45_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_45_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_45_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_45_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_45_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_45_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_45_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_45_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_45_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_45_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_45_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_45_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_45_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_45_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_45_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_45_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_45_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_45_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_45_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_45_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_45_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_45_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_45_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_45_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_45_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_45_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_45_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_45_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_45_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_45_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_45_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_45_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_45_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 4;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_46_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_46_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_46_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_46_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_46_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_46_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_46_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_46_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_46_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_46_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_46_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_46_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_46_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_46_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_46_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_46_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_46_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_46_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_46_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_46_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_46_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_46_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_46_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_46_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_46_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_46_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_46_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_46_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_46_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_46_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_46_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_46_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_46_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_46_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_46_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_46_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_46_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_46_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_46_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_46_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_46_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_46_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 4;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_47_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_47_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_47_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_47_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_47_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_47_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_47_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_47_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_47_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_47_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_47_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_47_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_47_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_47_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_47_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_47_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_47_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_47_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_47_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_47_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_47_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_47_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_47_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_47_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_47_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_47_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_47_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_47_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_47_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_47_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_47_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_47_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_47_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_47_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_47_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_47_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_47_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_47_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_47_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_47_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_47_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_47_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 5;
    j = 5;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_55_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_55_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_55_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_55_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_55_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_55_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_55_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_55_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_55_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_55_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_55_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_55_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_55_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_55_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_55_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_55_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_55_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_55_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_55_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_55_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_55_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_55_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_55_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_55_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_55_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_55_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_55_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_55_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_55_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_55_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_55_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_55_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_55_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_55_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_55_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_55_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_55_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_55_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_55_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_55_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_55_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_55_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 5;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_56_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_56_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_56_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_56_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_56_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_56_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_56_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_56_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_56_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_56_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_56_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_56_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_56_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_56_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_56_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_56_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_56_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_56_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_56_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_56_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_56_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_56_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_56_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_56_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_56_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_56_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_56_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_56_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_56_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_56_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_56_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_56_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_56_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_56_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_56_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_56_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_56_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_56_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_56_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_56_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_56_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_56_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 5;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_57_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_57_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_57_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_57_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_57_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_57_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_57_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_57_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_57_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_57_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_57_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_57_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_57_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_57_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_57_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_57_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_57_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_57_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_57_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_57_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_57_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_57_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_57_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_57_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_57_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_57_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_57_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_57_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_57_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_57_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_57_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_57_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_57_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_57_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_57_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_57_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_57_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_57_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_57_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_57_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_57_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_57_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 6;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_66_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_66_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_66_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_66_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_66_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_66_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_66_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_66_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_66_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_66_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_66_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_66_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_66_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_66_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_66_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_66_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_66_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_66_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_66_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_66_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_66_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_66_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_66_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_66_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_66_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_66_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_66_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_66_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_66_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_66_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_66_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_66_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_66_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_66_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_66_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_66_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_66_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_66_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_66_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_66_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_66_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_66_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 6;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_67_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_67_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_67_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_67_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_67_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_67_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_67_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_67_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_67_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_67_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_67_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_67_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_67_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_67_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_67_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_67_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_67_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_67_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_67_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_67_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_67_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_67_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_67_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_67_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_67_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_67_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_67_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_67_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_67_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_67_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_67_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_67_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_67_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_67_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_67_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_67_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_67_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_67_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_67_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_67_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_67_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_67_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 7;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m);
    hess_temp = zeros(n*m,n*m);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P11(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_77_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_77_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_77_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_77_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_77_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_77_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                
                temp_1_1 = MUSTEM_77_f_prime_1_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_2 = MUSTEM_77_f_prime_1_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_3 = MUSTEM_77_f_prime_1_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_4 = MUSTEM_77_f_prime_1_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_5 = MUSTEM_77_f_prime_1_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_1_6 = MUSTEM_77_f_prime_1_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_2_1 = MUSTEM_77_f_prime_2_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_2 = MUSTEM_77_f_prime_2_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_3 = MUSTEM_77_f_prime_2_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_4 = MUSTEM_77_f_prime_2_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_5 = MUSTEM_77_f_prime_2_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_2_6 = MUSTEM_77_f_prime_2_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_3_1 = MUSTEM_77_f_prime_3_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_2 = MUSTEM_77_f_prime_3_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_3 = MUSTEM_77_f_prime_3_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_4 = MUSTEM_77_f_prime_3_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_5 = MUSTEM_77_f_prime_3_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_3_6 = MUSTEM_77_f_prime_3_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_4_1 = MUSTEM_77_f_prime_4_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_2 = MUSTEM_77_f_prime_4_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_3 = MUSTEM_77_f_prime_4_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_4 = MUSTEM_77_f_prime_4_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_5 = MUSTEM_77_f_prime_4_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_4_6 = MUSTEM_77_f_prime_4_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_5_1 = MUSTEM_77_f_prime_5_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_2 = MUSTEM_77_f_prime_5_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_3 = MUSTEM_77_f_prime_5_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_4 = MUSTEM_77_f_prime_5_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_5 = MUSTEM_77_f_prime_5_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_5_6 = MUSTEM_77_f_prime_5_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_6_1 = MUSTEM_77_f_prime_6_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_2 = MUSTEM_77_f_prime_6_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_3 = MUSTEM_77_f_prime_6_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_4 = MUSTEM_77_f_prime_6_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_5 = MUSTEM_77_f_prime_6_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                temp_6_6 = MUSTEM_77_f_prime_6_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*(X_norm(:,sample_idx(s))*X_norm(:,sample_idx(s))');
                
                temp_temp = [temp_1_1,temp_1_2,temp_1_3,temp_1_4,temp_1_5,temp_1_6;
                             temp_2_1,temp_2_2,temp_2_3,temp_2_4,temp_2_5,temp_2_6;
                             temp_3_1,temp_3_2,temp_3_3,temp_3_4,temp_3_5,temp_3_6;
                             temp_4_1,temp_4_2,temp_4_3,temp_4_4,temp_4_5,temp_4_6;
                             temp_5_1,temp_5_2,temp_5_3,temp_5_4,temp_5_5,temp_5_6;
                             temp_6_1,temp_6_2,temp_6_3,temp_6_4,temp_6_5,temp_6_6;];
                         
                if or(or(isnan(sum(temp,'all')),isinf(sum(temp,'all'))), or(isnan(sum(temp_temp,'all')),isinf(sum(temp_temp,'all'))))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp = der_temp + temp;
                    hess_temp = hess_temp + temp_temp;           
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij =der_temp;
    hess_ij = hess_temp;
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    Hess = Hess + hess_ij;
    %disp(Loss)
    %disp(Grad)
    
    Loss = Loss*(-1/(length(J)-num_skip));
    Grad = Grad*(-1/(length(J)-num_skip));
    Hess = Hess*(-1/(length(J)-num_skip));
    disp(num_skip)
end