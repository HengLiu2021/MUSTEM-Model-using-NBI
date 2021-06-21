function [Loss,Grad] = MUSTEM_Loss_wGrad(Beta_iter, X_norm, I, J)
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
    
    num_skip = 0;
    %%%%%%%%%%%%%
    i = 1;
    j = 1;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
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
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 1;
    j = 2;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P12(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_12_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_12_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_12_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_12_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_12_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_12_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 1;
    j = 3;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P13(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_13_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_13_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_13_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_13_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_13_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_13_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)    
    %%%%%%%%%%%%%
    i = 1;
    j = 4;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P14(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_14_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_14_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_14_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_14_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_14_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_14_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end

            
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 1;
    j = 5;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P15(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_15_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_15_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_15_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_15_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_15_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_15_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 1;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P16(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_16_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_16_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_16_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_16_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_16_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_16_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 1;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P17(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
                temp_1 = MUSTEM_17_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_17_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_17_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_17_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_17_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_17_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 2;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P22(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_22_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_22_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_22_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_22_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_22_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_22_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 2;
    j = 3;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P23(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_23_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_23_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_23_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_23_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_23_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_23_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 4;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P24(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)
  
                temp_1 = MUSTEM_24_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_24_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_24_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_24_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_24_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_24_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 5;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P25(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_25_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_25_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_25_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_25_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_25_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_25_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P26(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_26_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_26_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_26_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_26_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_26_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_26_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 2;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P27(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_27_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_27_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_27_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_27_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_27_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_27_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 3;
    j = 3;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P33(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_33_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_33_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_33_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_33_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_33_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_33_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 3;
    j = 4;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P34(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_34_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_34_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_34_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_34_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_34_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_34_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 3;
    j = 5;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P35(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_35_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_35_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_35_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_35_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_35_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_35_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 3;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P36(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_36_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_36_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_36_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_36_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_36_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_36_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 3;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P37(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_37_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_37_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_37_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_37_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_37_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_37_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 4;
    j = 4;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P44(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_44_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_44_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_44_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_44_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_44_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_44_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    %%%%%%%%%%%%%
    i = 4;
    j = 5;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P45(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_45_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_45_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_45_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_45_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_45_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_45_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 4;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P46(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_46_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_46_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_46_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_46_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_46_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_46_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 4;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P47(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_47_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_47_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_47_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_47_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_47_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_47_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 5;
    j = 5;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P55(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_55_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_55_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_55_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_55_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_55_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_55_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 5;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P56(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_56_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_56_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_56_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_56_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_56_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_56_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 5;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P57(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_57_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_57_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_57_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_57_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_57_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_57_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 6;
    j = 6;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P66(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_66_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_66_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_66_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_66_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_66_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_66_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 6;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P67(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_67_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_67_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_67_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_67_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_67_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_67_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    %%%%%%%%%%%%%
    i = 7;
    j = 7;
    %%%        
    sample_idx = find(and(I==i,J==j)==1); 
    L = length(sample_idx);

    der_temp = zeros(n,m,L);
    loss_temp = zeros(1,L);
    if L>0
        for s = 1:L
            Theta_iter = exp(Beta_iter*X_norm(:,sample_idx(s)));
            Pij_s = MUSTEM_P77(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6));
            if and(Pij_s >= 1e-8,Pij_s<=1)

                temp_1 = MUSTEM_77_f_prime_1(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(1)*X_norm(:,sample_idx(s))';
                temp_2 = MUSTEM_77_f_prime_2(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(2)*X_norm(:,sample_idx(s))';
                temp_3 = MUSTEM_77_f_prime_3(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(3)*X_norm(:,sample_idx(s))';
                temp_4 = MUSTEM_77_f_prime_4(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(4)*X_norm(:,sample_idx(s))';
                temp_5 = MUSTEM_77_f_prime_5(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(5)*X_norm(:,sample_idx(s))';
                temp_6 = MUSTEM_77_f_prime_6(Theta_iter(1),Theta_iter(2),Theta_iter(3),Theta_iter(4),Theta_iter(5),Theta_iter(6))*Theta_iter(6)*X_norm(:,sample_idx(s))';
                temp = [temp_1;temp_2;temp_3;temp_4;temp_5;temp_6];
            
                if or(isnan(sum(temp,'all')),isinf(sum(temp,'all')))
                    num_skip = num_skip+1;
                else
                    loss_temp(1,s) = log(Pij_s);
                    der_temp(1,:,s) = temp_1;
                    der_temp(2,:,s) = temp_2;
                    der_temp(3,:,s) = temp_3;
                    der_temp(4,:,s) = temp_4;
                    der_temp(5,:,s) = temp_5;
                    der_temp(6,:,s) = temp_6;
                end
            else
                num_skip = num_skip+1;
            end
        end
    end

    Loss_ij = sum(loss_temp,2);
    der_ij = sum(der_temp,3);
    %disp(Loss_ij)
    %disp(der_ij)
    Loss = Loss + Loss_ij;
    Grad = Grad + der_ij;
    %disp(Loss)
    %disp(Grad)
    
    Loss = Loss*(-1/(length(J)-num_skip));
    Grad = Grad*(-1/(length(J)-num_skip));
    %disp(num_skip)
end