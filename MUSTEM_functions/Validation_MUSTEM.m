function MUSTEM_Bridge = Validation_MUSTEM(Feature_Matrix,index_test,X_train_max,X_train_min,Beta)

    %%%%%%%%%%%%%% 
    Feature_Matrix_Test = Feature_Matrix(:,:,:,index_test);
    [~,~,num_years,num_bridges_test] = size(Feature_Matrix_Test);

    %%% Calculate Transition Matrix for each test bridge 1993-2019
	X_test = squeeze(Feature_Matrix_Test);% size = num_features x num_years x num_bridges
    X_test = X_test(1:23,:,:);
    X_test_norm = (X_test-X_train_min)./(X_train_max-X_train_min);
    X_test_norm = cat(1,ones(1,num_years,num_bridges_test),X_test_norm);
    
    Transition_Matrix = zeros(7,7,(num_years-1)*num_bridges_test);

	k = 1;
    for brdg = 1:num_bridges_test
        for year = 1:num_years-1
			Theta = exp(Beta*X_test_norm(:,year,brdg));
			P11 = MUSTEM_P11(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P12 = MUSTEM_P12(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P13 = MUSTEM_P13(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P14 = MUSTEM_P14(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P15 = MUSTEM_P15(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P16 = MUSTEM_P16(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P17 = MUSTEM_P17(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			
			P22 = MUSTEM_P22(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P23 = MUSTEM_P23(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P24 = MUSTEM_P24(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P25 = MUSTEM_P25(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P26 = MUSTEM_P26(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P27 = MUSTEM_P27(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			
			P33 = MUSTEM_P33(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P34 = MUSTEM_P34(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P35 = MUSTEM_P35(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P36 = MUSTEM_P36(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P37 = MUSTEM_P37(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			
			P44 = MUSTEM_P44(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P45 = MUSTEM_P45(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P46 = MUSTEM_P46(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P47 = MUSTEM_P47(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			
			P55 = MUSTEM_P55(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P56 = MUSTEM_P56(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P57 = MUSTEM_P57(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			
			P66 = MUSTEM_P66(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			P67 = MUSTEM_P67(Theta(1),Theta(2),Theta(3),Theta(4),Theta(5),Theta(6));
			
			P77 = 1.0;
			
			T_brdg_year = [P77,  0,  0,  0,  0,  0,  0;
						   P67,P66,  0,  0,  0,  0,  0;
						   P57,P56,P55,  0,  0,  0,  0;
						   P47,P46,P45,P44,  0,  0,  0;
						   P37,P36,P35,P34,P33,  0,  0;
						   P27,P26,P25,P24,P23,P22,  0;
						   P17,P16,P15,P14,P13,P12,P11;];
		   
		    
			Transition_Matrix(:,:,k) = T_brdg_year;
			k = k+1;
        end
    end


    class_representation =[3,4,5,6,7,8,9];
    state_space = class_representation;
    od = 1;
    
    Prb_State_ALL = zeros(num_years,7,num_bridges_test);
	CR_test = squeeze(Feature_Matrix_Test(:,end,:,:)); %size = num_years x num_bridges
	
	
    for brdg = 1:num_bridges_test
        Transition_Matrix_i = Transition_Matrix(:,:,1+(num_years-1)*(brdg-1):(num_years-1)*brdg);
        %%% Markov Chain
        Prb_State = zeros(num_years-1,length(class_representation));%exclude the initial state, add later
        State_vector = zeros(num_years,length(class_representation)^od);% equivalent state vector including the initial one
        [~,index,~] = intersect(state_space', CR_test(1,brdg)','rows');
        State_vector(1,index) = 1;
        for step = 1:num_years-1 %exclude the initial state               
           Prb_State(step,:) = State_vector(step,:)*Transition_Matrix_i(:,:,step);
           %%% update state vector for next step
           State_vector_temp = State_vector(step,:)'.*Transition_Matrix_i(:,:,step);
           State_vector_mat = zeros(length(class_representation)^(od-1),length(class_representation));
           for i = 1:length(class_representation)^(od-1)
               State_vector_mat(i,:) = sum(State_vector_temp(1+(i-1)*length(class_representation):i*length(class_representation),:),1);
           end
           State_vector(step+1,:) = reshape(State_vector_mat,[1,length(class_representation)^od]);
        end
        %%%%% Also output the last record
        CCR = CR_test(1,brdg);
        Prb_State_current = zeros(1,7);   
        Prb_State_current(CCR-2) = 1;
        Prb_State = [Prb_State_current;Prb_State]; 
        Prb_State_ALL(:,:,brdg) = Prb_State;
    end


    MUSTEM_Bridge = zeros(num_years,num_bridges_test);
    for brdg = 1:num_bridges_test
        MUSTEM_Bridge(:,brdg) = Prb_State_ALL(:,:,brdg)*class_representation';
    end

end