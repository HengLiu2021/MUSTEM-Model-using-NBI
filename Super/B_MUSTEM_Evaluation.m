clc
clear
error_Bridge_MUSTEM = zeros(27,10);
for fold = 1:10
    load(['./Samples\Feature_Matrix_fold_',num2str(fold),'.mat']);
    load(['Beta_quasiNewton_fold_',num2str(fold),'.mat']);
    load(['X_train_maxmin_fold_',num2str(fold),'.mat']);
    
    %%% Data
    Feature_Matrix_Test = Feature_Matrix(:,:,:,index_test);
    [~,~,num_years,num_bridges_test] = size(Feature_Matrix_Test);
    class_representation = [3,4,5,6,7,8,9];
    Data_Bridge = squeeze(Feature_Matrix_Test(1,end,:,:));%27 Years X 1000 Bridges

    %%% Model
    MUSTEM_Bridge = Validation_MUSTEM(Feature_Matrix,index_test,X_train_max,X_train_min,Beta);
    
    error_Bridge_MUSTEM(:,fold) = sum((MUSTEM_Bridge-Data_Bridge).^2,2)/num_bridges_test;
end

%save('error_MUSTEM_deck.mat','error_Bridge_MUSTEM')