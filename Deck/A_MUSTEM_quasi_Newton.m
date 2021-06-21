clc
clear

fold = 1;
load(['./Samples\Samples_Labels_fold_',num2str(fold),'.mat']);


X = squeeze(Samples);

X_norm = (X(1:23,:)-min(X(1:23,:),[],2))./(max(X(1:23,:),[],2)-min(X(1:23,:),[],2));
% expand a unit row to indicate constant term in Beta
X_norm = [ones(1,length(Labels));X_norm];
%

%
m = 24;
n = 6;
%

I = X(end,:)';
J = Labels;

%rng(0);
Beta_0 = importdata('Beta_quasiNewton_fold_1.mat');
options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true);

my_fun = @(Beta)(MUSTEM_Loss_wGrad(Beta, X_norm, I, J));

Beta = fminunc(my_fun,Beta_0,options);
save(['Beta_quasiNewton_fold_',num2str(fold)],'Beta');

X_train_max = max(X(1:23,:),[],2);
X_train_min = min(X(1:23,:),[],2);
save(['X_train_maxmin_fold_',num2str(fold)]);


for fold = 2:10
    load(['./Samples\Samples_Labels_fold_',num2str(fold),'.mat']);


    X = squeeze(Samples);

    X_norm = (X(1:23,:)-min(X(1:23,:),[],2))./(max(X(1:23,:),[],2)-min(X(1:23,:),[],2));
    % expand a unit row to indicate constant term in Beta
    X_norm = [ones(1,length(Labels));X_norm];
    %

    %
    m = 24;
    n = 6;
    %

    I = X(end,:)';
    J = Labels;

    %rng(0);
    Beta_0 = importdata('Beta_quasiNewton_fold_1.mat');
    options = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton','SpecifyObjectiveGradient',true);

    my_fun = @(Beta)(MUSTEM_Loss_wGrad(Beta, X_norm, I, J));

    Beta = fminunc(my_fun,Beta_0,options);
    save(['Beta_quasiNewton_fold_',num2str(fold)],'Beta');
    
    X_train_max = max(X(1:23,:),[],2);
    X_train_min = min(X(1:23,:),[],2);
    save(['X_train_maxmin_fold_',num2str(fold)],'X_train_max','X_train_min');
end



































