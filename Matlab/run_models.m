% Copyright (c) by Armin Lederer (TUM) under BSD License 
% Last modified: Armin Lederer 06/2021

%% evaluate MoE-LoG-GP, gPoE-LoG-GP and ISSGP on SARCOS data

clear all; close all; clc
rng(0);

%% download data and structure it

if(~exist('Sarcos_train.mat','file'))
    url = 'http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat';
    Sarcos_train = websave('Sarcos_train',url);
end
if(~exist('Sarcos_test.mat','file'))
    url = 'http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat';
    Sarcos_test = websave('Sarcos_test',url);
end

load('Sarcos_train.mat');
load('Sarcos_test.mat');

p = randperm(size(sarcos_inv,1));
X_train = sarcos_inv(p,1:21)-mean(sarcos_inv(:,1:21),1);
Y_train = sarcos_inv(p,22)-mean(sarcos_inv(:,22));

clear sarcos_inv sarcos_inv_test url Sarcos_train Sarcos_test

%% evaluate methods in online learning

[smse_moe,msll_moe,tpred_moe,tup_moe] = run_MoE_LoG_GP(X_train,Y_train);
[smse_gpoe,msll_gpoe,tpred_gpoe,tup_gpoe] = run_gPoE_LoG_GP(X_train,Y_train);
[smse_issgp,msll_issgp,tpred_issgp,tup_issgp] = run_ISSGP(X_train,Y_train);


%% illustrate results

figure(); hold on;
semilogy(smse_moe);
semilogy(smse_gpoe);
semilogy(smse_issgp);
xlabel('N_{train}'); ylabel('SMSE'); title('regression error');
legend('MoE-LoG-GP', 'gPoE-LoG-GP','ISSGP');

figure(); hold on;
plot(msll_moe);
plot(msll_gpoe);
plot(msll_issgp);
xlabel('N_{train}'); ylabel('MSLL'); title('predictive distribution quality');
legend('MoE-LoG-GP', 'gPoE-LoG-GP','ISSGP');

figure(); hold on;
semilogy(tpred_moe);
semilogy(tpred_gpoe);
semilogy(tpred_issgp);
xlabel('N_{train}'); ylabel('t_{up}'); title('prediction time');
legend('MoE-LoG-GP', 'gPoE-LoG-GP','ISSGP');

figure(); hold on;
semilogy(tup_moe);
semilogy(tup_gpoe);
semilogy(tup_issgp);
xlabel('N_{train}'); ylabel('t_{up}'); title('update time');
legend('MoE-LoG-GP', 'gPoE-LoG-GP','ISSGP');