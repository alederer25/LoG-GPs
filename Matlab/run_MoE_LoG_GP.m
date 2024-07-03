function [smse,msll,t_pred,t_up] = run_MoE_LoG_GP(X,Y)
% IN: 
%   X       N x E   matrix of training inputs
%   Y       N x 1   vector of training targets
% OUT: 
%   smse    N x 1   standardized mean squared error
%   msll    N x 1   mean standardized log loss
%   t_pred  N x 1   computation times for predictions
%   t_pred  N x 1   computation times for updates
% E: state space dimension, N number of training samples
% Copyright (c) by Armin Lederer (TUM) under BSD License 
% Last modified: Armin Lederer 06/2021



% hyperparameter optimization
Npretrain = 1000;
disp('Hyperparameter training ...');
gp = fitrgp(X(1:Npretrain,:),Y(1:Npretrain,1),'KernelFunction','ardsquaredexponential','Standardize',false);

ls = gp.KernelInformation.KernelParameters(1:end-1);
sf = gp.KernelInformation.KernelParameters(end);
sn = gp.Sigma;


% MoE-LoG-GP learning
disp('Pre-training of MoE-LoG-GP ...');
MoE_LoG_GP = MOE_LOG_GP();
MoE_LoG_GP.sigL = min(10^6,ls);
MoE_LoG_GP.sigF = sf;
MoE_LoG_GP.sigN = sn;

MoE_LoG_GP.pts = 100;
MoE_LoG_GP.loadHyp = false;
MoE_LoG_GP.wo = 100;
MoE_LoG_GP.divMethod = 2;
MoE_LoG_GP.xSize = size(X,2);
try
    MoE_LoG_GP.N = 18;
    MoE_LoG_GP.setupData();
catch
    MoE_LoG_GP.N = 16;
    MoE_LoG_GP.setupData();
end



% Pretrain on existing data
for i = 1:Npretrain
    MoE_LoG_GP.update(X(i,:)',Y(i,:));    
end


disp('Online Learning with MoE-LoG-GP ...');
mu = zeros(length(Y)-Npretrain,1);
sig = zeros(length(Y)-Npretrain,1);
lik = zeros(length(Y)-Npretrain,1);
t_pred = zeros(length(Y)-Npretrain,1);
t_up = zeros(length(Y)-Npretrain,1);

f = waitbar(0);
for i = Npretrain+1:size(X,1)
    tic;
    [mu(i-Npretrain,1),sig(i-Npretrain,1),lik(i-Npretrain,1)] = MoE_LoG_GP.predict(X(i,:)',Y(i,:)); 
    t_pred(i-Npretrain) = toc;
    
    tic;
    MoE_LoG_GP.update(X(i,:)',Y(i,:)); 
    t_up(i-Npretrain) = toc;
    waitbar(i/(length(Y)-Npretrain),f);
end
close(f);

disp('Computing performance measures ...');
smse = cumsum((mu-Y(Npretrain+1:end)).^2)./linspace(1,length(Y)-Npretrain,length(Y)-Npretrain)'/var(Y);

% split up msll in chunks of size 1000
sy = var(Y);
my = mean(Y);
for i = 1:floor((size(X,1)-Npretrain)/1000)
    Yte = Y(Npretrain+(i-1)*1000+1:Npretrain+i*1000);
    likte = lik((i-1)*1000+1:i*1000);
    lik0 = max(1e-300,normpdf(Yte,my,sqrt(sy)));
    msll((i-1)*1000+1:i*1000) = - likte + log(lik0);
end
Yte = Y(Npretrain+i*1000+1:end);
likte = lik(i*1000+1:end);
msll(i*1000+1:length(Y)-Npretrain) = - likte + log(normpdf(Yte,my,sqrt(sy)));

msll = (cumsum(msll)./linspace(1,length(Y)-Npretrain,length(Y)-Npretrain))';

end




