function [smse,msll,t_pred,t_up]=run_ISSGP(X,Y)
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

Nfeat = 200;

% hyperparameter optimization
Npretrain = 1000;
disp('Hyperparameter training ...');
gp = fitrgp(X(1:Npretrain,:),Y(1:Npretrain,1),'KernelFunction','ardsquaredexponential','Standardize',false);

ls = gp.KernelInformation.KernelParameters(1:end-1);
sf = gp.KernelInformation.KernelParameters(end);
sn = gp.Sigma;


% I-SSGP learning
disp('Pre-training of ISSGP ...');
numFeat = Nfeat;
hyper = [ls;sf;sn];
issgp = ISSGP(size(X,2), 1, numFeat, hyper); 

%pretrain on existing data
for i = 1:Npretrain
    issgp.update(X(i,:)',Y(i,:));    
end

disp('Online Learning with ISSGP ...');
mu = zeros(length(Y)-Npretrain,1);
sig = zeros(length(Y)-Npretrain,1);
lik = zeros(length(Y)-Npretrain,1);
t_pred = zeros(length(Y)-Npretrain,1);
t_up = zeros(length(Y)-Npretrain,1);

f = waitbar(0);
for i = Npretrain+1:size(X,1)
    tic;
    [mu(i-Npretrain,1),sig(i-Npretrain,1)] = issgp.predict(X(i,:)'); 
    t_pred(i-Npretrain) = toc;
    
    tic;
    issgp.update(X(i,:)',Y(i,:)); 
    t_up(i-Npretrain) = toc;
    
    lik(i-Npretrain) = 0.5*mean((mu(i-Npretrain,1)-Y(i,1)).^2./(sig(i-Npretrain,1)+sn^2))+0.5*mean(log(sig(i-Npretrain,1)+sn^2))+0.5*log(2*pi);
    waitbar(i/(length(Y)-Npretrain),f);

end
close(f);

disp('Computing performance measures ...');
smse = cumsum((mu-Y(Npretrain+1:end)).^2)./linspace(1,length(Y)-Npretrain,length(Y)-Npretrain)'/var(Y);

%split up msll in chunks of size 1000
sy = var(Y);
my = mean(Y);
for i = 1:floor((size(X,1)-Npretrain)/1000)
    Yte = Y(Npretrain+(i-1)*1000+1:Npretrain+i*1000);
    likte = lik((i-1)*1000+1:i*1000);
    lik0 = max(1e-300,normpdf(Yte,my,sqrt(sy)));
    msll((i-1)*1000+1:i*1000) = likte + log(lik0);
end
Yte = Y(Npretrain+i*1000+1:end);
likte = lik(i*1000+1:end);
msll(i*1000+1:length(Y)-Npretrain) = likte + log(normpdf(Yte,my,sqrt(sy)));

msll = (cumsum(msll)./linspace(1,length(Y)-Npretrain,length(Y)-Npretrain))';
end


