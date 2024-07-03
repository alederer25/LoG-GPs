classdef ISSGP < handle
%class for iterative learning with incremental sparse spectrum GPs
% E: state space dimension, N number of training samples
% Copyright (c) by Armin Lederer (TUM) under BSD License 
% Last modified: Armin Lederer 06/2021

    properties
        hyper; % hyperparameter
        X;% all input training data
        Y;% all target data
        sizey;% the dimension of y
        d;%the number of features
        R;
        w;
        b;
        ome;
        alpha; %prediction vector
        A;
    end
    
    methods
        function obj = ISSGP(sizex, sizey, d, hyper)
            obj.hyper.sigmaL = hyper(1:end-2);
            obj.hyper.sigmaF = hyper(end-1);
            obj.hyper.sigmaN = hyper(end);
            obj.d=d;
            obj.sizey = sizey;
            obj.R = obj.hyper.sigmaN.*eye(2*d);
            obj.w = zeros(2*d,1);
            obj.b = zeros(2*d,1);
            mu=zeros(d,sizex);
            sigma=diag(1./obj.hyper.sigmaL.^2);
            obj.ome = mvnrnd(mu,sigma);
            obj.A = obj.R'*obj.R;
        end
                
        function update(obj, x, y)
            %UPDATE b,R,w
            phi = [cos(obj.ome*x)',sin(obj.ome*x)']'.*obj.hyper.sigmaF./sqrt(obj.d);
            obj.A = obj.A+phi*phi';
            obj.R = chol(obj.A)';
            obj.b = obj.b+phi*y;
            obj.w = obj.R'\(obj.R\obj.b);
        end
        
        function [pred,var] = predict(obj, x)
            %PREDICT predicts the output with the GP
            pred = zeros(size(x,2),obj.sizey);
            var = ones(size(x,2),obj.sizey);
            for i = 1:size(x,2)
                phix = [cos(obj.ome*x(:,i))',sin(obj.ome*x(:,i))']'.*obj.hyper.sigmaF./sqrt(obj.d);
                pred(i) = phix'*obj.w;
                v = obj.R\phix;
                var(i) = obj.hyper.sigmaN^2*(v'*v);
                
            end
        end 
    end
end