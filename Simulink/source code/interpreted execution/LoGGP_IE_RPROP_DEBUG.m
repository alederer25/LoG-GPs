classdef LoGGP_IE_RPROP_DEBUG < matlab.System & matlab.system.mixin.Propagates & matlab.system.mixin.SampleTime & matlab.system.mixin.CustomIcon
% Copyright (c) by Alejandro Ordonez-Conejo under BSD License 
% Last modified: Armin Lederer 2022-02
    % Gaussian process regression. Look in source code for instructions.
    %{
    Data limit per GP: amount of data in a GP that triggers a division
    Limit of LGPs: max. amount of local GPs
    size k of vector x: [k,1] size of xTrain and xTests
    hyperplane method:
        1: mediand
        2: mean
        3: (max+min)/2
    if loadHyp:
        set fileName with variables:
            sigmaF
            sigmaN
            lengthScale (can be ARD or single value)
        inputs sigma_N, sigma_F, length-scale do not matter
    if !loadhyp
        set hyperparameters in sigma_N, sigma_F, length-scale
        fileName does not matter
    %}
    
    properties(Nontunable, Logical)
        ard = true;
    end

    properties (Nontunable)
        
        pts = 50; %Data limit per LGP
        N = 10000; %max number of local GPs
        xSize = 6;%Size of vector x
        divMethod = 3; %Hyperplane method
        wo = 300; %Ratio Width/overlapping
        timeRate = 0.005;%Sample time
    end
    properties(Access = protected)
        %properties pts,X,Y,K,invK and alpha must be set to the desired
        %amount of training points
        count; %amount of local GPs
        localCount;%amount of data trained in each LGP
        X; %training samples
        Y; %training targets
        K; %covariance matrices
        invK;
        L; %cholesky factors
        alpha; %L'\(L\y)
        auxAlpha; % L\y
        medians; %vector of hyperplanes
        parent; %vector of parent model
        children; %line 1: left child, line 2: right child
        overlaps; %line 1:cutting dimension, line 2:size of overlapping region
        auxUbic; %map a GP with the position of its data (K,L,alpha,X,Y,auxAlpha)
        mNum;
        
        sigmaF ; %sigma_F
        lengthS ; %lenght-scale
        sigmaN ; %sigma_N
        dlik0; %previous gradient
        delta; %previous delta from Rprop
    end
    
    methods(Access = protected)
        
        function [yTest,mN,tU,tP,pts] = stepImpl(obj,xTrain,yTrain,xTest)
            rng(0);
            tic;
            obj.update(xTrain,yTrain);
            tU = toc;
            mN = obj.mNum;
            pts = sum(obj.localCount);
            tic;
            yTest = obj.predict(xTest);
            tP = toc;
        end
        
        function setupImpl(obj)
            %initialize data
            obj.count = 1;
            
            obj.X = zeros(obj.xSize, obj.pts * obj.N);
            obj.Y = zeros(1, obj.pts * obj.N);
            obj.K = zeros(obj.pts, obj.pts * obj.N);
            obj.alpha = zeros(obj.pts,obj.N);
            obj.invK =zeros(obj.pts,obj.pts*obj.N);
            obj.localCount = zeros(1,2* obj.N -1);
            
            obj.medians =  zeros(obj.xSize, 2*obj.N-1);
            
            obj.parent = zeros(1, 2 * obj.N-1);
            obj.children = -1*ones(2, 2 * obj.N-1);
            
            obj.overlaps =  zeros(2, 2 * obj.N-1);
            
            obj.auxUbic = zeros(1, 2 * obj.N-1);
            obj.auxUbic(1,1) = 1;
            
            obj.sigmaN = ones(1,obj.N);
            obj.sigmaF = ones(1,obj.N);
            if obj.ard
                obj.lengthS = ones(obj.xSize,obj.N);
                obj.delta = 0.1*ones(obj.xSize+2,obj.N);
                obj.dlik0 = ones(obj.xSize+2,obj.N);
            else 
                obj.lengthS = ones(1,obj.N);
                obj.delta = 0.1*ones(3,obj.N);
                obj.dlik0 = ones(3,obj.N);
            end
            obj.mNum = 0;
            
        end
        
        function num = getNumInputsImpl(~)
            % Define total number of inputs for system with optional inputs
            num = 3;
        end
        
        function icon = getIconImpl(~)
            icon = "LoG-GP_DEBUG";
            % icon = ["My","System"]; % Example: multi-line text icon
        end
        
        function [out,o2,o3,o4,o5] = getOutputSizeImpl(~)
            % Return size for each output port
            out = [1 1];
            o2 = out;o3 = out;o4 = out;o5 = out;
        end
        
        function [out,o2,o3,o4,o5] = getOutputDataTypeImpl(~)
            % Return data type for each output port
            out = 'double';
            o2 = out;o3 = out;o4 = out;o5 = out;
        end
        
        function [out,o2,o3,o4,o5] = isOutputComplexImpl(~)
            % Return true for each output port with complex data
            out = false;
            o2 = out;o3 = out;o4 = out;o5 = out;
        end
        
        function [out,o2,o3,o4,o5] = isOutputFixedSizeImpl(~)
            % Return true for each output port with fixed size
            out = true;
            o2 = out;o3 = out;o4 = out;o5 = out;
        end
        
        function sts = getSampleTimeImpl(obj)
            % Define sample time type and parameters
            
            sts = obj.createSampleTime("Type", "Discrete", ...
                "SampleTime", obj.timeRate);
        end
        
        function kern = kernel(obj, Xi, Xj,model)%squared exponential kernel
            kern = (obj.sigmaF(obj.auxUbic(model))^2)*...
                exp(-0.5*sum(((Xi-Xj).^2)./(obj.lengthS(:,obj.auxUbic(model)).^2),1))';
        end
        
        function kern = fkernel(~,Xi,hyp) %kernel of Xi and itself (square matrix)
            
            kern = zeros(size(Xi,2), size(Xi,2));
            if size(hyp,1)>3
                for p = 1:size(hyp,1)-2
                    [k1] = meshgrid(Xi(p,:));
                    kern = kern + (k1-k1').^2./hyp(p+2)^2;
                end
            else
                for p = 1:size(Xi,1)
                    [k1] = meshgrid(Xi(p,:));
                    kern = kern + (k1-k1').^2./hyp(3)^2;
                end
            end
            kern = hyp(1)^2*exp(-0.5*kern);
        end
        
        function m = mValue(obj, model,cutD)%compute the hyperplane
            if obj.divMethod == 1
                m = median(obj.X(cutD, (obj.auxUbic(model)-1)*obj.pts+1:...
                    obj.auxUbic(model)*obj.pts));
                return
            elseif obj.divMethod == 2
                m = mean(obj.X(cutD, (obj.auxUbic(model)-1)*obj.pts+1:...
                    obj.auxUbic(model)*obj.pts));
                return
            elseif obj.divMethod == 3
                m = (max(obj.X(cutD, (obj.auxUbic(model)-1)*obj.pts+1:...
                    obj.auxUbic(model)*obj.pts))+ min(obj.X(cutD, ...
                    (obj.auxUbic(model)-1)*obj.pts+1:...
                    obj.auxUbic(model)*obj.pts)))/2 ;
                return
            end
        end
        
        function obj = rprop(obj,model)
            % get the previous hyperparameters
            x0 = [obj.sigmaF(obj.auxUbic(model));...
                obj.sigmaN(obj.auxUbic(model));...
                obj.lengthS(:,obj.auxUbic(model))];
            
            dmax = 50; dmin = 1e-6; etap = 1.2; etam = 0.5;
            %obtain the previous likelihood
            
            %get new hyperparameters and likelohood
            x1  = x0 + sign (obj.dlik0(:,obj.auxUbic(model))).*...
                obj.delta(:,obj.auxUbic(model));
            if abs(x1(2))<0.01
                x1(2) = sign(x1(2))*0.01;
            end
            if abs(x1(1)/x1(2))>1000
                x1(1) = sign(x1(1))*x1(2)*1000;
            end
            dfx1 = obj.grad(x1,model);
            %update delta and check it fullfils requirements
            d1 = ((obj.dlik0(:,obj.auxUbic(model)).*dfx1)>0).*...
                obj.delta(:,obj.auxUbic(model))*etap...
                + ((obj.dlik0(:,obj.auxUbic(model)).*dfx1)<0).*...
                obj.delta(:,obj.auxUbic(model))*etam ...
                + (obj.dlik0(:,obj.auxUbic(model)).*dfx1==0).*...
                obj.delta(:,obj.auxUbic(model));
            d1 = d1.*(d1>=dmin & d1 <= dmax)+(d1<dmin)*dmin + (d1>dmax)*dmax;
            %update properties
            obj.dlik0(:,obj.auxUbic(model)) = dfx1;
            obj.delta(:,obj.auxUbic(model)) = d1;
            obj.sigmaF(obj.auxUbic(model)) = x1(1);
            obj.sigmaN(obj.auxUbic(model)) = x1(2);
            obj.lengthS(:,obj.auxUbic(model)) = x1(3:end);
        end
        
        function [ df] = grad(obj,hyps,model)
            df = zeros(size(hyps,1),1);
            
            Ki = obj.fkernel(  obj.X(:,(obj.auxUbic(model)-1)*obj.pts+1:...
                (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model)),...
                hyps  );
            Ki = Ki + (Ki==0)*exp(-725);
            Kn = Ki + eye(size(Ki,1))*hyps(2)^2;
            obj.K(1:obj.localCount(model),(obj.auxUbic(model)-1)*obj.pts+1:...
                (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model))=Kn;
            obj.alpha(1:obj.localCount(model),obj.auxUbic(model)) = Kn\obj.Y((obj.auxUbic(model)-1)*obj.pts+1:...
                (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model))';
            
            obj.invK(1:obj.localCount(model),(obj.auxUbic(model)-1)*obj.pts+1:...
                (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model)) = inv(Kn);
            
            %get the value needed in al derivatives
            auxDer = obj.alpha(1:obj.localCount(model),obj.auxUbic(model))*...
                obj.alpha(1:obj.localCount(model),obj.auxUbic(model))'-...
                obj.invK(1:obj.localCount(model),(obj.auxUbic(model)-1)*obj.pts+1:...
                (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model));
            
            
            %derivatives wrt sigmaF and sigmaN
            df(1) = 0.5*sum(sum( auxDer.* (Ki.*2./hyps(1))' ,2));
            df(2) = 0.5*sum(sum( auxDer.* (eye(size(Ki,2))*2*hyps(2))' ,2 ));
            %derivatives wrt ls
            if obj.ard == 1
                for i = 1:size(hyps,1)-2
                    [k1] = meshgrid(obj.X(i,(obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model)));
                    df(2+i) =  0.5*sum(sum( auxDer.*( Ki.*((k1-k1').^2 ./ hyps(2+i)^3))',2 ));
                end
            else
                df(3) = 0.5*sum( sum( auxDer.*( (-2/hyps(3))*log(Ki./hyps(1)^2).*Ki)' ,2 ) );
            end
        end
        
        function updateParam(obj,x,model)
            if obj.localCount(model) == 1 %first point in model
                pos = obj.auxUbic(model)-1;
                obj.K(1,(pos)*obj.pts+1) = obj.kernel(x, x, model) + obj.sigmaN(obj.auxUbic(model));
                obj.invK(1,(pos)*obj.pts+1) = obj.K(1,(pos)*obj.pts+1);
                obj.alpha(1,pos+1) = obj.K(1,(pos)*obj.pts+1)\obj.Y((pos)*obj.pts+1);
            end
            %update hyperparameters, K and alpha
            obj.rprop(model);
        end
        
        function addPoint(obj, x, y, model)
            if obj.localCount(model) < obj.pts %if the model is not full
                obj.X(:,(obj.auxUbic(model)-1)*obj.pts+1+obj.localCount(model)) = x;
                obj.Y((obj.auxUbic(model)-1)*obj.pts+1+obj.localCount(model)) = y;
                obj.localCount(model) = obj.localCount(model) + 1;
                obj.updateParam(x,model)
            end
            if obj.localCount(model) == obj.pts %if full
                div = 1;
                while div == 1 %divide until no child set has all the data
                    [div,model] = obj.divide(model);
                end
            end
        end
        
        function [div, childModel] =  divide(obj, model)
            if obj.parent(end)~= 0 %no memory for more divisions
                div = -1;
                childModel = -1;
                disp('no more divisions allowed')
            else
                %obtain cutting dimension
                [~,cutD]=max(max(obj.X(:, (obj.auxUbic(model)-1)*obj.pts+1:...
                    obj.auxUbic(model)*obj.pts),[],2)-min(obj.X(:, (obj.auxUbic(model)-1)*obj.pts+1:...
                    obj.auxUbic(model)*obj.pts),[],2));
                %obtain hyperplane
                mP = obj.mValue(model,cutD);
                %compute borders
                maxV = max(obj.X(cutD, (obj.auxUbic(model)-1)*obj.pts+1:...
                    (obj.auxUbic(model)-1)*obj.pts+obj.pts));
                minV = min(obj.X(cutD, ...
                    (obj.auxUbic(model)-1)*obj.pts+1:...
                    (obj.auxUbic(model)-1)*obj.pts+obj.pts));
                %compute overlapping region
                if (maxV-minV) == 0 %if all data is the same
                    o = 0.1; %overlapping size
                else
                    o  = (maxV-minV)/obj.wo;
                end
                obj.medians(model)=mP;
                obj.overlaps(1,model)=cutD;
                obj.overlaps(2,model)=o;
                
                xL = zeros(obj.xSize,obj.pts); %matrix with x values for the left model
                xR = zeros(obj.xSize,obj.pts); %matrix with x values for the right model
                yL = zeros(1,obj.pts); %vector with y values for the left model
                yR = zeros(1,obj.pts); %vector with y values for the left model
                
                lcount = 0;
                rcount = 0;
                iL = zeros(1,obj.pts); %left index order vector
                iR = zeros(1,obj.pts); %right index order vector
                
                for i=1:obj.pts
                    xD = obj.X(cutD,(obj.auxUbic(model)-1)*obj.pts+i);%x value in cut dimension
                    if xD<mP-o/2 %if in left set
                        lcount = lcount+1;
                        xL(:,lcount) = obj.X(:,(obj.auxUbic(model)-1)*obj.pts+i);
                        yL(lcount) = obj.Y((obj.auxUbic(model)-1)*obj.pts+i);
                        iL(lcount) = i;
                    elseif xD >= mP-o/2 && xD <= mP+o/2 %if in overlapping
                        pL = 0.5 + (xD-mP)/(o);
                        if pL>=rand() %select left side
                            lcount = lcount+1;
                            xL(:,lcount) = obj.X(:,(obj.auxUbic(model)-1)*obj.pts+i);
                            yL(lcount) = obj.Y((obj.auxUbic(model)-1)*obj.pts+i);
                            iL(lcount) = i;
                        else
                            rcount = rcount + 1;
                            xR(:,rcount) = obj.X(:,(obj.auxUbic(model)-1)*obj.pts+i);
                            yR(rcount) = obj.Y((obj.auxUbic(model)-1)*obj.pts+i);
                            iR(rcount) = i;
                        end
                    elseif xD>mP+o/2 %if in right
                        rcount = rcount + 1;
                        xR(:,rcount) = obj.X(:,(obj.auxUbic(model)-1)*obj.pts+i);
                        yR(rcount) = obj.Y((obj.auxUbic(model)-1)*obj.pts+i);
                        iR(rcount) = i;
                    end
                end
                
                obj.localCount(model) = 0; %divided set is now "empty"
                if obj.count == 1
                    obj.count = obj.count+1; %update the total number of sets
                else
                    obj.count = obj.count+2;
                end
                obj.children(:,model) = [obj.count obj.count+1]'; %assign the children
                obj.parent(obj.count:obj.count+1) = model; %assign the parent
                
                %update parameters of new models
                obj.localCount(obj.count) = lcount;
                obj.auxUbic(obj.count) = obj.auxUbic(model);
                
                obj.localCount(obj.count+1) = rcount;
                obj.auxUbic(obj.count+1) = max(obj.auxUbic)+1;
                
                if lcount == obj.pts  %if left set has all the points now
                    div = 1; % output to keep dividing
                    childModel = (obj.count);
                elseif rcount == obj.pts %if right set has all the points now
                    %move K and invK to the position of right model:
                    obj.K(1:end, [(obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.pts, ...
                        (obj.auxUbic(obj.count+1)-1)*obj.pts+1:...
                        (obj.auxUbic(obj.count+1)-1)*obj.pts+obj.pts]) =...
                        obj.K(1:end, [(obj.auxUbic(obj.count+1)-1)*obj.pts+1:...
                        (obj.auxUbic(obj.count+1)-1)*obj.pts+obj.pts, ...
                        (obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.pts]);
                    obj.invK(1:end, [(obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.pts, ...
                        (obj.auxUbic(obj.count+1)-1)*obj.pts+1:...
                        (obj.auxUbic(obj.count+1)-1)*obj.pts+obj.pts]) =...
                        obj.invK(1:end, [(obj.auxUbic(obj.count+1)-1)*obj.pts+1:...
                        (obj.auxUbic(obj.count+1)-1)*obj.pts+obj.pts, ...
                        (obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.pts]);
                    %move alpha to the position of the right model
                    obj.alpha(:,[obj.auxUbic(model),obj.auxUbic(obj.count+1)]) = ...
                        obj.alpha(:,[obj.auxUbic(obj.count+1),obj.auxUbic(model)]);
                    %copy hyperparameters into the right model
                    obj.sigmaF(obj.auxUbic(obj.count+1)) = obj.sigmaF(obj.auxUbic(model));
                    obj.sigmaN(obj.auxUbic(obj.count+1)) = obj.sigmaN(obj.auxUbic(model));
                    obj.lengthS(:,obj.auxUbic(obj.count+1)) = obj.lengthS(:,obj.auxUbic(model));
                    %copy previous gradient and delta
                    obj.delta(:,obj.auxUbic(obj.count+1)) = obj.delta(:,obj.auxUbic(model));
                    obj.dlik0(:,obj.auxUbic(obj.count+1)) = obj.dlik0(:,obj.auxUbic(model));
                    
                    div = 1; % output to keep dividing
                    childModel = (obj.count+1);
                else %update alpha and  L values for the new models
                    B = (1:obj.pts);
                    C = [iL(1:lcount),iR(1:rcount)];
                    newK = obj.K(1:end, (obj.auxUbic(model)-1)*obj.pts+1:...
                        (obj.auxUbic(model)-1)*obj.pts+obj.pts);
                    %permute K:
                    newK(B,:) =  newK(C,:);
                    newK(:,B) = newK(:,C);
                    %set child Ks
                    obj.K(1:lcount,(obj.auxUbic(obj.count)-1)*obj.pts+1:...
                        (obj.auxUbic(obj.count)-1)*obj.pts+lcount) = newK(1:lcount,1:lcount);
                    obj.K(1:rcount,(obj.auxUbic(obj.count+1)-1)*obj.pts+1:...
                        (obj.auxUbic(obj.count+1)-1)*obj.pts+rcount) = newK(lcount+1:end,lcount+1:end);
                    %set child invKs
                    obj.invK(1:lcount,(obj.auxUbic(obj.count)-1)*obj.pts+1:...
                        (obj.auxUbic(obj.count)-1)*obj.pts+lcount) = inv(newK(1:lcount,1:lcount));
                    obj.invK(1:rcount,(obj.auxUbic(obj.count+1)-1)*obj.pts+1:...
                        (obj.auxUbic(obj.count+1)-1)*obj.pts+rcount) = inv(newK(lcount+1:end,lcount+1:end));
                    %compute child alphas
                    obj.alpha(1:lcount, obj.auxUbic(obj.count)) =  newK(1:lcount,1:lcount)\yL(1:lcount)';
                    obj.alpha(1:rcount, obj.auxUbic(obj.count+1)) = newK(lcount+1:end,lcount+1:end)\yR(1:rcount)';
                    %set hyperparameters for the right child, left inherits
                    obj.sigmaF(obj.auxUbic(obj.count+1)) = obj.sigmaF(obj.auxUbic(model));
                    obj.sigmaN(obj.auxUbic(obj.count+1)) = obj.sigmaN(obj.auxUbic(model));
                    obj.lengthS(:,obj.auxUbic(obj.count+1)) = obj.lengthS(:,obj.auxUbic(model));
                    %set delta and dlik0 for right child, left inherits
                    obj.delta(:,obj.auxUbic(obj.count+1)) = obj.delta(:,obj.auxUbic(model));
                    obj.dlik0(:,obj.auxUbic(obj.count+1)) = obj.dlik0(:,obj.auxUbic(model));
                    
                    div = -1; %stop the divide loop
                    childModel = -1;
                end
                %relocate X Y
                obj.X(:, (obj.auxUbic(obj.count)-1)*obj.pts+1:...
                    (obj.auxUbic(obj.count)-1)*obj.pts+obj.pts) = xL;
                obj.X(:, (obj.auxUbic(obj.count+1)-1)*obj.pts+1:...
                    (obj.auxUbic(obj.count+1)-1)*obj.pts+obj.pts) = xR;
                obj.Y((obj.auxUbic(obj.count)-1)*obj.pts+1:...
                    (obj.auxUbic(obj.count)-1)*obj.pts+obj.pts) = yL;
                obj.Y((obj.auxUbic(obj.count+1)-1)*obj.pts+1:...
                    (obj.auxUbic(obj.count+1)-1)*obj.pts+obj.pts) = yR;
                obj.auxUbic(model) = 0; %parent model will not have data
            end
        end
        
        
        function [pL,pR] = activation(obj, x, model)
            if obj.children(1,model) == -1 %return zeros when model has no children
                pL = 0;
                pR = 0;
                return
            end
            mP = obj.medians(model); %hyperplane value
            xD = x(obj.overlaps(1,model)); %x value in cut dimension
            o = obj.overlaps(2,model); %overlapping region
            if xD < mP-o/2
                pL = 1;
            elseif  xD >= mP-o/2 && xD <= mP+o/2 %if in overlapping
                pL = 0.5+(xD-mP)/(o);
            else
                pL = 0;
            end
            pR = 1-pL;
        end
        
        function update(obj,x,y)
            model = 1;
            while obj.children(1,model)~=-1 %if model is a parent
                %search for the leaf to asign the point
                [pL, ~] = obj.activation(x, model);
                if pL >= rand()
                    model = obj.children(1,model);%left child
                else
                    model = obj.children(2,model);
                end
            end
            %add the model to the randomly selected model
            obj.addPoint(x,y,model)
        end
        
        function out = predict(obj,x)
            moP = zeros(2,1000);% line 1: active GPs, line 2: global probability
            mCount = 1; %number of GPs used for predictions
            moP(1,1) = 1; %start in root
            moP(2,1) = 1;
            %while all the GPs found are not leaves
            %get the GPs for prediction and thier global probabilites
            while ~isequal( obj.children(1,moP(1,1:mCount)) , -1*ones(1,mCount) )
                for j=1:mCount
                    [pL, pR] = obj.activation(x,moP(1,j));
                    if pL > 0 && pR == 0
                        moP(1,j) = obj.children(1,moP(1,j));
                        moP(2,j) = moP(2,j)*pL;
                    elseif pR > 0 && pL == 0
                        moP(1,j) = obj.children(2,moP(1,j));
                        moP(2,j) = moP(2,j)*pR;
                    elseif pL>0 && pR>0
                        mCount = mCount + 1;
                        moP(1,mCount) = obj.children(2,moP(1,j));
                        moP(2,mCount) = moP(2,j)*pR;
                        moP(1,j) = obj.children(1,moP(1,j));
                        moP(2,j) = moP(2,j)*pL;
                    end
                end
            end
            out = 0;
            %prediction: weigthing prediction with proabilities
            for i=1:mCount
                model = moP(1,i);
                pred = ( obj.kernel(obj.X(:,(obj.auxUbic(model)-1)*obj.pts+1:...
                    (obj.auxUbic(model)-1)*obj.pts+obj.localCount(model)), x, model) )' * ...
                    obj.alpha(1:obj.localCount(model),obj.auxUbic(model));
                out = out+pred*moP(2,i);
            end
            obj.mNum = mCount;
        end
    end
    
    methods(Access = protected, Static)
        function groups = getPropertyGroupsImpl
            % Section to always display above any tabs.
            alwaysSection = matlab.system.display.Section(...
                'Title','','PropertyList',{'ard','pts','N','xSize'});
            
            
            % Section for the value parameters
            valueSection = matlab.system.display.Section(...
                'Title','Sample time',...
                'PropertyList',{'timeRate'});
            
            % Section for the threshold parameters
            thresholdSection = matlab.system.display.Section(...
                'Title','Parameters',...
                'PropertyList',{'divMethod','wo'});
            
            % Group with two sections: the valueSection and thresholdSection sections
            mainTab = matlab.system.display.SectionGroup(...
                'Title','Main', ...
                'Sections',[valueSection,thresholdSection]);
            
            % Return an array with the group-less section, the group with
            % two sections, and the group with no sections.
            groups = [alwaysSection,mainTab];
        end
        
        function simMode = getSimulateUsingImpl
            % Return only allowed simulation mode in System block dialog
            simMode = "Interpreted execution";
        end
        
    end
end
