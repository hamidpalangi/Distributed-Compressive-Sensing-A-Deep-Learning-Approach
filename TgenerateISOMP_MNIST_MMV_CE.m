function [rMM,sMM] = TgenerateISOMP_MNIST_MMV_CE(phi_m,L,DataPath,nImage,Idim,nsparseTrain,Bdim,TransformProps)

% This function generates training samples for MMV scenario
%% Contact: Hamid Palangi, Email: hamidp@ece.ubc.ca

% L: number of measurement vectors
% nImage: number of images per channel.

% Signal definition
n = size(phi_m,2); % length of the signal
m = size(phi_m,1); % number of measurements
k = nsparseTrain;
% The number of training samples is "M = nImage*k".
% Bdim: block dimension
noise = TransformProps.noise;
load(DataPath);
if strcmp(TransformProps.DataType,'Tr')
    Ims = ImsTr;
    clear ImsTr;
elseif strcmp(TransformProps.DataType,'Val')
    Ims = ImsVal;
    clear ImsVal;
elseif strcmp(TransformProps.DataType,'Test')
    Ims = ImsTest;
    clear ImsTest;
end
%% Generating nImage random sparse signals and measurements
    rM = zeros(m,L,k);
    sM = zeros(n,L,k);
    rMM = [];
    sMM = [];
    B2 = [];
    for iL=1:L
        Im1 = Ims{iL,1};
        for i2=1:nImage % for the # of images per channel.
            Im=Im1(:,:,i2);
            Im=imresize(Im,[Idim,Idim]);               
            B0 = im2col(Im,[Bdim Bdim],'distinct');
            B1 = zeros( Bdim*Bdim , size(B0,2) );
            for iB0=1:size(B0,2) % for the all number of blocks in image
                B1(:,iB0) = mnist_fun(B0(:,iB0),n,k);
            end                
            B2 = [B2 B1];
        end
    end
    nBlockPerImage = Idim*Idim / (Bdim*Bdim);
    nBlockPerChannel = nBlockPerImage * nImage;
    for i2B=1:nBlockPerChannel
        S0 = zeros(n,L);
        for iL=1:L
            S0(:,iL) = B2( : , i2B + (iL-1)*nBlockPerChannel ); % amplitudes of sparse signal
        end
        Y = phi_m*S0 + noise; % Measurements
        j = 1; rM(:,:,j) = Y; sM(:,:,j) = S0;
        cell_I = cell(L,1); 
% Outputs of following "while" are sM (targets) and rM (inputs)
        while j<k
            j = j+1;
            Stemp = sM(:,:,j-1);
            Rtemp = rM(:,:,j);
            for iL=1:L
                I1 = cell_I{iL,1};
                s1 = Stemp(:,iL);                
                [val1,idx1] = max( abs ( s1 ) );
                I1 = [I1;idx1];
                s1(I1) = 0;                
                phiT = phi_m(:,I1);
                x_temp = pinv(phiT)*Y(:,iL);  % Least squares
                Rtemp(:,iL) = Y(:,iL)-phiT*x_temp;
                Stemp(:,iL) = s1;
                cell_I{iL,1} = I1;
            end
            rM(:,:,j) = Rtemp;
            sM(:,:,j) = Stemp;
        end
        rMM = cat(3,rMM,rM);
        sMM = cat(3,sMM,sM);
    end
% Normalization starts******************************************
    nTrainSamples = size(rMM,3);
    for i4=1:nTrainSamples
       temp4 = rMM(:,:,i4);        
       for iL=1:L
           maxval = max( abs( temp4(:,iL) ) );
           if maxval > 1e-6 % To prevent division by zero.
                temp4(:,iL) = temp4(:,iL) / maxval;
           end
       end
       rMM(:,:,i4) = temp4;
    end
    nTrainSamples = size(sMM,3);
    for i4=1:nTrainSamples
       temp5 = sMM(:,:,i4); 
       for iL=1:L
           [val1,idx1] = max( abs ( temp5(:,iL) ) );
           temp6 = zeros(n,1);
           temp6(idx1,1) = 1;
           temp5(:,iL) = temp6;
       end
       sMM(:,:,i4) = temp5;
    end
% Normalization ends******************************************
end