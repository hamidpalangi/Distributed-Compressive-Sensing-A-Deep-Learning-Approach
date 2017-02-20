% Copy right:
% Permission is granted for anyone to copy, use, modify, or distribute 
% this program and accompanying programs and documents for any purpose, 
% provided this copyright notice is retained and prominently displayed, 
% along with a note saying that the original programs are available from 
% Hamid Palangi, "hamidp@ece.ubc.ca". 
% The programs and documents are distributed without any warranty, 
% express or implied. As the programs were written for research purposes only, 
% they have not been tested to the degree that would be advisable in any important 
% application. All use of these programs is entirely at the user's own risk.

% This Mfile implements the LSTM-CS method and generates Fig.4 of the
% paper. To refer please use the following:
% @ARTICLE{hp_LSTM_CS, 
% author={Hamid Palangi and Rabab Ward and Li Deng}, 
% journal={IEEE Transactions on Signal Processing}, 
% title={Distributed Compressive Sensing: A Deep Learning Approach}, 
% year={2016}, 
% volume={64}, 
% number={17}, 
% pages={4504-4518}, 
% month={Sept},
% }

% In generating Fig.4, the codes for Bayesian Compressive Sensing from
% following references are used. Please refer to their website for copy
% right terms and conditions.
% [1] Bayesian and Multitask Compressive Sensing: "http://www.ece.duke.edu/~lcarin/bcs_ver0.1.zip"
% [2]  Sparse Signal Recovery with Temporally Correlated Source Vectors
% Using Sparse Bayesian Learning: "http://sccn.ucsd.edu/~zhang/TMSBL_code.zip"

%% Contact: Hamid Palangi, Email: hamidp@ece.ubc.ca

close all;clear all;clc;
rand('state', 1);
randn('state', 2);
addpath('Other_Methods\MT_CS_demo');
addpath('Other_Methods\BCS_demo');
addpath('Other_Methods\TMSBL_code');
% Signal definition and setting: START
Cost_Function = 'CE';% Cross Entropy.
CodeType = strcat(['MMV_MNIST_LSTM_Demo_' Cost_Function]);
trained_net_path = 'Trained_Network\';
DataPathTr = 'MNIST_Data\ImsTr.mat';
DataPathVal = 'MNIST_Data\ImsVal.mat';
DataPathTest = 'MNIST_Data\ImsTest.mat';
ncell = 512;
NetPath = strcat([trained_net_path CodeType '_ncell_' num2str(ncell)]);
StepSize = 0.0003;
maxEpochLSTM = 25;
maxEpochGeneral = 25;
numhidGeneral = 512;
Idim = 24;% image dimension
Bdim = 12;% image block dimension
n = Bdim*Bdim; % length of the signal
m = floor(n/2); % # of measurements
nsparseTrain = floor(m/4); % # of sparsity levels generated in train data.
nsparse = n; % # of sparsity levels to evaluate performance
nBatch = 300; % # of minibatches for LSTM training.
L = 4;  % number of sparse vectors in MMV
nImageTrain = 200; % # of images per channel for training.
nImageVal = 3; % # of images per channel for validation set.
nImageTest = 10; % # of images per channel for test set.
winSize = 1;
input_zero_mean = 0; % if 1 the input (r), is normalized to have zero mean.
input_random_permutation = 1;
NOISE = 1; % NOISE = 0 means that we don't have noise
sigma_noise = 0.005;
Biased = 0; % Biased = 1 means that a network with bias is used
%           % Biased = 0 means that a network without bias is used
phi1 = randn(n,n);  
phi = orth(phi1); % orthogonalizing the columns
% Randomly select m lines of phi and construct phi_m
sel = randperm(n);   
phi_m = phi(sel(1:m),:);
for i1=1:size(phi_m,2)
   phi_m(:,i1) = phi_m(:,i1)/norm(phi_m(:,i1)); 
end
% Minimum residual to stop solver.
% Without noise
if (NOISE == 0)
    res_min = L*(1e-5);
    noise = zeros(m,L);
else
% With noise
noise = sigma_noise*randn(m,L);
res_min = L*norm(noise(:));
end
TransformProps.noise = noise;
% Signal definition and setting: END
% Training and Validation set generation: START
TransformProps.DataType = 'Tr';
[In,T] = TgenerateISOMP_MNIST_MMV_CE(phi_m,L,DataPathTr,nImageTrain,Idim,nsparseTrain,Bdim,TransformProps);
TransformProps.DataType = 'Val';
[In_Dev,T_Dev] = TgenerateISOMP_MNIST_MMV_CE(phi_m,L,DataPathVal,nImageVal,Idim,nsparseTrain,Bdim,TransformProps);
% Training and Validation set generation: END
% Random permutation: START
perm_In = randperm( size(In,3) );
In = In(:,:,perm_In);
T = T(:,:,perm_In);
perm_In = randperm( size(In_Dev,3) );
In_Dev = In_Dev(:,:,perm_In);
T_Dev = T_Dev(:,:,perm_In);
% Random permutation: END
% LSTM Training: START
[Ws,ce_lstm,ce_lstm_dev] = LSTM_TrainCE('input train',In,'target train',T...
        ,'input validation',In_Dev,'target validation',T_Dev,'number of cells',ncell...
        ,'step size',StepSize,'epoch max',maxEpochLSTM,'number of mini-batches',nBatch,...
        'trained network path',NetPath);%,'initial weight matrices',Ws_init);
% LSTM Training: END
% Because the cost function for NWSOMP is mean square error not cross
% entropy.
%%%%%%%%%%%%%%%%%%%%
TransformProps.DataType = 'Tr';
[In,T] = TgenerateISOMP_MNIST_MMV(phi_m,L,DataPathTr,nImageTrain,Idim,nsparseTrain,Bdim,TransformProps);
TransformProps.DataType = 'Val';
[In_Dev,T_Dev] = TgenerateISOMP_MNIST_MMV(phi_m,L,DataPathVal,nImageVal,Idim,nsparseTrain,Bdim,TransformProps);
% Random permutation: START
perm_In = randperm( size(In,3) );
In = In(:,:,perm_In);
T = T(:,:,perm_In);
perm_In = randperm( size(In_Dev,3) );
In_Dev = In_Dev(:,:,perm_In);
T_Dev = T_Dev(:,:,perm_In);
% Random permutation: END
%%%%%%%%%%%%%%%%%%%%
In = reshape(In,[size(In,1) size(In,2)*size(In,3)]);
T = reshape(T,[size(T,1) size(T,2)*size(T,3)]);
In_Dev = reshape(In_Dev,[size(In_Dev,1) size(In_Dev,2)*size(In_Dev,3)]);
T_Dev = reshape(T_Dev,[size(T_Dev,1) size(T_Dev,2)*size(T_Dev,3)]);
if Biased == 1
    In = [In;ones(1,size(In,2))]; % input with bias
    In_Dev = [In_Dev;ones(1,size(In_Dev,2))]; % input with bias
end
NumberofInputNeurons = size(In,1);
MaxEpoch = maxEpochGeneral;
if Biased == 1
    NumberofHiddenNeurons = numhidGeneral;
    InputDim = NumberofInputNeurons-1;
    Winit = zeros(NumberofHiddenNeurons, InputDim +1);
    Winit(:,1:InputDim)=1 - 2*rand(NumberofHiddenNeurons,InputDim);
    Winit(:,InputDim+1:InputDim+1) = rand(NumberofHiddenNeurons,1); % initial weights for bias
elseif Biased == 0
    NumberofHiddenNeurons = numhidGeneral;
    InputDim = NumberofInputNeurons;
    Winit = zeros(NumberofHiddenNeurons, InputDim);
    Winit(:,1:InputDim)=1 - 2*rand(NumberofHiddenNeurons,InputDim);
end
Reg = 100;
disp('NWSOMP Training: START');
if Biased == 1    
    [MSE_Tot, MSE_Tg, MSE_Test,Wnew,Unew] = OneLayerBackprop_new_Biased(.5,.09,StepSize,In,T,numhidGeneral,Reg,MaxEpoch,NumberofInputNeurons,Winit,In_Dev,T_Dev);
elseif Biased == 0
    [MSE_Tot, MSE_Tg, MSE_Test,Wnew,Unew] = OneLayerBackprop_new(.5,.09,StepSize,In,T,numhidGeneral,Reg,MaxEpoch,NumberofInputNeurons,Winit,In_Dev,T_Dev);
end
disp('NWSOMP Training: END');
figure;
subplot(3,1,1);
plot(MSE_Tot);ylabel('MSE Training and Rec');xlabel('epoch');
subplot(3,1,2);
plot(MSE_Tg);ylabel('MSE Training');xlabel('epoch');
subplot(3,1,3);
plot(MSE_Test);ylabel('MSE Test Data');xlabel('epoch');
title(strcat([' MaxEpoch = ',num2str(MaxEpoch),', Reg = ',num2str(Reg),', Hidden units = ',num2str(numhidGeneral)]));
if (NOISE == 0 && Biased == 0)
    saveas(gcf,strcat(['Results\MSE_with_Regularization_' num2str(Reg) '_num_' num2str(numhidGeneral) '_StepSize_' num2str(StepSize) '_NoBias_n' num2str(n) CodeType '_MultiChannel_Complete.fig']));
elseif (NOISE == 0 && Biased == 1)
    saveas(gcf,strcat(['Results\MSE_with_Regularization_' num2str(Reg) '_num_' num2str(numhidGeneral) '_StepSize_' num2str(StepSize) '_n' num2str(n) CodeType '_MultiChannel_Complete.fig']));
elseif (NOISE == 1 && Biased == 0)
    saveas(gcf,strcat(['Results\MSE_with_Regularization_' num2str(Reg) '_num_' num2str(numhidGeneral) '_StepSize_' num2str(StepSize) '_Noisy_NoBias_n' num2str(n) CodeType '_MultiChannel_Complete.fig']));
elseif (NOISE == 1 && Biased == 1)
    saveas(gcf,strcat(['Results\MSE_with_Regularization_' num2str(Reg) '_num_' num2str(numhidGeneral) '_StepSize_' num2str(StepSize) '_Noisy_n' num2str(n) CodeType '_MultiChannel_Complete.fig']));    
end
close all;
% Performance Evaluation of Different Solvers: START
disp('Performance Evaluation: START');
disp('"K" is the number of non-zero entries in each sparse vector.');
k_vec = 1:nsparse;
k_vec = k_vec';
errNISOMP = zeros(length(k_vec),1);
errSOMP = zeros(length(k_vec),1);
errBCS = zeros(length(k_vec),1);
errMTBCS = zeros(length(k_vec),1);
errRao = zeros(length(k_vec),1);
errLSTM = zeros(length(k_vec),1);

B1=cell(nImageTest,L);
load(DataPathTest);
for iL=1:L
    Im1 = ImsTest{iL,1};
    for i33=1:nImageTest % number of test images per channel
        Im=Im1(:,:,i33);
        Im=imresize(Im,[Idim,Idim]);
        B0 = im2col(Im,[Bdim Bdim],'distinct');
        B1{i33,iL} = B0;
    end
end
nBlockPerImage = Idim*Idim / (Bdim*Bdim);
nBlockPerChannel = nBlockPerImage * nImageTest;
cell_SOMP = cell(L,1);
cell_Y = cell(L,1);
cell_NISOMP = cell(L,1);
cell_BCS = {L,1};
cell_MTBCS = cell(L,1);
cell_Rao = cell(L,1);
cell_LSTM = cell(L,1);
B0_SOMP = zeros(size(B0));
B0_Y = zeros(m,L);
B0_NISOMP = zeros(size(B0));
B0_BCS = zeros(size(B0));
B0_MTBCS = zeros(size(B0));
B0_Rao = zeros(size(B0));
B0_LSTM = zeros(size(B0));
for iL=1:L
    cell_SOMP{iL,1} = B0_SOMP;
    cell_Y{iL,1} = B0_Y;
    cell_NISOMP{iL,1} = B0_NISOMP;
    cell_BCS{iL,1} = B0_BCS;
    cell_MTBCS{iL,1} = B0_MTBCS;
    cell_Rao{iL,1} = B0_Rao;
    cell_LSTM{iL,1} = B0_LSTM;
end
for i2=1:length(k_vec)
    errSOMPavg = 0;
    errNISOMPavg = 0;       
    errBCSavg = 0;
    errMTBCSavg = 0;
    errRaoavg = 0;
    errLSTMavg = 0;
    k = k_vec(i2,1); % sparsity of signal   
    v=[];
    vY = [];
    vSOMP = [];
    vNISOMP = [];
    vBCS = [];
    vMTBCS = [];
    vRao = [];
    vLSTM = [];
    for i33=1:nImageTest
        snr_val = 0;
        for iB0 = 1 : nBlockPerImage % for the all number of blocks in image
            S0 = zeros(n,L);
            for iL=1:L
                temp1 = B1{i33,iL};
                S0(:,iL) = mnist_fun(temp1( : , iB0 ),n,k); % amplitudes of sparse signal
            end               
            if (NOISE == 0)
% without noise
                Y = phi_m*S0;
            else
% with noise
                Y = phi_m*S0;
                Y = Y + noise;               
            end
% output of NISOMP
            S_NISOMP = NISOMPnew(phi_m,Y,res_min,Wnew,Unew,numhidGeneral);
% output of SOMP            
            s_SOMP = SOMP(phi_m,Y,res_min);
% LSTM-CS
            S_LSTM = ISOMP_LSTM(phi_m,Y,res_min,Ws,winSize,input_zero_mean);
% BCS
            s_BCS = zeros(n,L);
                for iL=1:L
                    y1 = Y(:,iL);
                    initsigma2 = std(y1(:))^2/1e2;
                    [weights,used,sigma2,errbars] = BCS_fast_rvm(phi_m,y1,initsigma2,1e-8);
                    s_BCS1 = zeros(n,1);
                    s_BCS1(used) = weights;
                    s_BCS(:,iL) = s_BCS1;
                end
% MT_BCS % Multitask BCS
                a = 1e2/0.1; b = 1;
                for iMT=1:L
                   phiMT{1,iMT} = phi_m; 
                   yMT{1,iMT} = Y(:,iMT);
                end
                S_MT = mt_CS(phiMT,yMT,a,b,1e-8);
% T-SBL: Sparse Baysian learning
                S_Rao = TSBL(phi_m, Y);
            for iL=1:L
                rec_imblock = reshape(Y(:,iL),[Bdim/2 Bdim]);
                temp = cell_Y{iL,1};
                temp(:,iB0) = rec_imblock(:);
                cell_Y{iL,1} = temp;
                
                rec_imblock = reshape(s_SOMP(:,iL),[Bdim Bdim]);
                temp = cell_SOMP{iL,1};
                temp(:,iB0) = rec_imblock(:);
                cell_SOMP{iL,1} = temp;

                rec_imblock = reshape(S_NISOMP(:,iL),[Bdim Bdim]);
                temp = cell_NISOMP{iL,1};
                temp(:,iB0) = rec_imblock(:);
                cell_NISOMP{iL,1} = temp;

                rec_imblock = reshape(s_BCS(:,iL),[Bdim Bdim]);
                temp = cell_BCS{iL,1};
                temp(:,iB0) = rec_imblock(:);
                cell_BCS{iL,1} = temp;

                rec_imblock = reshape(S_MT(:,iL),[Bdim Bdim]);
                temp = cell_MTBCS{iL,1};
                temp(:,iB0) = rec_imblock(:);
                cell_MTBCS{iL,1} = temp;

                rec_imblock = reshape(S_Rao(:,iL),[Bdim Bdim]);
                temp = cell_Rao{iL,1};
                temp(:,iB0) = rec_imblock(:);
                cell_Rao{iL,1} = temp;
                
                rec_imblock = reshape(S_LSTM(:,iL),[Bdim Bdim]);
                temp = cell_LSTM{iL,1};
                temp(:,iB0) = rec_imblock(:);
                cell_LSTM{iL,1} = temp;
            end
        end  
        errSOMPavg2 = 0;
        errNISOMPavg2 = 0;
        errBCSavg2 = 0;
        errMTBCSavg2 = 0;
        errRaoavg2 = 0;
        errLSTMavg2 = 0;
        for iL=1:L
            Im = col2im(B1{i33,iL},[Bdim Bdim],[Idim Idim],'distinct');
            Y = col2im(cell_Y{iL,1},[Bdim/2 Bdim],[Idim/2 Idim],'distinct');
            Im_recSOMP = col2im(cell_SOMP{iL,1},[Bdim Bdim],[Idim Idim],'distinct');
            Im_recNISOMP = col2im(cell_NISOMP{iL,1},[Bdim Bdim],[Idim Idim],'distinct');
            Im_recBCS = col2im(cell_BCS{iL,1},[Bdim Bdim],[Idim Idim],'distinct');
            Im_recMTBCS = col2im(cell_MTBCS{iL,1},[Bdim Bdim],[Idim Idim],'distinct');
            Im_recRao = col2im(cell_Rao{iL,1},[Bdim Bdim],[Idim Idim],'distinct');
            Im_recLSTM = col2im(cell_LSTM{iL,1},[Bdim Bdim],[Idim Idim],'distinct');

            v=cat(3,v,Im);
            vY = cat(3,vY,Y);
            vSOMP = cat(3,vSOMP,Im_recSOMP);
            vNISOMP = cat(3,vNISOMP,Im_recNISOMP);
            vBCS = cat(3,vBCS,Im_recBCS);
            vMTBCS = cat(3,vMTBCS,Im_recMTBCS);
            vRao = cat(3,vRao,Im_recRao);
            vLSTM = cat(3,vLSTM,Im_recLSTM);
            
            errSOMP1=sqrt(sum(sum((Im_recSOMP-Im).^2,1),2)/sum(sum(Im.^2,1),2));
            errNISOMP1=sqrt(sum(sum((Im_recNISOMP-Im).^2,1),2)/sum(sum(Im.^2,1),2));
            errBCS1=sqrt(sum(sum((Im_recBCS-Im).^2,1),2)/sum(sum(Im.^2,1),2));
            errMTBCS1=sqrt(sum(sum((Im_recMTBCS-Im).^2,1),2)/sum(sum(Im.^2,1),2));
            errRao1=sqrt(sum(sum((Im_recRao-Im).^2,1),2)/sum(sum(Im.^2,1),2));
            errLSTM1=sqrt(sum(sum((Im_recLSTM-Im).^2,1),2)/sum(sum(Im.^2,1),2));
            
            errSOMPavg = errSOMPavg + errSOMP1;
            errNISOMPavg = errNISOMPavg + errNISOMP1;       
            errBCSavg = errBCSavg + errBCS1;   
            errMTBCSavg = errMTBCSavg + errMTBCS1;   
            errRaoavg = errRaoavg + errRao1;   
            errLSTMavg = errLSTMavg + errLSTM1;
            
            errSOMPavg2 = errSOMPavg2 + errSOMP1;
            errNISOMPavg2 = errNISOMPavg2 + errNISOMP1;       
            errBCSavg2 = errBCSavg2 + errBCS1;   
            errMTBCSavg2 = errMTBCSavg2 + errMTBCS1;   
            errRaoavg2 = errRaoavg2 + errRao1;   
            errLSTMavg2 = errLSTMavg2 + errLSTM1;
        end
        disp( strcat('K = ',num2str(k),', Realization = ',num2str(i33),', MSE of LSTM-CS method = ',num2str(errLSTMavg2/L) ));
    end
    
    errNISOMP(i2,1) = errNISOMPavg / (nImageTest*L);
    errSOMP(i2,1) = errSOMPavg / (nImageTest*L);
    errBCS(i2,1) = errBCSavg / nImageTest;
    errMTBCS(i2,1) = errMTBCSavg / (nImageTest*L);
    errRao(i2,1) = errRaoavg / (nImageTest*L);
    errLSTM(i2,1) = errLSTMavg / (nImageTest*L);

end

% Result representation starts*****************************************

figure;
plot(k_vec,errSOMP,':rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',5); axis('tight');
xlabel('Sparsity');ylabel('Mean of Error');
hold on;
plot(k_vec,errNISOMP,':gs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',5); axis('tight');
plot(k_vec,errBCS,':rs','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','b','MarkerSize',5); axis('tight');
plot(k_vec,errMTBCS,':gs','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',5); axis('tight');
plot(k_vec,errRao,':ks','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',5); axis('tight');
plot(k_vec,errLSTM,':cs','LineWidth',2,'MarkerEdgeColor','c','MarkerFaceColor','c','MarkerSize',5); axis('tight');
legend('SOMP','NWSOMP','BCS','MT-BCS','T-SBL','LSTM-CS',3);
if (NOISE == 0 && Biased == 0)
    title(strcat([' MaxEpoch = ',num2str(maxEpochGeneral),', Reg = ',num2str(Reg),', Hidden units = ',num2str(numhidGeneral)]));
    saveas(gcf,strcat(['Results\Solver_with_Regularization_' num2str(Reg) '_num_' num2str(numhidGeneral) '_nreal_' num2str(nImageTest) '_StepSize_' num2str(StepSize) '_NoBias_n' num2str(n) CodeType '_Error_MultiChannel_Complete.fig']));
elseif (NOISE == 0 && Biased == 1)
    title(strcat([' MaxEpoch = ',num2str(maxEpochGeneral),', Reg = ',num2str(Reg),', Hidden units = ',num2str(numhidGeneral)]));
    saveas(gcf,strcat(['Results\Solver_with_Regularization_' num2str(Reg) '_num_' num2str(numhidGeneral) '_nreal_' num2str(nImageTest) '_StepSize_' num2str(StepSize) '_n' num2str(n) CodeType '_Error_MultiChannel_Complete.fig']));    
elseif (NOISE == 1 && Biased == 1)
    title(strcat([' MaxEpoch = ',num2str(maxEpochGeneral),', Reg = ',num2str(Reg),', Hidden units = ',num2str(numhidGeneral),', Noisy']));
    saveas(gcf,strcat(['Results\Solver_with_Regularization_' num2str(Reg) '_num_' num2str(numhidGeneral) '_nreal_' num2str(nImageTest) '_StepSize_' num2str(StepSize) '_Noisy_n' num2str(n) CodeType '_Error_MultiChannel_Complete.fig']));
elseif (NOISE == 1 && Biased == 0)
    title(strcat([' MaxEpoch = ',num2str(maxEpochGeneral),', Reg = ',num2str(Reg),', Hidden units = ',num2str(numhidGeneral),', Noisy']));
    saveas(gcf,strcat(['Results\Solver_with_Regularization_' num2str(Reg) '_num_' num2str(numhidGeneral) '_nreal_' num2str(nImageTest) '_StepSize_' num2str(StepSize) '_Noisy_NoBias_n' num2str(n) CodeType '_Error_MultiChannel_Complete.fig']));
end
disp('Performance Evaluation: END');