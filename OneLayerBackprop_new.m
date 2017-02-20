function [MSE_Tot, MSE_Tg, MSE_Test, InputWeight,OutputWeight]     = OneLayerBackprop_new(s_ratio_per_7,gamma,stepSize,Data_Train,Target_Train,NumberofHiddenNeurons,C,maxEpoch,NumberofInputNeurons,Winit,Data_Test,Target_Test)

% Dong Yu, Li Deng, "Efficient and effective algorithms for training
% single-hidden-layer neural networks", Pattern Recognition Letters, 
% Volume 33, Issue 5, 1 April 2012, Pages 554-558.


% I have just modified error calculation part to have the same criterion
% for comparison. I have also added one initial random weight as input to 
% this function to initial weights of both functions at the same point.
% Hamid Palangi, March 28. 2012, Email: hamidp@ece.ubc.ca
% July 28, 2012


% Data_Train = [Data_Train;ones(1,size(Data_Train,2))];

global NumberofTrainingData NumberofTestingData

NumberofTrainingData = size(Data_Train,2);
NumberofTestingData = size(Data_Test,2);
InputDim = NumberofInputNeurons;

InputWeight = Winit; 


 %% 


 
H = ones(NumberofHiddenNeurons, NumberofTrainingData);

IWT=InputWeight;

t =1.0; told=t;

%%%%%%Adding backfitting itereations; once for maxEpoch=1 %%%%%%%%%%

fid1 = fopen(strcat(['Results\accuracy_Regularization_' num2str(C) '_num_' num2str(NumberofHiddenNeurons) '_StepSize_' num2str(stepSize) '_NoBias.txt']), 'w');

for epoch=1:maxEpoch
    
    

%     if (mod(epoch,6) ==0  )
% 
%         stepSize= stepSize * s_ratio_per_7;
% 
%     end

% ****************************************************    
    
    H(1:NumberofHiddenNeurons, :) = 1 ./ (1 + exp(-InputWeight*Data_Train));       %always use sigmoid

    %%%%%%%%%%% Calculate output weights OutputWeight (beta_i)by pseudo-inverse

    OutputWeight=(eye(size(H,1))/C + H * H') \ H * Target_Train';                         % faster implementation

   

%figure ; mesh(OutputWeight)  

 

    Output_Train =(H' * OutputWeight)';

   

    MSE_Tg(epoch,1)=sqrt(mse(Target_Train - Output_Train));               %   Calculate training accuracy (RMSE) for regression case

   

    % do not apply softmax to visible data at the output

    OutputWeight_Reconstruct =(eye(size(H,1))/C + H * H') \ H * Data_Train';

    Output_Train_Reconstruct =(H' * OutputWeight_Reconstruct)';

    MSE_Re(epoch,1)=sqrt(mse(Data_Train -  Output_Train_Reconstruct));

    MSE_Tot(epoch,1) = MSE_Tg(epoch,1) + gamma*MSE_Re(epoch,1);

   

    

    %%%%%%%%%%% Calculate the testing accuracy

    H_test = ones(NumberofHiddenNeurons, NumberofTestingData);    

    H_test(1:NumberofHiddenNeurons,:) = 1 ./ (1 + exp(-InputWeight*Data_Test));

    

    

    Output_Test=(H_test' * OutputWeight)'; 
    
    MSE_Test(epoch,1)=sqrt(mse(Target_Test - Output_Test));
% *************************************************************
if epoch==1
        disp(strcat(['Epoch = 0' ', mse_tr = ' num2str(MSE_Tg(epoch,1))...
            ', mse_dev = ' num2str(MSE_Test(epoch,1)) ', initial mse, no training.']));
end


   if maxEpoch > 0,

  
            fprintf(fid1, '%f %f\n',MSE_Tg(epoch,1),MSE_Test(epoch,1));
            
            
       
       
            %backward (replace the old long code for backfitting with the new one:

            DH=H .* (1-H);

            PInvH = H' / (eye(size(H,1))/C+H*H');

            TPInvH=Target_Train*PInvH;

            DW1=((PInvH * (H*Target_Train') * TPInvH - Target_Train'*TPInvH)' .* DH) * Data_Train';

            DW1 = DW1(1:NumberofHiddenNeurons,:);
            norm_grad = norm(DW1);
        disp(strcat(['Epoch = ' num2str(epoch) ', mse_tr = ' num2str(MSE_Tg(epoch,1))...
            ', mse_dev = ' num2str(MSE_Test(epoch,1)) ',norm_grad = ' num2str(norm_grad) '.']));           

             % NEW: add gradient DW2 for the reconstruction MSE term

            if gamma ~= 0

            TPInvH2=Data_Train*PInvH;

            DW2=((PInvH * (H*Data_Train') * TPInvH2 - Data_Train'*TPInvH2)' .* DH) * Data_Train';

            DW2 = DW2(1:NumberofHiddenNeurons,:);

            end

            %%%%%%%%%%%%%%%%%%%%

            IWOld = InputWeight;

          

            if gamma ~= 0

              InputWeight = IWT - stepSize * (DW1 + gamma*DW2);

            else

              InputWeight = IWT - stepSize * DW1 ; 

            end

   

            ttemp = t;

            t = (1+sqrt(1+4*t*t))/2.0;

            IWT = InputWeight + told/t*(InputWeight-IWOld);  %IWT = InputWeight + (told-1)/t*(InputWeight-IWOld);

            told=ttemp;

            IWOld = IWT;

 

    end

   

  

end


clear H;
fclose(fid1);
 

 

end