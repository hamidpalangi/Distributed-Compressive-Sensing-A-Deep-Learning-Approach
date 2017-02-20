%    Goal: Compare T-MSBL, T-SBL, MSBL, MFOCUSS
%          In this demo, you will see T-MSBL and T-SBL outperform other two
%          algorithms. Particularly, please note that T-MSBL and T-SBL
%          adaptively learn the regularization parameter lambda, while MSBL
%          freezes lambda to the true noise variance (which is a near
%          optimal value). When T-MSBL and T-SBL freeze their lambda to the
%          optimal values, their performance can be further better.
%
%          This demo also shows the lambda learning rule of MSBL is very
%          poor. It should not be used in low SNR cases.
%
%  Author: Zhilin Zhang (z4zhang@ucsd.edu)
%    Date: May 20, 2011
% Version: 1.1


clear all;  

% Experiment Assignment
run_TMSBL = 1;           % Run T-MSBL
run_TSBL  = 1;           % Run T-SBL 
run_MSBL  = 1;           % Run MSBL
run_FOCUSS= 1;           % Run M-FOCUSS
iterNum   = 50;          % Trial number (i.e. number of repeating the experiment)
                         % For statistical result, iterNum should not less than 50

% Problem dimension
N = 30;                  % Row number of the dictionary matrix 
M = N * 5;               % Column number of the dictionary matrix
L = 3;                   % Number of measurement vectors
K = 7;                   % Number of nonzero rows (i.e. source number) in the solution matrix
beta = ones(K,1)*0.9;    % Temporal correlation of each nonzero row in the solution matrix.
                         % In biomedical applications, such as EEG, MRI,etc, such temporal 
                         % correlation could be higher than 0.95
                         
                         %====================================================================                         
SNR = 10;                % Note: When you change the SNR, you may need to accordingly change 
                         %       the input arguments for each algorithm
                         %====================================================================
                         
for it = 1 : iterNum
    fprintf('\n\nTrial #%d:\n',it);

    % Generate dictionary matrix with columns draw uniformly from the surface of a unit hypersphere
    Phi = randn(N,M);
    Phi = Phi./(ones(N,1)*sqrt(sum(Phi.^2)));
   
    % Generate the K nonzero rows, each row being an AR(1) process. All the AR(1) 
    % processes have different AR coefficients, which are randomly chosen from [0.7,1)     
    nonzeroW(:,1) = randn(K,1);
    for i = 2 : L*100
        nonzeroW(:,i) = beta .* nonzeroW(:,i-1) + sqrt(1-beta.^2).*(ones(K,1).*randn(K,1));
    end
    nonzeroW = nonzeroW(:,end-L+1:end);   % Ensure the AR processes are stable

    % Normalize each row
    nonzeroW = nonzeroW./( sqrt(sum(nonzeroW.^2,2)) * ones(1,L) );
    
    % Rescale each row such that the squared row-norm distributes in [1,scalefactor]
    scalefactor = 3; 
    mag = rand(1,K); mag = mag - min(mag);
    mag = mag/(max(mag))*(scalefactor-1) + 1;
    nonzeroW = diag(sqrt(mag)) * nonzeroW;

    % Locations of nonzero rows are randomly chosen
    ind = randperm(M);
    indice = ind(1:K);
    Wgen = zeros(M,L);
    Wgen(indice,:) = nonzeroW;

    % Noiseless signal
    signal = Phi * Wgen;

    % Observation noise   
    stdnoise = std(reshape(signal,N*L,1))*10^(-SNR/20);
    noise = randn(N,L) * stdnoise;

    % Noisy signal
    Y = signal + noise;


    %=================== T-MSBL (adaptively learn the lambda) ====================
    if run_TMSBL == 1,
    tic;
    
    % Depends on the SNR, choosing suitable values for input arguments:
    % If no noise,            Weight = TMSBL(Phi, Y, 'noise','no');
    % If SNR >= 23 dB,        Weight = TMSBL(Phi, Y, 'noise','small');
    % If 6dB < SNR <= 22 dB,  Weight = TMSBL(Phi, Y, 'noise','mild');
    % If SNR <= 6 dB,         Weight = TMSBL(Phi, Y, 'noise','large');
    % Note that you need not to know exactly the SNR value. Just roughly estimate it. 
    % For example, when the true SNR is 6 dB, either set 'noise'='mild' or
    % 'noise'='large' has the similar performance.
    %
    % Note: See the TMSBL code for details on input arguments ans set your
    % own values for specific problems.
    
    Weight1 = TMSBL(Phi, Y, 'noise','mild');
    
    time1 = toc;
    TIME1(it) = time1;


    % failure rate: F1 = 1: perfect recovery; F1 = 0: worst recovery
    F1 = perfSupp(Weight1,indice,'firstlargest', K);               
    fail_TMSBL(it) = (F1~=1);                              
    
    % Mean Square Error (MSE)
    mse_TMSBL(it) = (norm(Wgen - Weight1,'fro')/norm(Wgen,'fro'))^2;  
    
    fprintf('  T-MSBL(learn lambda): time = %5.2f; Findex = %3.2f, Ave-MSE = %3.2f%%; Ave-Time = %4.3f\n',...
        time1,F1,mean(mse_TMSBL)*100,mean(TIME1));
    end
    %====================================================================

    
    
    %======================= T-SBL (adaptively learn the lambda)==========
    if run_TSBL == 1,
    % Similar to TMSBL, see the code for details about the input argument
    tic;
    Weight2 = TSBL(Phi, Y, 'SNR','low');
    time2 = toc;
    TIME2(it) = time2;


    % failure rate: F2 = 1: perfect recovery; F2 = 0: worst recovery
    F2 = perfSupp(Weight2,indice,'firstlargest', K);               
    fail_TSBL(it) = (F2~=1);                              
    
    % Mean Square Error (MSE)
    mse_TSBL(it) = (norm(Wgen - Weight2,'fro')/norm(Wgen,'fro'))^2;  
    
    fprintf('   T-SBL(learn lambda): time = %5.2f; Findex = %3.2f, Ave-MSE = %3.2f%%; Ave-Time = %4.3f\n',...
        time2,F2,mean(mse_TSBL)*100,mean(TIME2));
    end
    
    
    
    
    %================== MSBL (adaptively learn the lambda) ======================= 
    lambda = 1e-2;                 % Initial value for the regularization parameter. 
    Learn_Lambda = 1;              % Using its lambda learning rule

    if run_MSBL == 1,
    tic;
    Weight3 = MSBL(Phi,Y, lambda, Learn_Lambda);
    time3 = toc;
    TIME3(it) = time3;
    
    
    % Failure rate
    F3 = perfSupp(Weight3,indice,'firstlargest', K);      
    fail_MSBL(it) = (F3~=1);      
    
    % MSE
    perf_MSBL(it) = (norm(Wgen - Weight3,'fro')/norm(Wgen,'fro'))^2;  
    
    fprintf('    MSBL(learn lambda): time = %5.2f; Findex = %3.2f, Ave-MSE = %3.2f%%; Ave-Time = %4.3f\n',...
        time3,F3,mean(perf_MSBL)*100,mean(TIME3));
    
    
    % now MSBL using the true noise variance as the lambda value and freeze it
    lambda0 = stdnoise^2;   
    Learn_Lambda0 = 0;     % do not use its lambda learning rule
    tic;
    Weight33 = MSBL(Phi,Y, lambda0, Learn_Lambda0);
    time33 = toc;
    TIME33(it) = time33;
    
    
    % Failure rate
    F33 = perfSupp(Weight33,indice,'firstlargest', K);      
    fail_MSBL33(it) = (F33~=1);      
    
    % MSE
    perf_MSBL33(it) = (norm(Wgen - Weight33,'fro')/norm(Wgen,'fro'))^2;  
    
    fprintf('MSBL(lambda=noise var): time = %5.2f; Findex = %3.2f, Ave-MSE = %3.2f%%; Ave-Time = %4.3f\n',...
        time33,F33,mean(perf_MSBL33)*100,mean(TIME33));
    
   
    end
    
    
    % ====================== MFOCUSS with near-optimal regularization =========
    if run_FOCUSS == 1,
    tic;
    lambda_opt = stdnoise^2;        % use the noise variance
    Weight4 = MFOCUSS(Phi, Y, lambda_opt);
    
    time4 = toc;
    TIME4(it) = time4;
    
    
    % Failure rate
    F4 = perfSupp(Weight4,indice,'firstlargest', K);      
    fail_MFOCUSS(it) = (F4~=1);      
    
    % MSE
    perf_MFOCUSS(it) = (norm(Wgen - Weight4,'fro')/norm(Wgen,'fro'))^2;  
    
    fprintf('MFOCUSS(lambda=noise var): time = %5.2f; Findex = %3.2f, Ave-MSE = %3.2f%%; Ave-Time = %4.3f\n',...
        time4,F4,mean(perf_MFOCUSS)*100,mean(TIME4));
    end
    
end




