% This demo produces the Fig.6 when SNR = 10dB and T-MSBL and MSBL use
% their lambda lerning rules to learn lambda during learning procedures

clear;

SNR  = 10;                % SNR
N    = 25;                % row number of the dictionary matrix
M    = 125;               % column number of the dictionary matrix
K    = 7;                 % source number
L    = 4;                 % number of measurement vectors
beta = ones(K,1)*(0.8);   % temporal correlation of each source

for it = 1:100             % repeat 100 times for this demo
    
    fprintf('\nTrial #%d: \n',it);
    
    % dictionary matrix with columns draw uniformly from the surface of a unit hypersphere
    Phi = randn(N,M);
    Phi = Phi./(ones(N,1)*sqrt(sum(Phi.^2)));

    % Generate L source vectors
    nonzeroW(:,1) = randn(K,1);
    alpha = ones(K,1);
    for i = 2 : L*100
        nonzeroW(:,i) = beta .* nonzeroW(:,i-1) + sqrt(1-beta.^2).*(sqrt(alpha).*randn(K,1));
    end
    nonzeroW = nonzeroW(:,end-L+1:end);
    
    % normalize along row
    nonzeroW = nonzeroW./( sqrt(sum(nonzeroW.^2,2)) * ones(1,L) );
    
    % select active sources at random locations
    ind = randperm(M);
    indice = ind(1:K);
    Wgen = zeros(M,L);
    Wgen(indice,:) = nonzeroW;


    % noiseless signal
    signal = Phi * Wgen;

    
    % observation noise
    stdnoise = std(reshape(signal,N*L,1))*10^(-SNR/20);
    noise = randn(N,L) * stdnoise;
    
    
    % noisy signal
    Y = signal + noise;
    
    

    %============================ T-MSBL ==========================
    [Weight3, gamma_ind3, gamma_est3, count3] = TMSBL(Phi, Y, 'noise','mild');
    
    [F3,P3,R3,IND3] = perfSupp(Weight3,indice,'firstlargest',K);
    fail_TMSBL(it) = (F3~=1);
    mse_TMSBL(it) = (norm(Wgen - Weight3,'fro')/norm(Wgen,'fro'))^2; 

    fprintf('  TMSBL: Mean MSE = %4.3f; Mean FR = %4.3f; \n',mean(mse_TMSBL), mean(fail_TMSBL));
    
   
    %============================ MSBL ==========================
    lambda_ini = 1e-2;
    [Weight1,gamma1,gamma_used1,count1] = MSBL(Phi, Y, lambda_ini, 1);

    [F1,P1,R1,IND1] = perfSupp(Weight1,indice,'firstlargest',K);
    fail_MSBL(it) = (F1~=1);
    mse_MSBL(it) = (norm(Wgen - Weight1,'fro')/norm(Wgen,'fro'))^2; 
        
    fprintf('   MSBL: Mean MSE = %4.3f; Mean FR = %4.3f; \n', mean(mse_MSBL), mean(fail_MSBL));
    



end