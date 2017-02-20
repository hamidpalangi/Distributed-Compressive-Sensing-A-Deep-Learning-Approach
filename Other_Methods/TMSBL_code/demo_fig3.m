%    Goal: This demo produces the Fig.3(d) when source number is 14

clear;

% Experiment variables (will be varied to produce each sub-figures)
% Here we produce Fig.3(d) when source nubmer is 14
K = 14;                 % soure number
beta = ones(K,1)*0.99;  % temporal correlation of each source

% problem dimension
N = 25;                 % column number of the dictionary matrix
M = 125;                % row number of the dictionary matrix
L = 4;                  % number of measurement vectors
iterNum = 50;           % In the paper we repeated 1000 times. But here we repeat 50 times to save time

for it = 1:iterNum
    fprintf('\n\nTrial #%d:\n',it);

    % Dictionary matrix with columns draw uniformly from the surface of a unit hypersphere
    Phi = randn(N,M);
    Phi = Phi./(ones(N,1)*sqrt(sum(Phi.^2)));

    % Generate L source vectors
    nonzeroW(:,1) = randn(K,1);
    alpha = ones(K,1);
    for i = 2 : L*20
        nonzeroW(:,i) = beta .* nonzeroW(:,i-1) + sqrt(1-beta.^2).*(sqrt(alpha).*randn(K,1));
    end
    % get stable AR signals with length L
    nonzeroW = nonzeroW(:,end-L+1:end);   
    
    % normalize along row
    nonzeroW = nonzeroW./( sqrt(sum(nonzeroW.^2,2)) * ones(1,L) );   
    
    % rescale rows such that their norms uniformly distribute in [1/3, 1]
    AdjustNorm = 1/3+rand(K,1)*(1-1/3); 
    nonzeroW = diag(AdjustNorm) * nonzeroW;
    
    % select active sources at random locations
    ind = randperm(M);
    indice = ind(1:K);
    Wgen = zeros(M,L);
    Wgen(indice,:) = nonzeroW;

    % noiseless signal
    Y = Phi * Wgen;


    %============================ 1.T-MSBL ==========================
    [Weight1, gamma_ind1, gamma_est1, count1] = TMSBL(Phi, Y, 'noise','no');
    
    F1 = perfSupp(Weight1,indice,'firstlargest',K);
    FR_TMSBL(it) = (F1~=1);
    
    MSE1 = (norm(Weight1-Wgen,'fro')/norm(Wgen,'fro'))^2;
    MSE_TMSBL(it) = MSE1;

    fprintf('    T-MSBL: MSE = %3.2f%%; Fail_Rate = %4.3f%%; \n',mean(MSE_TMSBL)*100,mean(FR_TMSBL)*100);


    %============================ 2.T-SBL ==========================      
    [Weight2, gamma_ind2, gamma_est2, count2, B2] = TSBL(Phi, Y, 'SNR','inf');

    F2 = perfSupp(Weight2,indice,'firstlargest',K);
    FR_TSBL(it) = (F2~=1);
    
    MSE2 = (norm(Weight2-Wgen,'fro')/norm(Wgen,'fro'))^2;
    MSE_TSBL(it) = MSE2;

    fprintf('     T-SBL: MSE = %3.2f%%; Fail_Rate = %4.3f%%; \n',mean(MSE_TSBL)*100,mean(FR_TSBL)*100);
    
    

    %============================ 3.MSBL ==========================
    lambda1 = 1e-10;
    Learn_Lambda = 0;
    [Weight3,gamma3,gamma_used3,count3] = MSBL(Phi, Y, lambda1, Learn_Lambda);

    F3 = perfSupp(Weight3,indice,'firstlargest',K);
    FR_MSBL(it) = (F3~=1);
    
    MSE3 = (norm(Weight3-Wgen,'fro')/norm(Wgen,'fro'))^2;
    MSE_MSBL(it) = MSE3;

    fprintf('     M-SBL: MSE = %3.2f%%; Fail_Rate = %4.3f%%; \n',mean(MSE_MSBL)*100,mean(FR_MSBL)*100);
    


    %============================ 4.MFOCUSS ===========================
    [Weight4, gamma_ind4, gamma_est4, count4] = MFOCUSS(Phi, Y, lambda1);
 
    F4 = perfSupp(Weight4,indice,'firstlargest',K);
    FR_FOCUSS(it) = (F4~=1);
    
    MSE4 = (norm(Weight4-Wgen,'fro')/norm(Wgen,'fro'))^2;
    MSE_FOCUSS(it) = MSE4;

    fprintf('    FOCUSS: MSE = %3.2f%%; Fail_Rate = %4.3f%%; \n',mean(MSE_FOCUSS)*100,mean(FR_FOCUSS)*100);
    
 
end