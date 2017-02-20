% This demo produces the Fig.8 in the paper.
% Enjoy increasing the temporal correlation to see the performance of T-MSBL

clear;

double tmpCorr;

% Set the temporal correlation value. You can add more "9"s to it (i.e. allow
% the temporal correlation approximates 1 closer. But note that,
% practically, when tmpCorr is larger than a value, the matrix computation
% will have numerical problems. 
% In this demo, it is advised that |tmpCorr| <= 1e-13.
tmpCorr = 0.999999999;


% Problem dimension
K = 12;                   % source number
L = 3;                    % number of measurement vectors
N = 40;                   % row number of the dictionary matrix
M = 128;                  % column number of the dictionary matrix
iterNum = 100;            % number of repeating the experiment

% generate Hadamard matrix of the size 128 x 128
H = hadamard(M);


for it = 1:iterNum
    fprintf('\n\nTrial #%d:\n',it);

    % generate the dictionary matrix, whose 40 rows are randomly chosen
    % from the rows of the Hadamard matrix
    rowLoc = randperm(M);
    loc = rowLoc(1:N);
    Phi = H(loc,:);
    
    % Generate L source vectors
    beta = ones(K,1)*tmpCorr;  
    nonzeroW(:,1) = randn(K,1);
    alpha = ones(K,1);
    for i = 2 : L
        nonzeroW(:,i) = beta .* nonzeroW(:,i-1) + sqrt(1-beta.^2).*(sqrt(alpha).*randn(K,1));
    end

    % rescale the rows
    nonzeroW = nonzeroW./( sqrt(sum(nonzeroW.^2,2)) * ones(1,L) );
    AdjustNorm = 0.3+rand(K,1)*(1-0.3); 
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
    
    % Failure Rate
    F1 = perfSupp(Weight1,indice,'firstlargest',K);
    FR_TMSBL(it) = (F1~=1);
    
    % MSE
    MSE1 = (norm(Weight1-Wgen,'fro')/norm(Wgen,'fro'))^2;
    MSE_TMSBL(it) = MSE1;

    fprintf('T-MSBL(Using 3 Measurement Vectors): Fail_Rate = %4.3f%%; MSE = %3.2f%%; \n',mean(FR_TMSBL)*100,mean(MSE_TMSBL)*100);

    
    
    %============================ 3.MSBL for MMV ==========================
    lambda1 = 1e-15;
    Learn_Lambda = 0;
    [Weight3,gamma3,gamma_used3,count3] = MSBL(Phi, Y, lambda1, Learn_Lambda);

    % Failure Rate
    F3 = perfSupp(Weight3,indice,'firstlargest',K);
    FR_MSBL(it) = (F3~=1);
    
    % MSE
    MSE3 = (norm(Weight3-Wgen,'fro')/norm(Wgen,'fro'))^2;
    MSE_MSBL(it) = MSE3;

    fprintf('  MSBL(Using 3 Measurement Vectors): Fail_Rate = %4.3f%%; MSE = %3.2f%%; \n',mean(FR_MSBL)*100,mean(MSE_MSBL)*100);

    
    %============================ 4.MSBL for SMV ==========================
    [Weight4,gamma4,gamma_used4,count4] = MSBL(Phi, Y(:,1), lambda1, Learn_Lambda);

    % Failure Rate
    F4 = perfSupp(Weight4,indice,'firstlargest',K);
    FR_SBL(it) = (F4~=1);
    
    % MSE
    MSE4 = (norm(Weight4-Wgen(:,1),'fro')/norm(Wgen(:,1),'fro'))^2;
    MSE_SBL(it) = MSE4;
    
    fprintf('  MSBL(Using 1 Measurement Vector) : Fail_Rate = %4.3f%%; MSE = %3.2f%%; \n',mean(FR_SBL)*100,mean(MSE_SBL)*100);
end

