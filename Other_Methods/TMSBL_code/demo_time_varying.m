% Show how to use T-MSBL in the time-varying sparsity model.
% To show the results by KF-CS and LS-CS, you need to download the codes
% from: http://home.engineering.iastate.edu/~namrata/research/LSCS_KFCS_code.zip
% 
% However, you can see the comparison in the following reference:
%
% Z. Zhang, B. D. Rao, Exploiting Correlation in Sparse Signal Recovery Problems: 
% Multiple Measurement Vectors, Block Sparsity, and Time-Varying Sparsity
% ICML 2011 Workshop on Structured Sparsity: Learning and Inference
%
% Or similar comparison in my blogs:
%
% http://marchonscience.blogspot.com/2011/04/is-mmv-more-suitable-for-dynamic.html
% http://marchonscience.blogspot.com/2011/04/is-mmv-more-suitable-for-dynamic_17.html
% http://marchonscience.blogspot.com/2011/04/is-mmv-more-suitable-for-dynamic_18.html
% 
% Feel free to send any questions to me: z4zhang@ucsd.edu
% May 19, 2010

clear; clc; close all;


L = 50;          % snapshot number
M = 256;         % column number of the dictionary matrix
N = 60;          % row number of the dictionary matrix
K = 15;          % number of the nonzero coefficients at the initial stage
addNZ = 10;      % added number of nonzero coefficients at each addition time
segLen = 20;     % duration of each added coefficient
remNZ = 5;       % removed number of nonzero coefficients at each deletion time

iteration = 20;   % independently repeat the experiments to see the averaged results

for it = 1 : iteration
    fprintf('Trial #%d\n',it);
    
    % generate the dictionary matrix from Gaussian distribution
    Phi = randn(N,M);
    Phi = Phi./(ones(N,1)*sqrt(sum(Phi.^2)));

    % generate the K time series existing during the whole time
    beta = 1- rand(K,1)*0.3;           % randomly choose temporal correlation of 
                                       % time series of each coefficient
    longS(:,1) = randn(K,1);
    for i = 2 : L*20
        longS(:,i) = beta .* longS(:,i-1) + sqrt(1-beta.^2).*(randn(K,1));
    end
    longS = longS(:,end-L+1:end);      % stable time series
    longS = longS./( sqrt(sum(longS.^2,2)) * ones(1,L) );
    
    % generate time series of coefficients
    D = addNZ * 3;   % possible largest number of nonzero coefficients
    beta = 1- rand(D,1)*0.3;           % randomly choose temporal correlation of 
                                       % time series of each coefficient
    pool(:,1) = randn(D,1);
    for i = 2 : segLen*20
        pool(:,i) = beta .* pool(:,i-1) + sqrt(1-beta.^2).*(randn(D,1));
    end
    pool = pool(:,end-segLen+1:end);  % stable time series
    pool = pool./( sqrt(sum(pool.^2,2)) * ones(1,segLen) );
    
    % initial stage of a time series should have increasing magnitude
    pool(:,[1,2,3]) = pool(:,[1,2,3]) .* (ones(D,1) * [0.2, 0.6, 1]);
    
    % when a time series disappears, the magnitude should be decreasing
    pool(:,[segLen-2,segLen-1,segLen]) = pool(:,[segLen-2,segLen-1,segLen]).*(ones(D,1)*[1,0.6,0.2]);
    
    % select K time series at L=1, 
    ind = randperm(M);
    indice0 = ind(1:K);
    Wgen = zeros(M,L);
    Wgen(indice0,:) = longS;
    
    % select another addNZ time series at L = 16
    restLoc = setdiff([1:M],indice0);
    ind = randperm(length(restLoc));
    indice1 = restLoc(ind(1:addNZ));
    Wgen(indice1,[16:15+segLen]) = pool(1:addNZ,:);
    
    % delete existing time series at L=26
    candidateLoc = union(indice0,indice1);
    ind_rem = randperm(K+addNZ);
    indice_rem = candidateLoc(ind_rem(1:remNZ));
    Wgen(indice_rem,[24:end]) = Wgen(indice_rem,[24:end]).*(ones(remNZ,1)*[0.8,0.4,zeros(1,25)]);
    
    % select another addNZ time series at L = 31
    restLoc2 = setdiff([1:M],union(indice0,indice1));
    ind2 = randperm(length(restLoc2));
    indice2 = restLoc2(ind2(1:addNZ));
    Wgen(indice2,[31:30+segLen]) = pool(addNZ+1:addNZ*2,:);
    
%     % See the pictures of the generated signals
%     imagesc(Wgen); colorbar;
    
    % noiseless signal
    signal = Phi * Wgen;

    % observation noise
    stdnoise = 0.01;
    noise = randn(N,L) * stdnoise;

    % noisy signal
    Y = signal + noise;

    % compute total SNR
    SNR_rec(it) = 20*log10(norm(signal,'fro')/norm(noise,'fro'));
    
    
    % ===================Run T-MSBL step by 5 =================
    tic;
    X_tsbl5 = [];
    for i=1:10
        
    % According to the SNR range, choose suitable input arguments
    % See the codes for details
    [X_tsbl1] = TMSBL(Phi, Y(:,[(i-1)*5+1:i*5]), 'noise','small');
    
    
    X_tsbl5 = [X_tsbl5,X_tsbl1];
    end
    TIME_tsbl5(it) = toc;

    for t = 1:L
        err_tsbl5(it,t) = norm( Wgen(:,t)-X_tsbl5(:,t) )^2;
        energy(it,t) = norm( Wgen(:,t) )^2;
    end
    

    % =================== Run T-MSBL step by 2 =================
    tic;
    X_tsbl2 = [];
    for i=1:25
    % According to the SNR range, choose suitable input arguments
    % See the codes for details
    [X_tsbl1] =  TMSBL(Phi, Y(:,[(i-1)*2+1:i*2]), 'noise','small');
    
    X_tsbl2 = [X_tsbl2,X_tsbl1];
    end
    TIME_tsbl2(it) = toc;

    for t = 1:L
        err_tsbl2(it,t) = norm( Wgen(:,t)-X_tsbl2(:,t) )^2;
    end
    
    
    
    % ===================Run MSBL steps by 5=================
    lambda = std(Y(:,1))*1e-2;
    Learn_Lambda = 1;
    
    tic;
    X_msbl5 = [];
    for i = 1 : 10
    [X_msbl1,gamma2,gamma_used2,count2] = MSBL(Phi, Y(:,[(i-1)*5+1:i*5]), lambda, Learn_Lambda);
    
    X_msbl5 = [X_msbl5,X_msbl1];
    end

    TIME_msbl5(it) = toc;

    for t = 1:L
        err_msbl5(it,t) = norm( Wgen(:,t)-X_msbl5(:,t) )^2;
    end
    
    % ===================Run MSBL2 steps by 2=================
    lambda = std(Y(:,1))*1e-2;
    Learn_Lambda = 1;
    tic;
    X_msbl2 = [];
    for i = 1 : 25
    [X_msbl1,gamma2,gamma_used2,count2] = MSBL(Phi, Y(:,[(i-1)*2+1:i*2]), lambda, Learn_Lambda);
    X_msbl2 = [X_msbl2,X_msbl1];
    end

    TIME_msbl2(it) = toc;

    for t = 1:L
        err_msbl2(it,t) = norm( Wgen(:,t)-X_msbl2(:,t) )^2;
    end
    
    
%     % ===================Run KF-CS ===============
%     tot = 50; n = N; m = M; lambdap = 0.1;
%     global n
%     global m
%     global tot
%     global lambdap
%     global tot
%     Pi0 = 0.8 * eye(M);
%     Q = 0.8 * eye(M);
%     R = stdnoise^2 * eye(N);
%     tic;
%     [X_kfcs,T_hat] = kfcs_full(Y,Pi0,Phi,Q,R);
%     TIME_kfcs(it) = toc;
%     
%     for t = 1:L
%         err_kfcs(it,t) = norm( Wgen(:,t)-X_kfcs(:,t) )^2;
%     end
%
%     %========================== LS-CS =====================
%     tic;
%     [x_upd_lscs,T_hat_lscs,x_upd_csres] = lscs_full(Y,Pi0,Phi,Q,R);
%     TIME_LSCS(it) = toc;
%     for t = 1:tot
%         err_lscs(it,t) = norm( Wgen(:,t)-x_upd_lscs(:,t) )^2;
%     end    

      save data_demo_timevarying;
end

figure(1);
semilogy(mean(err_tsbl2,1)./mean(energy,1),'r','linewidth',2.5);
hold on;semilogy(mean(err_tsbl5,1)./mean(energy,1),'r-.','linewidth',2.5);
hold on;semilogy(mean(err_msbl2,1)./mean(energy,1),'linewidth',2.5);
hold on;semilogy(mean(err_msbl5,1)./mean(energy,1),'-.','linewidth',2.5);
% hold on;semilogy(mean(err_kfcs,1)./mean(energy,1),'k','linewidth',2.5);
% hold on;semilogy(mean(err_lscs,1)./mean(energy,1),'c','linewidth',2.5);
legend('T-MSBL steps by 2','T-MSBL steps by 5','M-SBL steps by 2','M-SBL steps by 5');
ylabel('\bf\fontsize{20}Normalized MSE');
xlabel('\bf\fontsize{20}Time');


