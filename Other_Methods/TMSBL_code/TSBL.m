function [X, gamma_ind, gamma_est, count, B] = TSBL(Phi, Y, varargin)
% Sparse Bayesian Learning for the MMV model considering temporal correlation among source vectors.
% For fast implementation, see TMSBL.m
%
% Command Format:
% (1) For low-SNR case       (<=13dB):  [X, gamma_ind, gamma_est, count, B] = TSBL(Phi, Y)
% (2) For low-SNR case       (<=13dB):  [X, gamma_ind, gamma_est, count, B] = TSBL(Phi, Y, 'SNR','low')
%     For high-SNR case      (>13dB) :  [X, gamma_ind, gamma_est, count, B] = TSBL(Phi, Y, 'SNR','high')
%     For noiseless case             :  [X, gamma_ind, gamma_est, count, B] = TSBL(Phi, Y, 'SNR','inf')
%   [*** Just roughly guess SNR and then choose the above command for a good result (not necessarily optimal) ***]
% (3) Choose your own values for the parameters; for examples:
%     [X,gamma_ind,gamma_est,count,B] ...
%        = TSBL(Phi,Y, 'Prune_Gamma', 1e-3,   'lambda', 0.015,    'Learn_Lambda',1, ...
%                      'Enhance_Lambda',1,    'MAX_ITERS', 500,   'EPSILON', 1e-8,  'PRINT', 1);
%     [X,gamma_ind,gamma_est,count,B] = TSBL(Phi,Y, 'prune_gamma', 1e-4);
%    
% ============================== INPUTS ============================== 
%   Phi         : N X M dictionary matrix
%
%   Y           : N X L measurement matrix, i.e. Y = Phi * X + V.
%
%   'noise'     : If 'SNR'   = 'inf'   : use default values of parameters for noiseless cases
%                 If 'SNR'   = 'high'  : use default values for the cases
%                                        when noise is small (e.g. SNR > 13dB)
%                 If 'SNR'   = 'low'   : use default values for the cases
%                                        when noise is strong (e.g. SNR <= 13dB).
%                                        But if you have some knowledge about T-SBL,
%                                        I suggest you to select your own values
%                                        for input arguments.                                   
%    **** Note: if you use the input argument 'SNR', then all input arguments are invalid, except to: ****
%             'MAX_ITERS', 'EPSILON', 'PRINT' 
%    ****  This input argument choose a set of suitable parameter values for the four cases.  ****
%    ****  But the used parameter values may not be optimal. *****
%
%  'PRUNE_GAMMA'  : Threshold for prunning small hyperparameters gamma_i.
%                   In lightly noisy cases, you can set MIN_GAMMA = 1e-3 or 1e-4.
%                   In strongly noisy cases, set MIN_GAMMA = 1e-2 for better performance.
%                   [ Default value: MIN_GAMMA = 1e-2 ]
%
%   'lambda'    : Initial value for the regularization parameter. Note that setting lambda being
%                 noise variance only leads to sub-optimal performance. 
%                 For T-SBL, the optimal value is generally close to the noise variance.
%                 However, in most cases you can use the lambda learning
%                 rule to learn a near-optimal value, if you set Learn_Lambda = 1
%                 (1) In the noiseless cases, you can simply set lambda = 1e-10 or any other 
%                     very small values (e.g. 1e-9, 1e-11 are also okay),
%                     and run the algorithm without using the lambda
%                     learning rule (i.e. setting Learn_Lambda = 0).  
%                 (2) When SNR >= 5dB, you can use the lambda learning rule 
%                     to learn lambda. In this case, just roughly input an initial value for lambda, 
%                     and set Learn_Lambda = 1.
%                 (3) When SNR < 5dB, the lambda learning rule is not robust. So, you may need 
%                     to use cross-validation or other methods to estimate it. 
%                     In this case, set Learn_Lambda = 0, and try your best to get an
%                     optimal value for lambda as the input. You can
%                     roughly estimate the noise variance as the lambda value.              
%                 [ Default value: lambda = 1e-3 ]
%
% 'Learn_Lambda': (1) If Learn_Lambda = 1, use the lambda learning rule (thus the input lambda
%                     is just as initial value). But note the learning rule is not very good
%                     if SNR <= 5 dB (but still lead to better performance than some other algorithms)
%                 (2) If Learn_Lambda = 0, do not use the lambda learning rule, but use the input 
%                     lambda as its final value.
%                 [ Default value: Learn_Lambda = 1 ]
%
% 'Enhance_Lambda': (1) In low-SNR case (<=13dB), set Enhance_Lambda = 1 for better performance
%                   (2) In other cases, set Enhance_Lambda = 0
%                 [ Default value: Enhance_Lambda = 1 ]
%
%  'MAX_ITERS'    : Maximum number of iterations.
%                 [ Default value: MAX_ITERS = 1000 ]
%
%  'EPSILON'      : Threshold to stop the whole algorithm. 
%                 [ Default value: EPSILON = 1e-8   ]
%
%  'PRINT'        : Display flag. If = 1: show output; If = 0: supress output
%                 [ Default value: PRINT = 0        ]
%
% ==============================  OUTPUTS ============================== 
%   X            : The estimated solution matrix(size: M X L)
%   gamma_ind    : Indexes of nonzero rows in the solution matrix
%   gamma_est    : M X 1 vector of gamma_i values
%   B            : Estimated matrix B
%   count        : Number of iterations used
%
% ============= See Also ================================================
%   TMSBL     ARSBL     MSBL    tMFOCUSS
%
% ============= Reference ===============================================
%   [1] Zhilin Zhang, Bhaskar D. Rao, Sparse Signal Recovery with 
%   Temporally Correlated Source Vectors Using Sparse Bayesian Learning, 
%   IEEE Journal of Selected Topics in Signal Processing, 
%   Special Issue on Adaptive Sparse Representation of
%   Data and Applications in Signal and Image Processing, 2011
%
%
% ============= Author =============
%   Zhilin Zhang (z4zhang@ucsd.edu)
%
% ============= Version =============
%   1.0 (07/31/2010)
%   1.1 (09/12/2010)
%   1.2 (03/05/2011)
%   1.3 (05/20/2011) modify the lambda rule
%   1.4 (05/21/2011) modify the whole




% Dimension of the Problem
[N M] = size(Phi); 
[N L] = size(Y);  

% Default Parameter Values for Any Cases
EPSILON       = 1e-8;       % threshold for stopping iteration. 
MAX_ITERS     = 2000;       % maximum iterations
PRINT         = 0;          % don't show progress information


% Default Parameter Values (suitable for most noisy cases,  SNR <= 13 dB)
STATUS        = -1;         % use the default parameter values for mild noisy cases
PRUNE_GAMMA   = 1e-2;       % threshold for prunning small hyperparameters gamma_i
lambda        = 1e-3;       % initial value for lambda
LearnLambda   = 1;          % use the lambda learning rule to estimate lambda
EnhanceLambda = 1;          % use the enhance strategy to the lambda learning rule 


if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'snr'
                if strcmp(lower(varargin{i+1}),'inf'), STATUS = 0;
                elseif strcmp(lower(varargin{i+1}),'high'), STATUS = 1;
                elseif strcmp(lower(varargin{i+1}),'low'), STATUS = 2;
                else  error(['Unrecognized Value for Input Argument ''SNR''']);
                end
            case 'prune_gamma'
                PRUNE_GAMMA = varargin{i+1}; 
            case 'lambda'
                lambda = varargin{i+1}; 
            case 'learn_lambda'
                LearnLambda = varargin{i+1}; 
                if LearnLambda ~= 1 & LearnLambda ~= 0
                    error(['Unrecognized Value for Input Argument ''Learn_Lambda''']);
                end
            case 'enhance_lambda'
                EnhanceLambda = varargin{i+1}; 
                if EnhanceLambda ~= 1 & EnhanceLambda ~= 0
                    error(['Unrecognized Value for Input Argument ''Enhance_Lambda''']);
                end  
            case 'epsilon'   
                EPSILON = varargin{i+1}; 
            case 'print'    
                PRINT = varargin{i+1}; 
            case 'max_iters'
                MAX_ITERS = varargin{i+1};  
            otherwise
                error(['Unrecognized parameter: ''' varargin{i} '''']);
        end
    end
end


if STATUS == 0  
    % default values for noiseless case
    PRUNE_GAMMA   = 1e-4;      % threshold to prune out small gamma_i
    lambda        = 1e-10;     % fix lambda to this value, and,
    LearnLambda   = 0;         % do not use lambda learning rule, and,
    EnhanceLambda = 0;         % do not use the enhance strategy
    
elseif STATUS == 1 
    % default values for high-SNR cases, e.g. SNR > 13 dB
    PRUNE_GAMMA   = 1e-3;      % threshold to prune out small gamma_i
    lambda        = 1e-3;      % initial value for lambda and,
    LearnLambda   = 1;         % use the lambda learning rule, and,
    EnhanceLambda = 0;         % do not use the enhance strategy
    
elseif STATUS == 2
    % Default values for low-SNR cases, e.g. SNR <= 13 dB
    PRUNE_GAMMA   = 1e-2;       % threshold for prunning small hyperparameters gamma_i
    lambda        = 1e-2;       % initial value for lambda, and,
    LearnLambda   = 1;          % use the lambda learning rule to estimate lambda
    EnhanceLambda = 1;          % use the enhance strategy to the lambda learning rule for low SNR cases

end


if PRINT
    fprintf('\n====================================================\n');
    fprintf('   Running T-SBL...Information about parameters...\n');
    fprintf('====================================================\n');
    if STATUS ~= -1,
        fprintf('You Select the Global Input Argument ''Noise''. Other Arguments May be Invalid\n');
    end
    fprintf('PRUNE_GAMMA  : %e\n',PRUNE_GAMMA);
    fprintf('lambda       : %e\n',lambda);
    fprintf('LearnLambda  : %d\n',LearnLambda);    
    fprintf('EnhanceLambda: %d\n',EnhanceLambda);
    fprintf('EPSILON      : %e\n',EPSILON);
    fprintf('MAX_ITERS    : %d\n\n',MAX_ITERS);
end



% Transfer the multiple measurements into single measurement by vectorization
y = reshape(Y',N*L,1);       
D = kron(Phi,eye(L));

% Initialization 
Sigma0 = repmat(eye(L),[1 1 M]);
gamma = ones(M,1);
keep_list = [1:M]';
usedNum = length(keep_list);
mu_x = zeros(M*L,1);
count = 0;

% Iteration
while (1)
    count = count + 1;

    %=========== Prune weights as their hyperparameters go to zero==============

    if (min(gamma) < PRUNE_GAMMA)
        index = find(gamma > PRUNE_GAMMA);
        usedNum = length(index);
        
        % prune gamma and associated components in Sigma0, Phi
        gamma = gamma(index);  
        Sigma0 = Sigma0(:,:,index);
        Phi = Phi(:,index);
        D = kron(Phi,eye(L));
       
        % post-processing
        keep_list = keep_list(index);
    end
    
    %=================== Compute new weights =================
    mu_old = mu_x;
    
    DBD = zeros(N*L);
    for i = 1 : usedNum
        DBD = DBD + D(:, (i-1)*L+1: i*L ) * Sigma0(:,:,i) * D(:, (i-1)*L+1: i*L )';
    end
    H = D'*inv(DBD + lambda * eye(N*L));
    Ht = H*y;      HD = H * D;
    
    mu_x = zeros(usedNum*L,1);
    Sigma_x = repmat(zeros(L),[1 1 usedNum]);
    Cov_x = Sigma_x;
    B = zeros(L,L);
    for i = 1 : usedNum
        seg = [(i-1)*L+1 : i*L];
        mu_x(seg) = Sigma0(:,:,i) * Ht(seg);       % solution
        Sigma_x(:,:,i) = Sigma0(:,:,i) - Sigma0(:,:,i) * HD(seg,seg) * Sigma0(:,:,i); 
        Cov_x(:,:,i) = Sigma_x(:,:,i) + mu_x(seg) * mu_x(seg)';
        
        % Estimate covariance matrix
        B = B + Cov_x(:,:,i)/gamma(i);  
    end
    
    if usedNum <= N,
        B = eye(L); invB = B;
    else
        B = B./norm(B,'fro');
        invB = inv(B);
    end

    gamma_old = gamma;
    if LearnLambda == 1   % learn lambda (note that this rule is not robust at low SNR cases)
        
        for i =  1 : usedNum
            gamma(i) = sum(sum(invB .* Cov_x(:,:,i)))/L;     % learn gamma_i
            Sigma0(:,:,i) = B * gamma(i);
        end
        
        if EnhanceLambda == 1               % modify the lambda rule for low/mild SNR cases
            PGP = diag(Phi*diag(gamma)*Phi');
            lambda = norm(y - D * mu_x,2)^2/(N*L)  +  lambda * sum(PGP./(lambda+PGP))/N;
        else                                % use the original lambda rule for high SNR cases
            PGP = Phi*diag(gamma)*Phi';
            lambda = norm(y - D * mu_x,2)^2/(N*L) + lambda * trace(PGP*inv(lambda*eye(N)+PGP))/N;
        end

    else    % not learn lambda, but only learn gamma_i
        for i =  1 : usedNum
            gamma(i) = sum(sum(invB .* Cov_x(:,:,i)))/L;
            Sigma0(:,:,i) = B * gamma(i);
        end
    end

    
    % ================= Check stopping conditions, etc. ==============
    if (PRINT) disp([' iters: ',num2str(count),'   num coeffs: ',num2str(usedNum), ...
            '   gamma change: ',num2str(max(abs(gamma - gamma_old)))]); end;
    if (count >= MAX_ITERS), if PRINT, fprintf('Reach max iterations. Stop\n\n'); end; break;  end;

    if (size(mu_x) == size(mu_old))
        dmu = max(max(abs(mu_old - mu_x)));
        if (dmu < EPSILON)  break;  end;
    end;
    

end;


% Expand hyperparameters
gamma_ind = sort(keep_list);
gamma_est = zeros(M,1);
gamma_est(keep_list,1) = gamma;  


% Transfer to original weight matrix size (vec -> matrix)
X = zeros(M,L); 
for i = 1 : usedNum
    X(keep_list(i),:) = mu_x((i-1)*L+1 : i*L)';
end

if (PRINT) fprintf('\nT-SBL finished !\n'); end
return;



