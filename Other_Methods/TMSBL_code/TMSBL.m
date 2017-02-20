function [X, gamma_ind, gamma_est, count, B_est] = TMSBL(Phi, Y, varargin)
% Sparse Bayesian Learning for the MMV model exploiting temporal correlation of each source.
% It is the simplified version of T-SBL. It can also be used in:
%    (1) the single measurement vector (SMV) model (i.e. Y and X are single vectors, not matrices)
%    (2) the MMV model when there is no temporal correlation.
% 
% Command Format:
% (1) For general noisy case (6-22dB):  [X, gamma_ind, gamma_est, count, B_est] = TMSBL(Phi, Y)
% (2) For strongly noisy case (<=6dB):  [X, gamma_ind, gamma_est, count, B_est] = TMSBL(Phi, Y, 'noise','large')
%     For mild noisy case (6-22dB)   :  [X, gamma_ind, gamma_est, count, B_est] = TMSBL(Phi, Y, 'noise','mild')
%     For small noisy case (>22dB)   :  [X, gamma_ind, gamma_est, count, B_est] = TMSBL(Phi, Y, 'noise','small')
%     For noiseless case             :  [X, gamma_ind, gamma_est, count, B_est] = TMSBL(Phi, Y, 'noise','no')
%     [*** Just roughly guess SNR and then choose the above command for a good result (not necessarily optimal) ***]
% (3) Choose your own values for the parameters; for examples:
%     [X,gamma_ind,gamma_est,count,B_est] ...
%        = TMSBL(Phi,Y, 'Prune_Gamma',   1e-5,  'lambda',     0.015,     'Learn_Lambda',1, ...
%                       'Enhance_Lambda',1,     'Matrix_Reg', 2*eye(5),  'MAX_ITERS',   500, ...
%                       'Fix_B',         eye(L),'EPSILON',    1e-8,      'PRINT',       1 );
%     [X,gamma_ind,gamma_est,count,B_est] = TMSBL(Phi,Y, 'Learn_Lambda', 1, 'prune_gamma', 1e-4);
%    
%
% ============================== INPUT ARGUMENTS ============================== 
%   Phi         : N X M dictionary matrix
%   Y           : N X L available measurement matrix, i.e. Y = Phi * X + V.
%
%   'noise'     : If 'noise' = 'no'    : use default values of parameters for noiseless cases
%                 If 'noise' = 'small' : use default values for the cases
%                                        when noise is small (e.g. SNR > 22dB)
%                 If 'noise' = 'mild'  : use default values for the cases
%                                        when noise is mild (e.g. 6 dB < SNR <= 22dB)
%                 If 'noise' = 'large' : use default values for the cases
%                                        when noise is strong (e.g. SNR <= 6dB).
%                                        But if you have some knowledge about T-MSBL,
%                                        I suggest you to select your own values
%                                        for input arguments.                                   
%    **** Note: if you use the input argument 'noise', then all input arguments are invalid, except to: ****
%             'Fix_B', 'MAX_ITERS', 'EPSILON', 'PRINT' 
%    ****  This input argument choose a set of suitable parameter values for the four cases.  ****
%    ****  But the used parameter values may not be optimal. *****
%
%  'PRUNE_GAMMA'  : Threshold for prunning small hyperparameters gamma_i.
%                   In noisy cases, you can set MIN_GAMMA = 1e-3 or 1e-4.
%                   In strong noisy cases (e.g. SNR <= 6 dB), set MIN_GAMMA = 1e-2 for better 
%                   performance.
%                   [ Default value: MIN_GAMMA = 1e-3 for samlle or medium scale data or 1e-2 for 
%                     large-scale data]
%  
%   'lambda'    : Initial value for the regularization parameter. Note that setting lambda being
%                 noise variance only leads to sub-optimal performance. 
%                 The optimal value is generally 2-3 times the true noise variance.
%                 However, in most cases you can use the lambda learning
%                 rule to estimate it, if you set Learn_Lambda = 1
%                 (1) In the noiseless cases, you can simply set lambda = 1e-15 or any other 
%                     very small values (e.g. 1e-10 is also okay),
%                     and run the algorithm without using the lambda
%                     learning rule (i.e. setting Learn_Lambda = 0).  
%                 (2) When SNR >= 7dB, you can use the lambda learning rule 
%                     to learn lambda. In this case, just roughly set an initial value for lambda, 
%                     and set Learn_Lambda = 1.
%                 (3) When SNR <= 6 dB, the lambda learning rule may not be very good. So, you can use 
%                     cross-validation or other methods to estimate it. In this case, set Learn_Lambda = 0.             
%                     However, you can still roughly guess an initial value for
%                     lambda and use the lambda learning rule. Sometimes
%                     the performance is still better than other algorithms.
%                 [ Default value: lambda = 1e-3 ]
%
% 'Learn_Lambda': (1) If Learn_Lambda = 1, use the lambda learning rule (thus the input lambda
%                     is just as initial value). But note the learning rule is not very good
%                     if SNR <= 6 dB (but still lead to better performance than some other algorithms)
%                 (2) If Learn_Lambda = 0, do not use the lambda learning rule, but use the input 
%                     lambda as its final value.
%                 [ Default value: Learn_Lambda = 1 ]
%
% 'Enhance_Lambda': (1) In large/mild noisy cases (<=22dB), set Enhance_Lambda = 1 for better performance
%                   (2) In other cases, set Enhance_Lambda = 0
%                 [ Default value: Enhance_Lambda = 1 ]
%
%  'Matrix_Reg'   : Matrix used to add to the estimated matrix B in each iteration
%                   (mainly used in noisy environment), e.g. c * I, where c is a positive scalar
%                   and I is the identity matrix.
%                   (1)When noise is small (eg. SNR > 22 dB), use zero matrix of the size L x L
%                   (2)When noise is mild (eg. 7 dB - 22 dB), suggest to set Matrix_Reg = 2 * eye(L)
%                   (3)When noise is large (eg.SNR < 6 dB), suggest to set Matrix_Reg = 4 * eye(L)
%                   [  Default value: PLUS_MAT = 2*eye(L) ]
%
%  'Fix_B'        : Do not adaptively estimate the matrix B, but instead use the given value 
%                   as the estimate of B.
%                   [ Default: do not fix B (adaptive estimate B) ]
%
%  'MAX_ITERS'    : Maximum number of iterations.
%                   [ Default value: MAX_ITERS = 2000 ]
%
%  'EPSILON'      : Threshold to stop the whole algorithm (based on the change of the soluiton matrix). 
%                   [ Default value: EPSILON = 1e-8   ]
%
%  'PRINT'        : Display flag. 
%                   If 'PRINT'= 0: supress output; 
%                   If 'PRINT'= 1: print only parameter information
%                   If 'PRINT'= 2: print the algorithm's progress information and parameter information
%                   [ Default value: PRINT = 0        ]
% 
%
% ==============================  OUTPUTS ============================== 
%   X            : The estimated solution matrix(size: M X L)
%   gamma_ind    : Indexes of nonzero rows in the solution matrix
%   gamma_est    : M X 1 vector of gamma_i values
%   count        : Number of iterations used
%   B_est        : Estimated matrix B
%
%
% ==============================  Reference =============================
%   [1] Zhilin Zhang, Bhaskar D. Rao, Sparse Signal Recovery with 
%   Temporally Correlated Source Vectors Using Sparse Bayesian Learning, 
%   IEEE Journal of Selected Topics in Signal Processing, 
%   Special Issue on Adaptive Sparse Representation of
%   Data and Applications in Signal and Image Processing, 2011
%
%
% ==============================  Author ============================== 
%   Zhilin Zhang (z4zhang@ucsd.edu) 
% 
% ==============================  Version ============================== 
%   1.1 (08/12/2010)
%   1.2 (09/12/2010)   
%   1.3 (03/04/2011)  
%   1.4 (03/10/2011)
%   1.5 (05/12/2011) revised the learning rule for lambda and B
%   1.6 (05/20/2011) modified the lambda learning rule
%   1.7 (05/21/2011) add 'noise' input argument to relax the burden of selecting parameters
%   1.8 (05/23/2011) change the default value of 'MAX_ITERS' to 2000
%   1.9 (07/30/2011) change the default value of 'lambda' and 'prune_gamma'
%                    when status == 0 (noiseless case)
%   1.10 (11/11/2011) modify default value of 'prune_gamma' for large-scale dataset
%   1.11 (11/12/2011) modify default value for learning correlation
%
% ==============================  See Also ============================== 
%   TSBL    ARSBL    tMFOCUSS     MSBL
%
    


% Dimension of the Problem
[N M] = size(Phi); 
[N L] = size(Y);  

% Default Parameter Values for Any Cases
EPSILON       = 1e-8;       % threshold for stopping iteration. 
MAX_ITERS     = 2000;       % maximum iterations
PRINT         = 1;          % don't show progress information

% Default Parameter Values (suitable for MILD noisy cases, e.g. 6-22dB)
STATUS        = -1;         % use the default parameter values for mild noisy cases
PRUNE_GAMMA   = 1e-3;       % threshold for prunning small hyperparameters gamma_i
if N >= 500 | M>= 1000, PRUNE_GAMMA = 1e-2; end;
lambda        = 1e-3;       % initial value for lambda
LearnLambda   = 1;          % use the lambda learning rule to estimate lambda
EnhanceLambda = 1;          % enhance the lambda learning rule
LearnB        = 1;          % adaptively estimate the covariance matrix B
if L >= 10, LearnB = 0; B = eye(L); end;          % set for long time-series
MatrixReg     = 2*eye(L);   % added this matrix to regulate the estimated B in each iteration.



if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'noise'
                if strcmp(lower(varargin{i+1}),'no'), STATUS = 0;
                elseif strcmp(lower(varargin{i+1}),'small'), STATUS = 1;
                elseif strcmp(lower(varargin{i+1}),'mild'), STATUS = 2; 
                elseif strcmp(lower(varargin{i+1}),'large'), STATUS = 3;
                else  error(['Unrecognized Value for Input Argument ''Noise''']);
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
            case 'fix_b'
                B = varargin{i+1};
                LearnB = 0;
                if size(B,1) ~= size(B,2) | size(B)~= L
                    error(['Unrecognized Value for Input Argument ''Fix_B''']);
                end
            case 'matrix_reg'
                MatrixReg = varargin{i+1};
                if size(MatrixReg,1) ~= size(MatrixReg,2) | size(MatrixReg,1) ~= L
                    error(['Unrecognized Value for Input Argument ''Matrix_Reg''']);
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
    if N >= 500 | M>= 1000, PRUNE_GAMMA = 1e-3; end;  % set for large-scale dataset
    lambda        = 1e-15;     % fix lambda to this value, and,
    LearnLambda   = 0;         % do not use lambda learning rule, and,
    EnhanceLambda = 0;         % of course, no need the enhancing strategy
    MatrixReg     = 0;         % do not regulate the estimate matrix B
    
elseif STATUS == 1 
    % default values for small noise cases, e.g. SNR > 22 dB
    PRUNE_GAMMA   = 1e-3;      % threshold to prune out small gamma_i
    if N >= 500 | M>= 1000, PRUNE_GAMMA = 1e-2; end;  % set for large-scale dataset
    lambda        = 1e-3;      % initial value for lambda and,
    LearnLambda   = 1;         % use the lambda learning rule, and,
    EnhanceLambda = 0;         % no need to enhance the lambda learning rule
    MatrixReg     = 0;         % do not regulate the estimate matrix B
    
elseif STATUS == 2
    % Default values for mild noise cases, e.g. 6 dB < SNR <= 22 dB
    PRUNE_GAMMA   = 1e-3;       % threshold for prunning small hyperparameters gamma_i
    if N >= 500 | M>= 1000, PRUNE_GAMMA = 1e-2; end;  % set for large-scale dataset
    lambda        = 1e-3;       % initial value for lambda, and,
    LearnLambda   = 1;          % use the lambda learning rule to estimate lambda
    EnhanceLambda = 1;          % enhance the lambda learning rule for mild noisy cases
    MatrixReg     = 2*eye(L);   % added this matrix to regulate the estimated B in each iteration.
    
elseif STATUS == 3
    % Default values for mild noise cases, e.g. SNR <= 6 dB
    PRUNE_GAMMA   = 1e-2;       % threshold for prunning small hyperparameters gamma_i
    lambda        = 1e-2;       % initial value for lambda, and,
    LearnLambda   = 1;          % use the lambda learning rule to estimate lambda
    EnhanceLambda = 1;          % enhance the lambda learning rule for mild noisy cases
    MatrixReg     = 4*eye(L);   % added this matrix to regulate the estimated B in each iteration.

end




if PRINT == 2, 
    fprintf('\n\nRunning T-MSBL...\n');
end

if PRINT
    fprintf('\n====================================================\n');
    fprintf('           Information about parameters...\n');
    fprintf('====================================================\n');
    if STATUS ~= -1,
        fprintf('You Select the Global Input Argument ''Noise''\n');
        fprintf('So, some input arguments'' values set by you may be changed automatically\n');
        fprintf('If you want to use your own input arguments'' values, do not use the ''Noise'' argument\n\n');
    end
    fprintf('PRUNE_GAMMA  : %e\n',PRUNE_GAMMA);
    fprintf('lambda       : %e\n',lambda);
    fprintf('LearnLambda  : %d\n',LearnLambda);    
    fprintf('EnhanceLambda: %d\n',EnhanceLambda);
    fprintf('LearnB       : %d\n',LearnB);
    if LearnB == 0, 
        fprintf('Freeze B to\n'); disp(B);
    else
        fprintf('MatrixReg    :\n');disp(MatrixReg);
    end
    fprintf('EPSILON      : %e\n',EPSILON);
    fprintf('MAX_ITERS    : %d\n\n',MAX_ITERS);
end



% Initializations 
gamma = ones(M,1);  
keep_list = [1:M]';
usedNum = length(keep_list);
mu = zeros(M,L);
count = 0;

% Learning loop 
while (1)

    % =========== Prune weights as their hyperparameters go to zero ===========
    if (min(gamma) < PRUNE_GAMMA )
        index = find(gamma > PRUNE_GAMMA);
        gamma = gamma(index);  
        Phi = Phi(:,index);    
        keep_list = keep_list(index);
        usedNum = length(gamma);
    end;

    % ================ Evaluate the mean (i.e. current solution) ===============
    mu_old = mu;
    Gamma = diag(gamma);
    G = diag(sqrt(gamma));
    [U,S,V] = svd(Phi*G,'econ');
    [d1,d2] = size(S);
    if (d1 > 1)     diag_S = diag(S);
    else            diag_S = S(1);      end; 
    Xi = G * V * diag((diag_S./(diag_S.^2 + lambda + 1e-16))) * U';
    mu = Xi * Y;
    

    % ================= The Lambda Learning Rule =================
    % Generally, it is good when SNR >=7 dB. But when SNR <= 6 dB, it is not good enough, 
    % but still leads to better performance than many other algorithms. 
    if LearnLambda == 1
        
        if EnhanceLambda == 1
            % enhance the lambda learning rule for robustness in mild/strong noisy cases
            PGP = diag(Phi*Gamma*Phi');
            lambda = norm(Y - Phi * mu,'fro')^2/(N*L) + lambda * sum(PGP./(lambda+PGP))/N;
        else
            % the original learning rule in the paper
            PGP = Phi*Gamma*Phi';
            lambda = norm(Y - Phi * mu,'fro')^2/(N*L) + lambda * trace(PGP*inv(lambda*eye(N)+PGP))/N;
        end
    end

    % ======== modify mu to compensate negative affect of temporal correlation ======
    if LearnB == 1   % adaptively estimate B
        if usedNum <= N
            B = eye(L);   % This strategy improves MSE of the solution
        else
            B = zeros(L,L);
            for i =  1 : usedNum
                B = B + mu(i,:)' * mu(i,:)/gamma(i);
            end
            B = B + MatrixReg;    % this modification is used in low SNR cases
            B = B./norm(B,'fro');
        end
    end;
    
    mub = mu * inv(B);
    mu2_bar = sum(mub.*mu,2)/L;
    
    
    % ===================== Update hyperparameters ===================
    Sigma_w_diag = real( gamma - (sum(Xi'.*(Phi*Gamma)))');
    
    gamma_old = gamma;
    gamma = mu2_bar + Sigma_w_diag;


    % ================= Check stopping conditions, etc. ==============
    count = count + 1;
    if (PRINT==2) disp([' iters: ',num2str(count),'   num coeffs: ',num2str(usedNum), ...
            '   gamma change: ',num2str(max(abs(gamma - gamma_old)))]); end;
    if (count >= MAX_ITERS) break;  end;

    if (size(mu) == size(mu_old))
        dmu = max(max(abs(mu_old - mu)));
        if (dmu < EPSILON)  break;  end;
    end;

end;

% Expand hyperparameters 
gamma_ind = sort(keep_list);
gamma_est = zeros(M,1);
gamma_est(keep_list,1) = gamma;  

% expand the final solution
X = zeros(M,L);
X(keep_list,:) = mu;   

B_est = B;

if (PRINT==2) fprintf('\nT-MSBL finished !\n'); end
return;


