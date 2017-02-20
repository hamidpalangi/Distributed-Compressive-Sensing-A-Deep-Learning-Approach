function [X,gamma_ind,gamma_est,count] = MSBL(Phi, Y, lambda, Learn_Lambda, varargin)
% Sparse Bayesian Learning for Mulitple Measurement Vector (MMV) problems. 
% *** The version is suitable for noisy cases ***
% It can also be used for single measurement vector problem without any modification.
%
% Command Format:
% [X,gamma_ind,gamma_est,count] ...
%     = MSBL(Phi,Y,lambda,Learn_Lambda,'prune_gamma',1e-4,'max_iters',500,'epsilon',1e-8,'print',0);
% [X,gamma_ind,gamma_est,count] = MSBL(Phi,Y, lambda, Learn_Lambda, 'prune_gamma', 1e-4);
% [X,gamma_ind,gamma_est,count] = MSBL(Phi,Y, lambda, Learn_Lambda);
%
% ===== INPUTS =====
%   Phi         : N X M dictionary matrix
%
%   Y           : N X L measurement matrix
%
%   lambda      : Regularization parameter. Sometimes you can set it being the
%                 noise variance value, which leads to sub-optimal
%                 performance. The optimal value is generally slightly larger than the
%                 noise variance vlaue. You need cross-validation methods or
%                 other methods to find it. 
%
%  Learn_Lambda : If Learn_Lambda = 1, use the lambda as initial value and learn the optimal lambda 
%                 using its lambda learning rule. But note the
%                 learning rule is not robust when SNR <= 15 dB. 
%                    If Learn_Lambda = 0, not use the lambda learning rule, but instead, use the 
%                 input lambda as the final value.
%
%  'PRUNE_GAMMA' : Threshold for prunning small hyperparameters gamma_i.
%                  In noisy cases, you can set MIN_GAMMA = 1e-3 or 1e-4.
%                  In strong noisy cases (e.g. SNR < 5 dB), set MIN_GAMMA = 1e-2 for better 
%                  performance.
%                   [ Default value: MIN_GAMMA = 1e-3 ]
%
%  'MAX_ITERS'   : Maximum number of iterations.
%                    [ Default value: MAX_ITERS = 2000 ]
%
%  'EPSILON'     : Threshold to stop the whole algorithm. 
%                    [ Default value: EPSILON = 1e-8   ]
%
%  'PRINT'       : Display flag. If = 1: show output; If = 0: supress output
%                    [ Default value: PRINT = 0        ]
%
% ===== OUTPUTS =====
%   X          : the estimated solution matrix, or called source matrix (size: M X L)
%   gamma_ind  : indexes of nonzero gamma_i
%   gamma_est  : final value of the M X 1 vector of hyperparameter values
%   count      : number of iterations used
%
%
% *** Reference ***
% [1] David P. Wipf, Bhaskar D. Rao, An Empirical Bayesian Strategy for Solving
%     the Simultaneous Sparse Approximation Problem, IEEE Trans. Signal
%     Processing, Vol.55, No.7, 2007.
%
% *** Author ***
%   Zhilin Zhang (z4zhang@ucsd.edu) 
%   (Modified based on David Wipf's original code such that the code is suitable for noisy cases)
%
% *** Version ***
%   1.1 (02/12/2011)
%
% *** See Also ***
%   TSBL        TMSBL
%
  


% Dimension of the Problem
[N M] = size(Phi); 
[N L] = size(Y);  

% Default Control Parameters 
PRUNE_GAMMA = 1e-3;       % threshold for prunning small hyperparameters gamma_i
EPSILON     = 1e-8;       % threshold for stopping iteration. 
MAX_ITERS   = 2000;       % maximum iterations
PRINT       = 0;          % don't show progress information


if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'prune_gamma'
                PRUNE_GAMMA = varargin{i+1}; 
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

if (PRINT) fprintf('\nRunning MSBL ...\n'); end


% Initializations 
gamma = ones(M,1); 
keep_list = [1:M]';
m = length(keep_list);
mu = zeros(M,L);
count = 0;                        % iteration count


% *** Learning loop ***
while (1)

    % *** Prune weights as their hyperparameters go to zero ***
    if (min(gamma) < PRUNE_GAMMA )
        index = find(gamma > PRUNE_GAMMA);
        gamma = gamma(index);  % use all the elements larger than MIN_GAMMA to form new 'gamma'
        Phi = Phi(:,index);    % corresponding columns in Phi
        keep_list = keep_list(index);
        m = length(gamma);
    end;


    mu_old =mu;
    Gamma = diag(gamma);
    G = diag(sqrt(gamma));
        
    % ****** estimate the solution matrix *****
    [U,S,V] = svd(Phi*G,'econ');
   
    [d1,d2] = size(S);
    if (d1 > 1)     diag_S = diag(S);
    else            diag_S = S(1);      end;
       
    Xi = G * V * diag((diag_S./(diag_S.^2 + lambda + 1e-16))) * U';
    mu = Xi * Y;
    
    % *** Update hyperparameters, i.e. Eq(18) in the reference ***
    gamma_old = gamma;
    mu2_bar = sum(abs(mu).^2,2)/L;

    Sigma_w_diag = real( gamma - (sum(Xi'.*(Phi*Gamma)))');
    gamma = mu2_bar + Sigma_w_diag;

    % ***** the lambda learning rule *****
    % You can use it to estimate the lambda when SNR >= 15 dB. But when SNR < 15 dB, 
    % you'd better use other methods to estimate the lambda, since it is not robust 
    % in strongly noisy cases.
    if Learn_Lambda == 1
        lambda = (norm(Y - Phi * mu,'fro')^2/L)/(N-m + sum(Sigma_w_diag./gamma_old)); 
    end;
    
    
    % *** Check stopping conditions, etc. ***
    count = count + 1;
    if (PRINT) disp(['iters: ',num2str(count),'   num coeffs: ',num2str(m), ...
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

if (PRINT) fprintf('\nFinish running ...\n'); end
return;



