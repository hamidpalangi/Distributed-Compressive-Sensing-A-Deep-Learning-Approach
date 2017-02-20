function [Ws,ce_lstm,ce_lstm_dev] = LSTM_TrainCE(varargin)
% This function trains a LSTM network with following parameters:
% In: input train date.
% T: tartget train data
% Ws: a structure contatining all LSTM weight matrices.
%% Contact: hamidp@ece.ubc.ca
args = varargin; 
nargs= length(args);
random_init = 1;
nBatch = -1;
Bsize = -1;
for i=1:2:nargs
  switch args{i},
      case 'input train', In = args{i+1};
      case 'target train', T = args{i+1};
      case 'input validation', In_Dev = args{i+1};
      case 'target validation', T_Dev = args{i+1};
      case 'number of cells', ncell = args{i+1};
      case 'step size', StepSize = args{i+1};
      case 'epoch max', maxEpochLSTM = args{i+1};
      case 'number of mini-batches', nBatch = args{i+1};
      case 'initial weight matrices', Ws = args{i+1};random_init = 0;
      case 'trained network path', NetPath = args{i+1};
      case 'mini-batch size', Bsize = args{i+1};
      otherwise
          error('The option does not exist');
  end
end
n = size(T,1);
m = size(In,1);
L = size(In,2); % Memory length.
nTr = size(In,3); % Number of training samples at each time step.
nTr_Dev = size(In_Dev,3);
if nBatch ~= -1
    Bsize = floor(nTr/nBatch); % mini-batch size.
elseif Bsize ~= -1
    nBatch = floor(nTr/Bsize); % Number of mini-batches
end
% Random Initialization: START
if random_init
    Ws.W1 = 0.1*randn(ncell,m);
    Ws.W3 = 0.1*randn(ncell,m);
    Ws.W4 = 0.1*randn(ncell,m);
    Ws.Wrec1 = 0.1*randn(ncell);
    Ws.Wrec3 = 0.1*randn(ncell);
    Ws.Wrec4 = 0.1*randn(ncell);
    Ws.U = 0.1*randn(n,ncell);
end
% Random Initialization: END
% Random Initialization: START
diffWs.W1 = zeros(ncell,m);
diffWs.W3 = zeros(ncell,m);
diffWs.W4 = zeros(ncell,m);
diffWs.Wrec1 = zeros(ncell);
diffWs.Wrec3 = zeros(ncell);
diffWs.Wrec4 = zeros(ncell);
diffWs.U = zeros(n,ncell);
% Random Initialization: END
epoch = 0;
nupdate = 0;
disp('LSTM Training: START');
while epoch < maxEpochLSTM
    tic;
    % Nesterov method
    if ( nupdate < floor(0.1*maxEpochLSTM*nBatch) ) ||... 
       ( nupdate >= floor(0.9*maxEpochLSTM*nBatch) )
        mu = 0.9;
    else
        mu = 0.995;
    end
    err = 0;
    for iB=1:nBatch
        In1 = In(:,:, (iB-1)*Bsize+1 : iB*Bsize );
        T1 = T(:,:, (iB-1)*Bsize+1 : iB*Bsize );
        % New input
        Ws_grad.W1 = Ws.W1 + mu * diffWs.W1;
        Ws_grad.Wrec1 = Ws.Wrec1 + mu * diffWs.Wrec1;
        Ws_grad.W3 = Ws.W3 + mu * diffWs.W3;
        Ws_grad.Wrec3 = Ws.Wrec3 + mu * diffWs.Wrec3;
        Ws_grad.W4 = Ws.W4 + mu * diffWs.W4;
        Ws_grad.Wrec4 = Ws.Wrec4 + mu * diffWs.Wrec4;
        Ws_grad.U = Ws.U + mu * diffWs.U;
        % Gradients
        gs = LSTM_Grads(In1,T1,ncell,Ws_grad);
        % Clipping
        gs = LSTM_Grad_Clip(gs,100); % Clipping gradients to prevent gradient explosion.
        % Parameter update
        diffWs.W1 = mu*diffWs.W1 - StepSize*gs.W1;
        diffWs.Wrec1 = mu*diffWs.Wrec1 - StepSize*gs.Wrec1;
        diffWs.W3 = mu*diffWs.W3 - StepSize*gs.W3;
        diffWs.Wrec3 = mu*diffWs.Wrec3 - StepSize*gs.Wrec3;
        diffWs.W4 = mu*diffWs.W4 - StepSize*gs.W4;
        diffWs.Wrec4 = mu*diffWs.Wrec4 - StepSize*gs.Wrec4;
        diffWs.U = mu*diffWs.U - StepSize*gs.U;
        % Recover original parameters
        Ws.W1 = Ws.W1 + diffWs.W1;
        Ws.Wrec1 = Ws.Wrec1 + diffWs.Wrec1;
        Ws.W3 = Ws.W3 + diffWs.W3;
        Ws.Wrec3 = Ws.Wrec3 + diffWs.Wrec3;
        Ws.W4 = Ws.W4 + diffWs.W4;
        Ws.Wrec4 = Ws.Wrec4 + diffWs.Wrec4;
        Ws.U = Ws.U + diffWs.U;
        % Value of cost function: Training set
        % Forward pass: START
        y_init = zeros(ncell,Bsize);
        c_init = zeros(ncell,Bsize);        
        for iL=1:L
            r = reshape(In1(:,iL,:),[m Bsize]);
            s0 = reshape(T1(:,iL,:),[n Bsize]);        
            if iL==1
                Fs(iL,1) = FP_LSTM(r,Ws,y_init,c_init);
            else
                Fs(iL,1) = FP_LSTM(r,Ws,Fs(iL-1,1).y,Fs(iL-1,1).c);
            end 
            CE = -sum( s0 .* log( Fs(iL,1).s ) , 1 );
            err = err + sum( CE , 2 );
        end
        % Forward pass: END    
        nupdate = nupdate + 1;
    end
    % Value of cost function: Validation set
    % Forward pass: START
    err_dev = 0;
    y_init = zeros(ncell,nTr_Dev);
    c_init = zeros(ncell,nTr_Dev);        
    for iL=1:L
        r = reshape(In_Dev(:,iL,:),[m nTr_Dev]);
        s0 = reshape(T_Dev(:,iL,:),[n nTr_Dev]);        
        if iL==1
            Fs(iL,1) = FP_LSTM(r,Ws,y_init,c_init);
        else
            Fs(iL,1) = FP_LSTM(r,Ws,Fs(iL-1,1).y,Fs(iL-1,1).c);
        end 
        CE = -sum( s0 .* log( Fs(iL,1).s ) , 1 );
        err_dev = err_dev + sum( CE , 2 );
    end
    epoch = epoch + 1;
    epochTime = toc;
    save(strcat([NetPath '_Ws_Epoch' num2str(epoch)]),'Ws');    
    ce_lstm(epoch,1) = err;
    ce_lstm_dev(epoch,1) = err_dev;
    disp(strcat('epoch = ',num2str(epoch),', ce_lstm = ',num2str(err),...
        ', ce_lstm_dev = ', num2str(err_dev),', epochTime = ', num2str(floor(epochTime)),' sec.'));
    save(strcat([NetPath '_CEvecTr']),'ce_lstm');
    save(strcat([NetPath '_CEvecDev']),'ce_lstm_dev');
end
disp('LSTM Training: END');


