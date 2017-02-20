function [gs] = LSTM_Grads(In,T,ncell,Ws)
% This function trains a LSTM network with following parameters:
% In: input train date.
% T: tartget train data
% gs: a structure contatining all LSTM gradient matrices.
%% Contact: hamidp@ece.ubc.ca

n = size(T,1);
m = size(In,1);
L = size(In,2); % Memory length.
nTr = size(In,3); % Number of training samples at each time step.
% Pre-allocation: START
for iL=1:L
    Fs(iL,1) = struct('yg',zeros(ncell,nTr),'i',zeros(ncell,nTr),'c',zeros(ncell,nTr),...
                      'o',zeros(ncell,nTr),'y',zeros(ncell,nTr),'z',zeros(n,nTr),'s',zeros(n,nTr));
    deltas(iL,1) = struct('rec1',zeros(ncell,nTr),'rec3',zeros(ncell,nTr),'rec4',zeros(ncell,nTr),...
                          'u',zeros(n,nTr));
end
% Pre-allocation: END
% Setting gradients to zero
gs.W1 = zeros(ncell,m);
gs.Wrec1 = zeros(ncell);
gs.W3 = zeros(ncell,m);
gs.Wrec3 = zeros(ncell);
gs.W4 = zeros(ncell,m);
gs.Wrec4 = zeros(ncell);    
gs.U = zeros(n,ncell);

gW3c = zeros(ncell,m);
gWrec3c = zeros(ncell);
gW4c = zeros(ncell,m);
gWrec4c = zeros(ncell);
for iL=1:L
    deltas(iL,1).rec1 = zeros(ncell,nTr);
    deltas(iL,1).rec3 = zeros(ncell,nTr);
    deltas(iL,1).rec4 = zeros(ncell,nTr);
    deltas(iL,1).u = zeros(n,nTr);
end
% Forward pass: START
y_init = zeros(ncell,nTr);
c_init = zeros(ncell,nTr);
for iL=1:L
    r = reshape(In(:,iL,:),[m nTr]);        
    if iL==1
        Fs(iL,1) = FP_LSTM(r,Ws,y_init,c_init);
    else
        Fs(iL,1) = FP_LSTM(r,Ws,Fs(iL-1,1).y,Fs(iL-1,1).c);
    end                
end
% Forward pass: END
% Backward pass: START
s0 = reshape(T(:,L,:),[n nTr]);
diff = Fs(L,1).s - s0;       
e = Ws.U' * diff;
% U
deltas(L,1).u = diff;
% Output Gate
deltas(L,1).rec1 = Fs(L,1).o .* ( 1 - Fs(L,1).o ) .* tanh( Fs(L,1).c ) .* e;
% Input Gate
deltas(L,1).rec3 = ( 1 - tanh( Fs(L,1).c ) ).*( 1 + tanh( Fs(L,1).c ) ) .*...
             Fs(L,1).o .* e;
% yg
deltas(L,1).rec4 = deltas(L,1).rec3;
for iL=L-1:-1:1
    s0 = reshape(T(:,iL,:),[n nTr]);
    diff = Fs(iL,1).s - s0;
    e = Ws.U' * diff;
% U
    deltas(iL,1).u = diff;
% Output Gate
    deltas(iL,1).rec1 = ( Fs(iL,1).o .* (1-Fs(iL,1).o) .* tanh( Fs(iL,1).c ) )...
                        .* (Ws.Wrec1' * deltas(iL+1,1).rec1 + e);
% Input Gate
    temp3_4 = ( 1 - tanh( Fs(iL,1).c ) ).*( 1 + tanh( Fs(iL,1).c ) ).* Fs(iL,1).o;
    deltas(iL,1).rec3 = temp3_4 .* (Ws.Wrec3' * deltas(iL+1,1).rec3 + e);
% yg
    deltas(iL,1).rec4 = temp3_4 .* (Ws.Wrec4' * deltas(iL+1,1).rec4 + e);        
end
% Backward pass: END
% Gradients: START
% U gradient
for iL=1:L        
    gs.U = gs.U + deltas(iL,1).u * Fs(iL,1).y';
end
% Output Gate, Input Gate, yg
for iL=1:L
    r = reshape(In(:,iL,:),[m nTr]);
    % Output Gate        
    gs.W1 = gs.W1 + deltas(iL,1).rec1 * r';
    if iL>1
        gs.Wrec1 = gs.Wrec1 + deltas(iL,1).rec1 * Fs(iL-1,1).y';
    end
    % Input Gate and yg for L=1.        
    if iL==1
        % for iL=1, dc(iL-1)/dW is zero.
        bi = Fs(iL,1).yg .* Fs(iL,1).i .* ( 1 - Fs(iL,1).i );
        bg = Fs(iL,1).i .* ( 1 - Fs(iL,1).yg ) .* ( 1 + Fs(iL,1).yg );
        gs.W3 = gs.W3 + ( deltas(iL,1).rec3 .* bi ) * r';
        gs.W4 = gs.W4 + ( deltas(iL,1).rec4 .* bg ) * r';
    end
end
% Input Gate and yg for rest of L.
if L>1
    for ir=1:nTr
        gW3c_prev = zeros(ncell,m);
        gWrec3c_prev = zeros(ncell);
        gW4c_prev = zeros(ncell,m);
        gWrec4c_prev = zeros(ncell);
        for iL=2:L
            r = reshape(In(:,iL,ir),[m 1]);
            % Input Gate
            bi_r = Fs(iL,1).yg(:,ir) .* Fs(iL,1).i(:,ir) .* ( 1 - Fs(iL,1).i(:,ir) );
            gWrec3c = gWrec3c_prev + bi_r * Fs(iL-1,1).y(:,ir)';                
            gs.Wrec3 = gs.Wrec3 + sparse( diag( deltas(iL,1).rec3(:,ir) ) ) * gWrec3c;
            gWrec3c_prev = gWrec3c;
            gW3c = gW3c_prev + bi_r * r';                
            gs.W3 = gs.W3 + sparse( diag( deltas(iL,1).rec3(:,ir) ) ) * gW3c;
            gW3c_prev = gW3c;
            % yg
            bg_r = Fs(iL,1).i(:,ir) .* ( 1 - Fs(iL,1).yg(:,ir) ) .* ( 1 + Fs(iL,1).yg(:,ir) );
            gWrec4c = gWrec4c_prev + bg_r * Fs(iL-1,1).y(:,ir)';
            gs.Wrec4 = gs.Wrec4 + sparse( diag( deltas(iL,1).rec4(:,ir) ) ) * gWrec4c;
            gWrec4c_prev = gWrec4c;
            gW4c = gW4c_prev + bg_r * r';                
            gs.W4 = gs.W4 + sparse( diag( deltas(iL,1).rec4(:,ir) ) ) * gW4c;
            gW4c_prev = gW4c;
        end
    end
end
gs.W1 = gs.W1 ./ nTr;
gs.W3 = gs.W3 ./ nTr;
gs.W4 = gs.W4 ./ nTr;
gs.Wrec1 = gs.Wrec1 ./ nTr;
gs.Wrec3 = gs.Wrec3 ./ nTr;
gs.Wrec4 = gs.Wrec4 ./ nTr;
gs.U = gs.U ./ nTr;
% Gradients: END
