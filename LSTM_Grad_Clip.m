function [gs] = LSTM_Grad_Clip(gs,th)
% Gradient clipping for LSTM
%% Contact: hamidp@ece.ubc.ca

I = find(gs.W1>th);
gs.W1(I) = th;
I = find(gs.Wrec1>th);
gs.Wrec1(I) = th;
I = find(gs.W3>th);
gs.W3(I) = th;
I = find(gs.Wrec3>th);
gs.Wrec3(I) = th;
I = find(gs.W4>th);
gs.W4(I) = th;
I = find(gs.Wrec4>th);
gs.Wrec4(I) = th;
I = find(gs.U>th);
gs.U(I) = th;
