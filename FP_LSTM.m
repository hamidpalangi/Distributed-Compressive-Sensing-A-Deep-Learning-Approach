function [Fs] = FP_LSTM(r,Ws,y_prev,c_prev)
% LSTM forward pass
%% Contact: hamidp@ece.ubc.com

Fs.yg = tanh( Ws.W4*r + Ws.Wrec4*y_prev );
Fs.i = Ws.W3*r + Ws.Wrec3*y_prev; Fs.i = sigmf(Fs.i,[1 0]);% Note: sigmf(x,[a c])=1/(1+exp( -a*(x-c) ))
Fs.c = c_prev + Fs.i .* Fs.yg;
Fs.o = Ws.W1*r + Ws.Wrec1*y_prev; Fs.o = sigmf(Fs.o,[1 0]);
Fs.y = Fs.o .* tanh( Fs.c );
Fs.z = Ws.U*Fs.y;
den = sum( exp( Fs.z ),1 ); % den is a 1xnTr vector.
nTr = size(den,2);
for j=1:nTr
    Fs.s(:,j) = exp( Fs.z(:,j) ) ./ den(1,j);
end