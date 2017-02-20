function S_OMP = ISOMP_LSTM(phi_m,Y,res_min,Ws,winSize,input_zero_mean)

%% Contact: Hamid Palangi, Email: hamidp@ece.ubc.ca
% Distributed Compressed Sensing with Long Short Term Memory.
m = size(phi_m,1);
n = size(phi_m,2);
ncell = size(Ws.Wrec1,1);
L = size(Y,2);
iter_max = n;
done=0;
shiftLen = floor(winSize/2);
Rinit = Y;
R = zeros(m*winSize,L);
% Input data preparation for CNN: START**
temp1 = Rinit; 
temp2 = zeros(m,L+2*shiftLen); % one shift for left and one for right.
temp2( : , shiftLen+1 : shiftLen+L ) = temp1;
for iL=1:L       
   temp3 = temp2( : , iL:iL-1+winSize );
   R(:,iL) = temp3(:);
end
% Input data preparation for CNN: END**
S_OMP = zeros(n,L);
cell_I = cell(L,1);
j = 0;
Rtemp = zeros(m,L);
Stemp = cell(L,1);
y_init = zeros(ncell,1);
c_init = zeros(ncell,1);
while done == 0    
    j = j+1;
    normR=0;
    for iL=1:L        
        r1 = R(:,iL);y1 = Y(:,iL);I = cell_I{iL,1}; 
        if input_zero_mean == 1
            r1 = r1 - mean(r1);
            r1 = r1/max( abs( r1 ) );
        else
            r1 = r1/max( abs( r1 ) );
        end
        % LSTM forward pass: START
        if iL==1
            Fs(iL,1) = FP_LSTM(r1,Ws,y_init,c_init);
        else
            Fs(iL,1) = FP_LSTM(r1,Ws,Fs(iL-1,1).y,Fs(iL-1,1).c);
        end 
        % LSTM forward pass: END
        s_hat = Fs(iL,1).s;        
        [val1,idx1] = max( abs(s_hat) );
        if length( find( I == idx1 ) ) == 0
            I = [I;idx1];
            phiT = phi_m(:,I);
            s_temp = pinv(phiT)*y1;  % Least squares
            r1temp = y1-phiT*s_temp;
            normR = normR + norm(r1temp);
        else
            continue;
        end       
        cell_I{iL,1} = I;
        Rtemp(:,iL) = r1temp;
        Stemp{iL,1} = s_temp;
    end 
    if (j > iter_max) || (normR < res_min )
        done = 1;
    end
    if done == 1
        for iL=1:L
            I = cell_I{iL,1};
            S_OMP(I,iL) = Stemp{iL,1};
        end
    else
        % Input data preparation for CNN: START**
        temp1 = Rtemp; 
        temp2 = zeros(m,L+2*shiftLen); % one shift for left and one for right.
        temp2( : , shiftLen+1 : shiftLen+L ) = temp1;
        for iL=1:L       
           temp3 = temp2( : , iL:iL-1+winSize );
           R(:,iL) = temp3(:);
        end
        % Input data preparation for CNN: END**
    end

end



