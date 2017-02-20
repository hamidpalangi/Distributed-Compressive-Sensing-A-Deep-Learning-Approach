function S_OMP = SOMP(phi_m,Y,res_min)


%% Contact: Hamid Palangi, Email: hamidp@ece.ubc.ca
% Simaltaneous Orthogonal Matching Pursuit
m = size(phi_m,1);
n = size(phi_m,2);
L = size(Y,2);
iter_max = L*m;done=0;
j = 0; R = Y; S_OMP = zeros(n,L); C = 0;
I = [];   
while done == 0
    j = j+1;
    C = phi_m'*R; 
    for iC=1:size(C,1)
       Cnew(iC,1) = norm(C(iC,:)); 
    end
    [val1,idx1] = max( Cnew );
    if length( find( I == idx1 ) ) == 0
        I = [I;idx1];
        phiT = phi_m(:,I);
        S_temp = lscov(phiT,Y);  % Least squares
        R = Y-phiT*S_temp;
        normR = 0;
        for iR=1:size(R,2)
            normR = normR + norm(R(:,iR));
        end
        if (j > iter_max) || (normR < res_min)
            done = 1;
        end
    else
        done = 1;
    end
    if done == 1
        for iSOMP=1:size(Y,2)
            S_OMP(:,iSOMP) = zeros(n,1);
            S_OMP(I,iSOMP) = S_temp(:,iSOMP);
        end        
    end
end
