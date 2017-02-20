function S_OMP = NISOMPnew(phi_m,Y,res_min,Wnew,Unew,num)


%% Contact: Hamid Palangi, Email: hamidp@ece.ubc.ca
% num: number of hidden units
% Non-Linear Weighted Simaltaneous Orthogonal Matching Pursuit (NISOMP)

m = size(phi_m,1);
n = size(phi_m,2);
L = size(Y,2);
iter_max = L*m;done=0;

j = 0; R = Y; S_OMP = zeros(n,L);
I = [];      
while done == 0
    j = j+1;
    for i5=1:size(R,2) 
        R(:,i5) = R(:,i5)/max( abs( R(:,i5) ) );
    end
    H_test = ones(num, size(R,2));     
    H_test(1:num,:) = 1 ./ (1 + exp(-Wnew*R));
    S_hatTemp=(H_test' * Unew)';  
    S_hat = S_hatTemp;
    row_norms = zeros(n,1);
    for inorm=1:size(S_hat,1)
       row_norms(inorm,1) = norm( S_hat(inorm,:) ); 
    end
    [val1,idx1] = max( row_norms );
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





