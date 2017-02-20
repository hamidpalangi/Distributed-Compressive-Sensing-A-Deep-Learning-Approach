
1. Description: 
This package includes the codes of T-SBL/T-MSBL and simulations in the following paper

Zhilin Zhang, Bhaskar D. Rao, Sparse Signal Recovery with 
Temporally Correlated Source Vectors Using Sparse Bayesian Learning, 
IEEE Journal of Selected Topics in Signal Processing, 
Special Issue on Adaptive Sparse Representation of
Data and Applications in Signal and Image Processing, 2011

Feel free to contact me for any questions (z4zhang@ucsd.edu)


2. Version: 2.3 (updated on July 30, 2011)


3. I strongly suggest you to spend 3 minutes to read the pdf file before use the codes:

         Z.Zhang, Master the Usage of TSBL and TMSBL in 3 Minutes


4. File Description:
    [ Core codes ]
       TSBL.m           :    code of T-SBL  (version: 1.4)
       TMSBL.m        :   code of T-MSBL (version: 1.9)
     
    [ Auxiliary codes]
       perfSupp.m        :  code for measuring failure rates
       MSBL.m            :  code of M-SBL (used for performance comparison)
       MFOCUSS.m     :  code of M-FOCUSS (used for performance comparison)

    [ Demo codes]
       demo.m                           : demo for comparison of T-SBL,T-MSBL and M-SBL
       demo_fig3.m                  : demo for re-producing Fig.3 in the above IEEE paper
       demo_fig6_SNR10.m     : demo for re-producing Fig.6 in the above IEEE paper
       demo_fig8.m                   : demo for re-producing Fig.8 in the above IEEE paper
       demo_time_varying.m    : demo showing the use of T-MSBL for the time-varying sparsity model
       demo_identicalVector.m : demo showing the use of T-MSBL for the MMV model with identical source vectors



