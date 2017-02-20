Copy right:
Permission is granted for anyone to copy, use, modify, or distribute 
this program and accompanying programs and documents for any purpose, 
provided this copyright notice is retained and prominently displayed, 
along with a note saying that the original programs are available from 
Hamid Palangi, "hamidp@ece.ubc.ca". 
The programs and documents are distributed without any warranty, 
express or implied. As the programs were written for research purposes only, 
they have not been tested to the degree that would be advisable in any important 
application. All use of these programs is entirely at the user's own risk.

This folder includes  all necessary programs to implement the LSTM-CS method and generates Fig.4 of the
paper. To refer please use the following:
@ARTICLE{hp_LSTM_CS, 
author={Hamid Palangi and Rabab Ward and Li Deng}, 
journal={IEEE Transactions on Signal Processing}, 
title={Distributed Compressive Sensing: A Deep Learning Approach}, 
year={2016}, 
volume={64}, 
number={17}, 
pages={4504-4518}, 
month={Sept},
}

In the folder "Other_Methods", the codes for Bayesian Compressive Sensing from
following references are used. Please refer to their website for copy
right terms and conditions.
[1] Bayesian and Multitask Compressive Sensing: "http://www.ece.duke.edu/~lcarin/bcs_ver0.1.zip"
[2]  Sparse Signal Recovery with Temporally Correlated Source Vectors
Using Sparse Bayesian Learning: "http://sccn.ucsd.edu/~zhang/TMSBL_code.zip"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

To use, simply copy and unzip the "LSTM_CS_Demo" to your local machine and run the program in "Main.m". This will 
regenerate the Fig. 4 of the above paper. 

Please note that running time to regenerate Fig. 4 will be about 16 hours on one core i7 computer. This is because: 
(a) It runs on CPU not GPU. 
(b) It runs reconstruction algorithm 1,440 times (we have 144 different sparsity levels and 10 realization for each one).

If you want just use the codes for your research and learn how they work, 
I recommend to create a faster debug setting for yourself by following modifications to "Main.m":
(a) Set "ncell" in line 47 to 5.
(b) Set "maxEpochLSTM" in line 50 to 5.
(c) Comment line 178 and use "k_vec = 1:20:nsparse;" instead.

Please note that debug setting will give you very bad results but it runs quite fast and helps you to understand the code.

For this work, I have implemented Long Short-Term Memory module of the proposed model from scratch in MATLAB. 
I did this because I wanted to understand it better. 
For sure there are better implementations of LSTM out there.
This is also a good opportunity to learn how exactly it is implemented if you are intereted. 
I have put another README file for LSTM.

The description of the content of it is as follows:

1. Main.m: Calls all necessary sub-functions to train LSTM-CS and NWSOMP models, and evaluate their performance. 
2. FP_LSTM.m: Forward pass of LSTM
3. ISOMP_LSTM.m: LSTM-CS solver
4. LSTM_Grad_Clip.m: Performs gradient clipping for LSTM-CS training
5. LSTM_Grads.m: Calculates LSTM-CS gradients.
6. LSTM_TrainCE.m: Call all subfunctions to train LSTM-CS.
7. mnist_fun.m: Sorts a sparse vector and keeps the k largest non-zero values.
8. NISOMPnew.m: NWSOMP solver.
9. OneLayerBackprop_new.m: For training NWSOMP solver.
10. OneLayerBackprop_new_Biased.m: For training NWSOMP solver when we have bias.
11. SOMP.m: SOMP Solver.
12. TgenerateISOMP_MNIST_MMV.m: Generates training data for NWSOMP.
13. TgenerateISOMP_MNIST_MMV_CE.m: Generates training data for LSTM-CS.
14. MNIST_Data: This folder includes the prepared MNIST data. If you want to prepare it yourself please 
download the MNIST data from "http://yann.lecun.com/exdb/mnist/" and run the "Prepare_MNIST_Data.m".
15. Other_Methods: This folder includes the mfiles for Bayesian methods. Please refer to the corresponding
websites for mentioned above for copy right terms and conditions.
16. Results: The output curves are saved here.
17. Trained_Network: The trained model parameters are saved here.


If you need further information or if you have any feedback, please contact Hamid Palangi "hamidp@ece.ubc.ca".

Good luck!