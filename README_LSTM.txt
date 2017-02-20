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

This README file describes how to use implemented Long Shor-Term Memory (LSTM) module. 
To refer please use the following:
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I have implemented Long Short-Term Memory module of the proposed model from scratch in MATLAB. 
I did this because I wanted to understand it better. 
For sure there are better implementations of LSTM out there.
This is also a good opportunity to learn how exactly it is implemented if you are intereted.

Sample use:

[Ws,ce_lstm,ce_lstm_dev] = LSTM_TrainCE('input train',In,'target train',T...
        ,'input validation',In_Dev,'target validation',T_Dev,'number of cells',ncell...
        ,'step size',StepSize,'epoch max',maxEpochLSTM,'number of mini-batches',nBatch,...
        'trained network path',NetPath);%,'initial weight matrices',Ws_init);

Inputs:
In: is a NxLxnT matrix where "N" is the length of input vector, "L" is the length of time dependency and "nT" is the number of training examples.		
T: The same as "In" but includes output labels.
In_Dev: The same as "In" for validation examples.
T_Dev: The same as "T" for validation examples.		
NetPath: After each iteration trained LSTM parametes are saved here.
Ws_init: If training interrupts we can load the last set of parameters in Ws_init and fine tune from there.

Outputs:
Ws: This is a struct including all LSTM parameters. 
	Input weights: Ws.W1,Ws.W2,Ws.W3,Ws.W4
	Recurrent weights: Ws.Wrec1,Ws.Wrec2,Ws.Wrec3,Ws.Wrec4
ce_lstm: value of cost function over epochs on training set.	
ce_lstm_dev: value of cost function over epochs on validation set.
		
Description of rest of the M files:

1. FP_LSTM.m: Forward pass of LSTM
2. LSTM_Grad_Clip.m: Performs gradient clipping for LSTM training
3. LSTM_Grads.m: Calculates LSTM gradients.
4. LSTM_TrainCE.m: Call all subfunctions to train LSTM.

If you need further information or if you have any feedback, please contact Hamid Palangi "hamidp@ece.ubc.ca".

Good luck!