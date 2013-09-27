% Main function of ESN Training and Testing;

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 11, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

clear;
clc;
% Generate ESN for training
net = esn_net(25, 600, 1);

% Generate training data
[I_data, T_data] = seq_gen_esn(26);

% Train ESN
net_trained = esn_train(net,I_data,T_data);

% Test ESN
[original_out,net_out,error] = esn_test(net_trained);
