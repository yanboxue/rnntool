function [I_data, T_data] = seq_gen_esn(len_subset)
% seq_gen_esn: generate training sequence for ESN
% [I_data, T_data] = seq_gen_esn(len_subset)
% where
% len_subset - length of subset = IUC + OUC
% I_data     - Input data of the network
% T_data     - Target data of the network

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 20, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

%>>>>>>>>>>>>> Initilization <<<<<<<<<<<<<<<<<<<<<
t = 0 : 3999;                      % training sequence time interval
y = signal(t);                     % generate training sequence
len = length(y);
incom_data = (rand(1,len)>0.00);   % incomplete data ratio
y = y.*incom_data;
num_subset = len - len_subset + 1; % number of subset
fprintf('Training sequence generation is in process, please wait...\n')
%>>>>>>>>>>>>>>>>> Main Loop <<<<<<<<<<<<<<<<<<<<<
for i = (1:num_subset),
    I_data(i,:) = y(i:len_subset-2+i); 
    T_data(i,:) = y(len_subset-1+i);
end;
fprintf('Training sequence is generated.\n');
