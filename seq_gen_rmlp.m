function [I_data, T_data] = seq_gen_rmlp(len_seq,len_subset,num_subset,start_point)
% function: [I_data, T_data] = seq_gen_rmlp(len_subset)
% generate training sequence for RMLP network.
% randomly choose a starting point and generate overlapped training data
% where I_data    - Input data of training sequence
%       T_data    - Target data of training sequence
%       len_subset- subset length = I_data + T_data
%       len_seq   - length of sequence: select data from 0 : leq_seq-1
%       num_subset- number of subsets
%       start_point - starting point of the subset
% See also: seq_gen_esn

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% July 6, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

t = 0 : len_seq-1;
y = signal(t);
for i = (1:num_subset),
    I_data(i,:) = y(start_point + (i-1) : start_point + len_subset + i - 3);
    T_data(i,:) = y(start_point + len_subset + i - 2);
end;
