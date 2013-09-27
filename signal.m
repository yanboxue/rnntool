function x = signal(t)
% function: x = signal(t)
% where t  - time_interval
%       x  - signal output

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% June 6, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

x = sin(t+sin(t.^2));