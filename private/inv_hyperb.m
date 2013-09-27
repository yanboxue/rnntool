function y = inv_hyperb(x)
% y = inv_hyperb (x)
% inverse of hyperbolic function
% x - input data
% y - output data

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 20, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

y = 0.5*log((1+x)./(1-x));