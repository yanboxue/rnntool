function [X1, X2, out] = rmlp_run(net,I_data,R_data)
% RMLP_run - Run RMLP
% where the first hidden layer is recurrent and the second
% one is not.
% Bias input is not considered.
% [X1,X2,out] = rmlp_run(net, I_data, R_data)
% Input: 
% net    - trained RMLP network
% I_data - input data
% R_data - recurrent data of last state
% Output:
% X1     - output of the first hidden layer
% X2     - output of the second hidden layer
% out    - network output

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 12, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

% Fetch parameters from net
ANC  = net.numAllNeurons;
IUC  = net.numInputUnits;
OUC  = net.numOutputUnits;
HUC1 = net.numHiddenLayer1;
HUC2 = net.numHiddenLayer2;
weights_all   = [net.weights.value]; % get weights value
weights_group = [net.weights.dest]; % define the group that the weights belong to

% divide the weights of the net into group from #1 to #ANC
for i = (1:ANC),
    weights(i).value  = weights_all(min(find(weights_group == i)) : max(find(weights_group == i)));
    weights(i).length = length(find(weights_group == i));
end;

% parameter checking
[inpSize, inpNum] = size(I_data');
[recSize, recNum] = size(R_data');
if inpSize ~= IUC, 
    error ('Number of input units and input pattern size do not match.'); 
end;
if recSize ~= HUC1, 
    error ('Number of last recurrent units and current recurrent units do not match.'); 
end;

X1 = []; % output row vector of first hidden layer
X2 = []; % output row vector of second hidden layer
out = []; % network output row vector 
% output of first hidden layer
for i = (1:HUC1),
    x(i).value = hyperb(weights(i).value*[R_data,I_data]');
    X1 = [X1, x(i).value];
end;

% output of second hidden layer
for i = (HUC1+1:HUC1+HUC2),
    x(i).value = hyperb(weights(i).value*X1');
    X2 = [X2, x(i).value];
end;

% output of the network - linear neuron
for i = (HUC1+HUC2+1:ANC),
    x(i).value = weights(i).value*X2';
    out = [out, x(i).value];
end;