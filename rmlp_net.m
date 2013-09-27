function net = rmlp_net(IUC, HUC1, HUC2, OUC)
% RMLP_net - setup recurrent multilayer perceptron network
% where the first hidden layer is recurrent and the second
% one is not.
% Bias input is not considered.
% net = rmlp_net(IUC, HUC1, HUC2, OUC)
% where
% ======================================================
% Outputs include:
% net - new network structure
% ======================================================
% Inputs include:
% IUC  - number of input units
% HUC1 - number of hidden units for the first hidden layer
% HUC2 - number of hidden units for the second hidden layer
% OUC  - number of output units

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 10, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

% set number of all units
AUC = IUC + 2*HUC1 + HUC2 + OUC;

% set number of all neurons
ANC = HUC1 + HUC2 + OUC;

% set numbers of units
net.numInputUnits    = IUC;
net.numOutputUnits   = OUC;
net.numHiddenLayers  = 2;
net.numHiddenLayer1  = HUC1;
net.numHiddenLayer2  = HUC2;
net.numAllUnits      = AUC;
net.numAllNeurons    = ANC;

% set neuron masks
net.maskInputUnits   = [zeros(HUC1,1); ones(IUC, 1); zeros(AUC-IUC-HUC1, 1)];
net.maskOutputUnits  = [zeros(AUC-OUC, 1); ones(OUC, 1)];
net.indexInputUnits  = find(net.maskInputUnits);
net.indexOutputUnits = find(net.maskOutputUnits);

% number of weights: initialization
n=1;
% set weights
weight = struct('dest',0,'source',0,'layer',0,'delay',0,'value',0,'const',false,'act',1,'wtype',1);

% weights for input layer to first hidden layer
for i = (1 : HUC1),
    % recurrent weights
    for j = (1 : HUC1),  
        net.weights(n) = weight;
        net.weights(n).dest   = i;
        net.weights(n).source = j;
        net.weights(n).layer  = 1;
        net.weights(n).delay  = 1;
        n = n+1;
    end;
    % weights for input to first hidden layer
    for j = (HUC1+1 : IUC+HUC1),
        net.weights(n) = weight;
        net.weights(n).dest   = i;
        net.weights(n).source = j;
        net.weights(n).layer  = 1;
        n = n+1;
    end;
end;

% weights for first hidden layer to second hidden layer
for i = (HUC1+1 : HUC1+HUC2),
    for j = (1 : HUC1),
        net.weights(n) = weight;
        net.weights(n).dest   = i;
        net.weights(n).source = j;
        net.weights(n).layer  = 2;
        n = n+1;
    end;
end;

% weights for first hidden layer to second hidden layer
for i = (HUC1+HUC2+1 : HUC1+HUC2+OUC),
    for j = (1 : HUC2),
        net.weights(n) = weight;
        net.weights(n).dest   = i;
        net.weights(n).source = j;
        net.weights(n).layer  = 3;
        n = n+1;
    end;
end;

% set number of weights
net.numWeights = n-1;

% initialize weight matrices from [-0.25, 0.25]
for i=(1:net.numWeights),
    net.weights(i).value = rand ./ 2  - 0.25;
end;