function net = esn_net(IUC, HUC, OUC)
% esn_net - setup Echo State Network
% net = esn_net(IUC, HUC, OUC)
% where
% ======================================================
% Outputs include:
% net  : new network structure
% ======================================================
% Inputs include:
% IUC  : number of input units
% HUC  : number of hidden units in the dynamic reservoir
% OUC  : number of output units
% See also: esn_train, esn_test, seq_gen_esn, rmlp_net

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 19, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

% set number of all units
AUC = IUC + HUC + OUC;
net.bl_out = 1;  % 1: the output is linear neuron, 0: not    
net.int_bk = 0; % intensity of feedback weights
net.attenu = 2/3; % attenuation ratio for the signal

% set numbers of units
net.numInputUnits    = IUC;
net.numOutputUnits   = OUC;
net.numHiddenLayer   = HUC;
net.numAllUnits      = AUC;

% set neuron masks
net.maskInputUnits   = [ones(IUC, 1); zeros(AUC-IUC, 1)];
net.maskOutputUnits  = [zeros(AUC-OUC, 1); ones(OUC, 1)];
net.indexInputUnits  = find(net.maskInputUnits);
net.indexOutputUnits = find(net.maskOutputUnits);

% weights matrix initialization
dr_sp_den = 0.3;          % sparse density of reservoir weights matrix
alpha     = 0.7;          % spectral radius to scale reservoir weights
W0        = 2*rand(HUC) - 1;           % element of WO in [-1,1];
W0        = W0.*(rand(HUC)<dr_sp_den); % let W0 be sparse with density dr_sp_den
net.reservoirWeights = alpha*W0/max(abs(eig(W0))); % dynamic reservoir weights matrix
net.inputWeights     = rand(HUC,IUC)-0.5;          % input weights
net.backWeights      = (rand(HUC,OUC)-0.5);        % backward weights from output layer
net.outputWeights    = zeros(OUC,HUC);             % output weights matrix
