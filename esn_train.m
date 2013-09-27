function net_trained = esn_train(net,I_data,T_data)
% esn_train - train Echo State Network
% net_trained = esn_train(net)
% where
% ======================================================
% Inputs include:
% net    - ESN to be trained
% I_data - Input data of training sequence
% T_data - Target data of training sequence
% ======================================================
% Outputs include:
% net_trained  - ESN after training
% See also: seq_gen_esn, esn_net, esn_test

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 19, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

%>>>>>>>>> Obtain parameters from RMLP net <<<<<<<<<<<<<<<<<<
AUC  = net.numAllUnits;           % number of all units
IUC  = net.numInputUnits;         % number of input units
OUC  = net.numOutputUnits;        % number of output units
HUC  = net.numHiddenLayer;        % number of hidden units
drWeights = net.reservoirWeights; % dynamic reservoir weights matrix
inWeights = net.inputWeights;     % input matrix
bkWeights = net.backWeights;      % backward weights from output layer
ouWeights = net.outputWeights;    % output weights matrix
bl_out    = net.bl_out;           % type of output neuron
int_bk    = net.int_bk;           % intensity of feedback
attenu    = net.attenu;           % attenuation ratio for the signal

%>>>>>>>>>>>>>>>>> Parameter Check <<<<<<<<<<<<<<<<<<<<<<<<<<
[inpSize, inpNum] = size(I_data');
[tarSize, tarNum] = size(T_data');
if inpSize ~= IUC, 
    error ('Number of input units and input pattern size do not match.'); 
end;
if tarSize ~= OUC, 
    error ('Number of output units and target pattern size do not match.'); 
end;
if inpNum ~= tarNum, 
    error ('Number of input and output patterns are different.'); 
end;

%>>>>>>>>>>>  Initialization of Training <<<<<<<<<<<<<<<<<<<<
I_data  = attenu*I_data;
T_data  = attenu*T_data;
X(1,:)  = zeros(1,HUC);               % initial reservoir state
I1_data = [zeros(1,inpSize); I_data]; % add zero to initial input
T1_data = [zeros(1,tarSize); T_data]; % add zero to initial output
timeflag= cputime;                    % a timer to save the training time
wb = waitbar(0, 'Echo State Network Training in Progress...'); 
T0 = 1000;                            % washout time
fprintf('\nThe echo state network training is in process...\n');

%>>>>>>>>>>>>>>> Main Loop of ESN Training <<<<<<<<<<<<<<<<<<
for i = (1:inpNum),
    waitbar(i/inpNum,wb)
    set(wb,'name',['Progress = ' sprintf('%2.1f',i/inpNum*100) '%']);
    X(i+1,:) = hyperb((inWeights*I1_data(i+1,:)' + drWeights*X(i,:)' + ...
                int_bk*bkWeights*T1_data(i,:)' + 0.001*(rand(1,HUC)-0.5)')');
end;
close(wb);
fprintf('Please wait for another while...\n');

%>>>>>> Calculate output weights and update ESN <<<<<<<<<<<<<
if (bl_out == 1),
    ouWeights = (pinv(X(T0+2:end,:))*(T_data(T0+1:end,:)))'; % linear output
else
    ouWeights = (pinv(X(T0+2:end,:))*(inv_hyperb(T_data(T0+1:end,:))))';
end;
net.outputWeights = ouWeights;
net_trained       = net;
timeflag          = cputime - timeflag;
fprintf('Training accomplished! Total time is %2.2f hours.\n',timeflag/3600);