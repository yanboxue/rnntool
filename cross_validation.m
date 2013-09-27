function mse = cross_validation(net)
% [mse1,mse2] = cross_validation(net, time_index)
% Cross-validation function for using in training
% where:
% ================================================
% Inputs include
% net        - the RMLP network for cross-validation
% ================================================
% Outputs include
% mse        - RMSE of the data

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 18, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

IUC  = net.numInputUnits;
OUC  = net.numOutputUnits;
HUC1 = net.numHiddenLayer1;
t    = 6000:6299;
y    = signal(t);
len_subset    = IUC + OUC;           % length of subset
X1   = rand(1,HUC1);

for i = (1:length(t)-(len_subset-1)),
    I_data(i,:) = y(i: i+(len_subset-2));
    T_data(i,:) = y(i+(len_subset-1));
end;

for i = 1:length(t)-(len_subset-1),
    [X11, X2, out(i)] = rmlp_run(net,I_data(i,:),X1);
    X1 = X11;
end;
mse = sqrt(mean((out(1:end) - T_data(1:end)').^2));
