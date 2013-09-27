function [original_out,net_out,error] = esn_test(net)
% esn_test - test Echo State Network
% [original_out,net_out,error] = esn_test(net)
% where
% ======================================================
% Inputs include:
% net           - ESN to be tested
% ======================================================
% Outputs include:
% original_out  - original output of testing data
% net_out       - ESN output
% error         - error of output
% See also: esn_train, esn_net, seq_gen_esn

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 20, 2006
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
len_subset= IUC + OUC;            % subset length
bl_out    = net.bl_out;           % type of output neuron
int_bk    = net.int_bk;           % intensity of feedback
attenu    = net.attenu;           % attenuation ratio for the signal

%>>>>>>>>>>>> Testing Parameters Setting <<<<<<<<<<<<<<<<<<<
S_point = 4000;            % starting point of testing data
testNum = 3000;            % number of testing data
X(1,:)  = zeros(1,HUC);    % initial reservoir state
t       = [S_point : S_point+len_subset-1]; 
y0      = rand(1,OUC)-0.5; % initial output

%>>>>>>>>>>>>>>>>>>>>>>>> Check parameter <<<<<<<<<<<<<<<<<<
if length(t) ~= len_subset, 
    error('Length of testing data subset and the network structure do not match');
end;

%>>>>>>>>>>>>>>>>> Testing Main Routine <<<<<<<<<<<<<<<<<<<<<
wb = waitbar(0, 'Echo State Network Testing in Progress...');
for i = (1:testNum),
    waitbar(i/testNum,wb)
    set(wb,'name',['Progress = ' sprintf('%2.1f',i/testNum*100) '%']);
    y = attenu*signal(t);                                   % generate testing data
    X(i+1,:) = hyperb((inWeights*y(1:end-OUC)' + ...
               drWeights*X(i,:)' + int_bk*bkWeights*y0')'); % update reservoir state
    if (bl_out == 1),
        Y(i+1,:) = ouWeights*X(i+1,:)';        % update output state - Linear output
    else
        Y(i+1,:) = hyperb(ouWeights*X(i+1,:)');  % update output state - nonlinear output
    end;
    
    % update state for next iteration and output
    original_out(i) = (1/attenu)*y(end-OUC+1:end);   % original output
    net_out(i)      = (1/attenu)*Y(i+1,:);           % network output
    error(i)        = net_out(i) - original_out(i);  % errors
    y0 = Y(i+1,:);                                   % store the output for next calculation
    t  = t + 1;                                      % Move one-step forward
end;
close(wb);

%>>>>>>>>>>>>>>>>>> Plotting <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
subplot(211);
plot([S_point+1:S_point+testNum],original_out,'b',[S_point+1:S_point+testNum],net_out,'r');
hold on; grid on;
legend('Original sequence','Network output');
xlabel('time'); ylabel('Amplitude');
subplot(212);
plot([S_point+1:S_point+testNum],error,'b'); 
hold on; grid on;
xlabel('Time'); ylabel('Output error');
RMSE = sqrt(mean((net_out(1:end) - original_out(1:end)).^2))