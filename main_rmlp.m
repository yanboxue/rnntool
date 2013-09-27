% Main function of RMLP Training;

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 11, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

% Globalize some variables
global NUM_EPOCH
global NUM_SUBSET
global LEN_SEQ
NUM_EPOCH    = 3;                 % number of epochs
NUM_SUBSET   = 300;                % number of subsets in training data
LEN_SEQ      = 4000;               % length of sequence for training

% Generate RMLP network for training
net = rmlp_net(99,10,10,1);

%======================= BPTT-DEKF =====================================
[dekf_net_trained, dekf_mse, dekf_mse_cross] = rmlp_train_bptt_dekf(net);
% Plot training result
figure;
subplot(211);
plot(1:NUM_EPOCH, dekf_mse,'r.-');
hold on; 
grid on; 
set(gca,'YScale','log');
legend('Output RMSE for training Data');
xlabel('Number of epoch');
ylabel('RMSE');
subplot(212);
plot(1:NUM_EPOCH, dekf_mse_cross,'bx-'); 
set(gca,'YScale','log');
legend('RMSE of cross validated data');
xlabel('Number of epoch');
ylabel('RMSE');
hold on;
grid on;
% Test trained RMLP network
[dekf_original_out,dekf_net_out,dekf_error]  = rmlp_test(dekf_net_trained,'N');

% %======================= BPTT-GEKF =====================================
% [gekf_net_trained, gekf_mse, gekf_mse_cross] = rmlp_train_bptt_gekf(net);
% % Plot training result
% figure;
% subplot(211);
% plot(1:NUM_EPOCH, gekf_mse,'r.-');
% hold on; 
% grid on; 
% set(gca,'YScale','log');
% legend('Output RMSE for training Data');
% xlabel('Number of epoch');
% ylabel('RMSE');
% subplot(212);
% plot(1:NUM_EPOCH, gekf_mse_cross,'bx-'); 
% set(gca,'YScale','log');
% legend('RMSE of cross validated data');
% xlabel('Number of epoch');
% ylabel('RMSE');
% hold on;
% grid on;
% % Test trained RMLP network
% [gekf_original_out,gekf_net_out,gekf_error]  = rmlp_test(gekf_net_trained,'N');

