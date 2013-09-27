function [net_trained, mse, mse_cross] = rmlp_train_bptt_gekf(net)
% RMLP_train_bptt_gekf - Train the RMLP using BPTT-GEKF
% where the first hidden layer is recurrent and the second one is not.
% Bias input is not considered.
% ==============================================
% net = rmlp_train_bptt_gekf(net, I_data, O_data)
% net         - network structure being trained
% net_trained - trained network
% mse         - RMSE of trained network
% mse_cross   - RMSE of cross-validated data

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 18, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network

% Globalize some variables
global NUM_EPOCH
global NUM_SUBSET
global LEN_SEQ

%>>>>>>>>> Obtain parameters from RMLP net <<<<<<<<<<<<<<<<<<<<
ANC  = net.numAllNeurons;
IUC  = net.numInputUnits;
OUC  = net.numOutputUnits;
HUC1 = net.numHiddenLayer1;
HUC2 = net.numHiddenLayer2;
num_weights   = net.numWeights;
num_groups    = ANC;
len_subset    = IUC + OUC;           % length of subset
weights_all   = [net.weights.value]; % get weights value
weights_group = [net.weights.dest];  % define the group that the weights belong to
% Divide the weights of RMLP net into group from #1 to #ANC
for i = (1:num_groups),
    weights(i).value  = weights_all(min(find(weights_group == i)) : max(find(weights_group == i)));
    weights(i).length = length(find(weights_group == i));
end;
%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

%>>>>>>>>>>>>>>>> Initialization of Training <<<<<<<<<<<<<<<<<<
num_Epoch  = NUM_EPOCH;             % number of epochs
num_subset = NUM_SUBSET;            % number of subsets in training data
len_seq    = LEN_SEQ;               % length of sequence for training
R = annealing(100,5,num_Epoch);     % anneal R from 100 to 5
Q = annealing(1E-2,1E-6,num_Epoch); % anneal Q from 1E-2 to 1E-6
learning_rate = annealing(1,1E-5,num_Epoch); % learning_rate;
n = 1;                             % a counter for plotting
m = 1;                             % a counter for cross-validation plotting
timeflag = cputime;                % a timer for saving the training time
start_point = ceil((len_seq-num_subset-len_subset+2)*rand(1,num_Epoch)); %starting point of training data
%>>>>>>>>>>>>>> End of training initialization >>>>>>>>>>>>>>>>

%>>>>>>>>> Main loop - Decoupled Extended Kalman Filter: DEKF <<<<<<<<<<<<<<
for k = (1:num_Epoch), 
   %>>>>>>>>>>>>>>>>>>> Generate training data <<<<<<<<<<<<<<<<<<<<<
    [I_data, T_data]  = seq_gen_rmlp(len_seq,len_subset,num_subset,start_point(k));
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
    %>>>>>>>>>>>> Set the waitbar - Initialization <<<<<<<<<<<<<<<<<<
    wb1 = waitbar(0, 'RMLP Neural Network Training (BPTT-GEKF) in Progress...');
    set(wb1,'name',['Epoch = ' sprintf('%2.1f',k)]);
    %>>>>>>>>>>>> Initialize some variables <<<<<<<<<<<<<<<<<<<<<<<
    X1_0 = zeros(1,HUC1);
    K_global = 0.01^(-1)*eye(num_weights);
    weights0 = zeros(HUC1,HUC1+IUC);
    %>>>>>>>>>>>>>>> End of initialization >>>>>>>>>>>>>>>>>>>>>>>>>
    
    %>>>>>>>>>>> Initialization of recurrent layer states <<<<<<<<<<<
    %Run the RMLP network for two times to get the initial recurrent
    %network states X1_1 and X1_2
    [X1_1 X2 out(1)] = rmlp_run(net,I_data(1,:),X1_0); 
    [X1_2 X2 out(2)] = rmlp_run(net,I_data(2,:),X1_1);
    %>>>>>>>>>>>>>> End of twice RMLP runnings >>>>>>>>>>>>>>>>>>>>>
    
    for j = (3:inpNum), % number of datasets
            %>>>>>>>>>>>>>>>>> Display Waitbar <<<<<<<<<<<<<<<<<<<<<<
            waitbar(j/inpNum,wb1)
            set(wb1,'name',['Epoch = ' sprintf('%2.1f', k) ',Progress = ' sprintf('%2.1f',j/inpNum*100) '%']);
            %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
            %>>>>>>>>>>>>>>>>>>> Initialization <<<<<<<<<<<<<<<<<<<<<
            temp1 = 0;     % a temporatory variable in Gamma
            weights1 = []; % weights from input to first hidden layer of dimension: HUC1 x (IUC + HUC1)
            weights2 = []; % weights from first to second hidden layer of dimension: HUC2 x HUC1
            weights3 = []; % weights from second hidden layer to output layer of dimension: OUC x HUC2
            C_global = []; % C for Global EKF
          
            % weights matrix between input layer and first hidden layer
            for i = (1:HUC1),
                weights1 = [weights1; weights(i).value];
            end;
            
            % weights matrix between first and second hidden layer
            for i = (HUC1+1:HUC1+HUC2),
                weights2 = [weights2; weights(i).value];
            end;
            
            % weights matrix between second hidden layer and output layer
            for i = (HUC1+HUC2+1:ANC),
                weights3 = [weights3; weights(i).value];
            end;
            %>>>>>>>>>>>>>> End of Initialization >>>>>>>>>>>>>>>>>>>>
            
            %>>>>>>>>> Forward running of RMLP network <<<<<<<<<<<<<<<
            [X1_3 X2 out(j)] = rmlp_run(net,I_data(j,:),X1_2);
            %>>>>>>>>>>>>>> End of Forward running >>>>>>>>>>>>>>>>>>>
                 
            %>>>>>>>>> Backward Propagation of Error <<<<<<<<<<<<<<<<<<
            %>>>>>>>>>>>>>> Jacobian Matrix C Calculation <<<<<<<<<<<<<
            % X2 (row vector): output of second hidden layer
            % output neuron is supposed to be linear
            C_global = [reshape(X2,1,OUC*HUC2), C_global];
            % X1_3 (row vector): output of first hidden layer at time t
            D1 = (weights3*diag(d_hyperb(weights2*X1_3')))'*X1_3;
            C_global = [reshape(D1',1,HUC1*HUC2),C_global];
            % X1_2 (row vector): output of first hidden layer at time t-1
            % [X1_2 I_data(t,:)]: input of first hidden layer at time t
            % X1_1 (row vector): output of first hidden layer at time t-2
            % [X1_1 I_data(t-1,:)]: input of first hidden layer at time t-1
            D2 = (weights3*diag(d_hyperb(weights2*X1_3'))*...
                 weights2*diag(d_hyperb(weights1*[X1_2 I_data(j,:)]')))'*[X1_2 I_data(j,:)];
            D2 = D2 + (weights3*diag(d_hyperb(weights2*X1_3')) * ...
                 weights2*diag(d_hyperb(weights1*[X1_2 I_data(j,:)]'))* ...
                 weights1(:,1:HUC1)*diag(d_hyperb(weights0*[X1_1 I_data(j-1,:)]')))' *[X1_1 I_data(j-1,:)]; 
            C_global = [reshape(D2',1,(HUC1+IUC)*HUC2),C_global];
            %>>>>>>>>>>>>>>>>>>>>> End of Jacobian >>>>>>>>>>>>>>>>>>>>>>
            

            %>>>>>>>> Decoupled Extended Kalman Filter <<<<<<<<<<<<<<<<<<<
            alpha = T_data(j) - out(j);  % innovation of output   
            temp1 = C_global*K_global*C_global'; % about 0.2 seconds for this step
            Gamma =  inv(temp1+R(k));            
            G_global = K_global*C_global'*Gamma; 
            % Update the weights only if the innovation is larger than
            % a thereshould
            if abs(alpha) > 5E-2,
                weights_all = weights_all + learning_rate(k)*(G_global*alpha)'; 
            end;
            % Re-calculte the Ricatti equation - Most time-consuming step
            K_global = K_global - G_global*C_global*K_global + Q(k)*eye(num_weights);
            %>>>>>>>>>>>>>>>>>>>> End of DEKF >>>>>>>>>>>>>>>>>>>>>>>>>>
            
            %>>>>>>>>> Update the weights of the RMLP net <<<<<<<<<<<<<<<
            for i = (1:num_weights),
                net.weights(i).value = weights_all(i); 
            end;
            %>>>>>>>>>>>>>> End of weights updating <<<<<<<<<<<<<<<<<<<<<
            
            %>>>>>>>>>> Redivide the weights into group <<<<<<<<<<<<<<<<<
            for i = (1:num_groups),
                weights(i).value  = weights_all(min(find(weights_group == i)) : max(find(weights_group == i)));
            end;
            %>>>>>>>>>>>>>>>> End of weights division <<<<<<<<<<<<<<<<<<<
            
            % Recurrent states replacement
            X1_1 = X1_2;
            X1_2 = X1_3;
            weights0 = weights1;
    end;
    %>>>>>>>>>>>>>>>>>>> End of One Epoch <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    close(wb1);              % close the waitbar
   
    %>>>>>>>>>>>>>>>>>>>> Calculate RMSE  <<<<<<<<<<<<<<<<<<<<<<<<<
    mse(k) = sqrt(mean((out(1:end) - T_data(1:end)').^2));
    if mse(k) < 1E-5, break; end;
    fprintf('Epoch: %d, RMSE: %f\n', k, mse(k));
    mse_cross(k) = cross_validation(net,len_subset);
    n = n+1;
    fprintf('Epoch: %d, Output RMSE: %f, Cross-validated RMSE: %f\n', k, mse(k), mse_cross(k));
end;
%>>>>>>>>>>>>>>>>>>>>>>>> End of Main Loop <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
net_trained = net;
timeflag = cputime - timeflag;
fprintf('Training accomplished and the total time-comsuming is %2.2f hours',timeflag/3600);