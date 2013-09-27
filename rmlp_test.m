function [original_out,net_out,error] = rmlp_test(net_trained,RT_plot)
% [original_out,net_out,error] = rmlp_test(net_trained,RT_plot)
% Testing the performance of trained RMLP network.
% where:
% ========================================================
% Inputs include:
% net_trained  -    the trained network used for testing 
% RT_plot      -    real-time plot option: 'Y' or 'N'
% ========================================================
% Outputs include:
% original_out -    original output of the test sequence
% net_out      -    network output
% error        -    error of the testing data (= net_out - original_out)

%%%% Author: Yanbo Xue & Le Yang
%%%% ECE, McMaster University
%%%% yxue@grads.mcmaster.ca; yangl7@psychology.mcmaster.ca
%%%% May 18, 2006
%%%% This is a joint work by Yanbo and Le
%%%% For Project of Course of Dr. Haykin: Neural Network


%>>>>>>>>>>>>>>>>>>>>> Initialization <<<<<<<<<<<<<<<<<<<<<<<<
HUC1 = net_trained.numHiddenLayer1; % Number of first hidden layer neurons
IUC  = net_trained.numInputUnits;   % Number of input layer neurons (linear)
OUC  = net_trained.numOutputUnits;  % Number of output layer neurons
X1   = zeros(1,HUC1);               % Initial state of the recurrent input
len_subset   = IUC+OUC;             % Length of the subset data
num_testdata = 2000;                % Number of testing data
S_point      = 6000;   %Starting point of the testing data generation
                       %The first testing data should be at S_point+len_subset-1
TS_point= S_point + len_subset -1;  % Starting point of testing data output
TE_point= TS_point + num_testdata -1;%Ending point of testing data
t = [S_point : S_point+len_subset-1];% Vector t for generating testing data

%>>>>>> Set up waitbar to show the process of tesing <<<<<<<<<
wb2 = waitbar(0, 'RMLP Neural Network Testing in Progress...');

%>>>>>>>>>>>>>>>>>> Main testing loop <<<<<<<<<<<<<<<<<<<<<<<<<
for i = (TS_point : TE_point),
    y = signal(t); % Testing function  
    [X11, X2, out] = rmlp_run(net_trained,y(1:len_subset-1),X1);
    
    if (RT_plot == 'Y'),    % 'YES' for real-time plotting 
        if (i>=1+TS_point), % Draw a line from current data to former data when i>=2
            % Draw first subfigure for original output vs network output
            subplot(211);
            plot(i,y(len_subset),'b',i,out,'r'); %Draw output data pixels
            line([i-1,i],[y0,y(len_subset)],'Color','b');  
            line([i-1,i],[out0,out],'Color','r');
            hold on; grid on;
            legend('Original sequence','Network output');
            xlabel('Time'); ylabel('Amplitude');
            % Rolling display windows when the data number > 100
            if (i>100+TS_point),
                xlim([i-100,i]); 
            end;
            
            % Draw second subfigure for original output vs network output
            subplot(212);
            plot(i,out-y(len_subset),'b'); hold on; grid on;
            line([i-1,i],[error0,out-y(len_subset)],'Color','b');
            xlabel('Time'); ylabel('Output error');
            % Rolling display windows when the data number > 100
            if (i>100+TS_point),
                xlim([i-100,i]);
            end;
        end;
    end;
    
    y0     = y(len_subset);    %Save as former original output
    out0   = out;              %Save as former network output
    error0 = out-y(len_subset);%Save as former error

    % For final plotting with option of 'Non-real-time drawing'
    original_out(i-TS_point+1) = y0;
    net_out(i-TS_point+1)      = out0;
    error(i-TS_point+1)        = error0;

    t = t + 1;  % Move one-step forward
    X1 = X11;   % Save state for next recurrent network input

    %>>>>>>>>>>>>>> Update Waitbar <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    waitbar((i-TS_point)/(TE_point-TS_point),wb2);
    set(wb2,'name',['Progress = ' sprintf('%2.1f',(i-TS_point)/(TE_point-TS_point)*100) '%']);
end;
close(wb2);     % Close waitbar

%>>>>>>>>>>> Final Plotting For Overall Data <<<<<<<<<<<<<<<<<<<<<<<
if (RT_plot ~= 'Y'),
    figure;
    subplot(211);
    plot([TS_point:TE_point],original_out,'b',[TS_point:TE_point],net_out,'r');
    hold on; grid on;
    legend('Original sequence','Network output');
    xlabel('time'); ylabel('Amplitude');
    subplot(212);
    plot([TS_point:TE_point],error,'b'); 
    hold on; grid on;
    xlabel('Time'); ylabel('Output error');
end;

%>>>>>>> Harness the might-happening warnings <<<<<<<<<<<<<<<<<<<<<
warning off MATLAB:colon:operandsNotRealScalar;