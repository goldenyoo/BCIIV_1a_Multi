% ----------------------------------------------------------------------- %
%    File_name: Eval.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_02_10                           
%                                                            
 % ----------------------------------------------------------------------- %
function [predictions] = Eval(answer,interest_freq_band,interest_P, training_data,training_label,ref)
data_label = string(answer(1,1));   
m = double(string(answer(2,1))); % feature vector will have length (2m)
referencing = double(string(answer(3,1))); % Non(0), CAR(1), LAP(2)


%% 
% Load file

FILENAME = strcat('C:\Users\유승재\Desktop\Motor Imagery EEG data\BCICIV_1_mat\BCICIV_eval_ds1',data_label,'.mat');
chunk = 400;
fs = 100;

load(FILENAME);

% Data rescale
cnt= 0.1*double(cnt);
cnt = cnt';
%% Preprocessing
if referencing ~= 0
    %%% Calculate differential voltage
    for i = 1 : size(cnt,1)
        cnt(i,:) = cnt(i,:) - cnt(ref,:);
    end

    % common average
    if referencing == 1         
        cnt_y = cnt(3:55,:); % Exclude electrode (AF3, AF4, O1, O2, PO1, PO2)        
        Means = (1/size(cnt_y,1))*sum(cnt_y);
        for i = 1 : size(cnt_y,1)
            cnt_y(i,:) = cnt_y(i,:) - Means; % CAR
        end
     % LAP   
    elseif referencing == 2 
        cnt_n = myLAP(cnt,nfo); % Laplacian
        cnt_y = cnt_n(3:55,:); % Exclude electrode (AF3, AF4, O1, O2, PO1, PO2)
    end
else
        %%% Calculate differential voltage
    for i = 1 : size(cnt,1)
        cnt(i,:) = cnt(i,:) - cnt(ref,:);
    end
    
    cnt_y = cnt(3:55,:); % Exclude electrode (AF3, AF4, O1, O2, PO1, PO2)
end

clear cnt 

%% 
for fb = 1:size(interest_freq_band,1)
        low_f = interest_freq_band(fb,1);
        high_f = interest_freq_band(fb,2);

  bpFilt = designfilt('bandpassiir','SampleRate',fs,'PassbandFrequency1',low_f, ...
    'PassbandFrequency2',high_f,'StopbandFrequency1',low_f-2,'StopbandFrequency2',high_f+2, ...
    'StopbandAttenuation1',40,'StopbandAttenuation2',40, 'PassbandRipple',1,'DesignMethod','cheby2');

        
        % Apply BPF
        for i = 1:size(cnt_y,1)
            cnt_x(i,:) = filtfilt(bpFilt, cnt_y(i,:));            
        end
        filtered{fb} = cnt_x;
end

%% 
predictions = [];
iter = 1;
while iter + chunk <= size(filtered{1},2)
    for fb = 1:size(interest_freq_band,1)
        cnt_c = filtered{fb};
        
        E = cnt_c(:, iter:iter+chunk);
        Z = interest_P{fb}'*E;
        
        % Feature vector
        tmp_ind = size(Z,1);
        Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
        
        var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
        fp = log(var_vector);
        
        evaluation_trial(1,1+2*m*(fb-1):2*m*fb) = fp;
    end
    
    % Run classifier
    
    [prediction] = myClassifier(evaluation_trial,training_data,training_label);    
    predictions = [predictions prediction];
    
    iter = iter + 1;
end
predictions = [predictions repmat(0,1,chunk)];

end
% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
