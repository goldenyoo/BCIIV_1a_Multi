% ----------------------------------------------------------------------- %
%    File_name: Eval.m
%    Programmer: Seungjae Yoo
%
%    Last Modified: 2020_02_26
%
% ----------------------------------------------------------------------- %
function [predictions] = Eval(answer,P_01,P_02,P_12, X_train, Y_train)
% Input parameters
data_label = string(answer(1,1));
m = double(string(answer(2,1))); % feature vector will have length (2m)
referencing = double(string(answer(5,1))); % Non(0), CAR(1), LAP(2)
f1 = double(string(answer(3,1)));
f2 = double(string(answer(4,1)));

ref=29;

%%
% Load file

FILENAME = strcat('C:\Users\유승재\Desktop\Motor Imagery EEG data\BCICIV_1_mat\BCICIV_eval_ds1',data_label,'.mat');
chunk = 150;
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

low_f = f1;
high_f = f2;

bpFilt = designfilt('bandpassiir','SampleRate',fs,'PassbandFrequency1',low_f, ...
    'PassbandFrequency2',high_f,'StopbandFrequency1',low_f-2,'StopbandFrequency2',high_f+2, ...
    'StopbandAttenuation1',40,'StopbandAttenuation2',40, 'PassbandRipple',1,'DesignMethod','cheby2');


% Apply BPF
for i = 1:size(cnt_y,1)
    cnt_x(i,:) = filtfilt(bpFilt, cnt_y(i,:));
end
filtered{1} = cnt_x;


%% Train SVM
 Mdl = fitcecoc(X_train,Y_train);
%   Mdl = fitcsvm(X_train,Y_train);
%   Mdl = fitclinear(X_train,Y_train);

%%

predictions = [];
iter = 1;
while iter + chunk <= size(filtered{1},2)
    if rem(iter,100000)==0
        fprintf('%d / %d\n',iter,size(filtered{1},2)-chunk);
    end
    
    cnt_c = filtered{1};
    
    E = cnt_c(:, iter:iter+chunk-1);
    
    Z = P_01'*E;
    
    % Feature vector
    tmp_ind = size(Z,1);
    Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
    
    var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
    fp_01 = log(var_vector);
    
     Z = P_02'*E;
    
    % Feature vector
    tmp_ind = size(Z,1);
    Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
    
    var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
    fp_02 = log(var_vector);
    
     Z = P_12'*E;
    
    % Feature vector
    tmp_ind = size(Z,1);
    Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
    
    var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
    fp_12 = log(var_vector);
    
    fp = [fp_01' fp_02' fp_12'];
    
    evaluation_trial = fp;
    
    % Run classifier
    
    prediction = predict(Mdl,evaluation_trial);
    predictions = [predictions repmat(prediction,1,chunk*0.8)];
    
    iter = iter + chunk*0.8;
end
    predictions = [predictions repmat(0,1,size(filtered{1},2)- iter + 1)];

end
% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
