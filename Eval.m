% ----------------------------------------------------------------------- %
%    File_name: Eval.m
%    Programmer: Seungjae Yoo
%
%    Last Modified: 2020_02_26
%
% ----------------------------------------------------------------------- %
function [predictions] = Eval(answer,P,M_train,Q_train)
% Input parameters
data_label = string(answer(1,1));
m = double(string(answer(2,1))); % feature vector will have length (2m)
referencing = double(string(answer(5,1))); % Non(0), CAR(1), LAP(2)
f1 = double(string(answer(3,1)));
f2 = double(string(answer(4,1)));

ref=33;

%%
% Load file

FILENAME = strcat('C:\Users\유승재\Desktop\Motor Imagery EEG data\BCICIV_1_mat\BCICIV_eval_ds1',data_label,'.mat');

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
        cnt_y = cnt; % Exclude electrode (AF3, AF4, O1, O2, PO1, PO2)
        Means = (1/size(cnt,1))*sum(cnt);
        for i = 1 : size(cnt_y,1)
            cnt_y(i,:) = cnt_y(i,:) - Means; % CAR
        end
        cnt_y = cnt_y([27 29 31 44 46 50 52 54],:);
        % LAP
    elseif referencing == 2
        cnt_n = myLAP(cnt,nfo); % Laplacian
        cnt_y = cnt_n([27 29 31 44 46 50 52 54],:); % Exclude electrode (AF3, AF4, O1, O2, PO1, PO2)
    end
else
    %%% Calculate differential voltage
    for i = 1 : size(cnt,1)
        cnt(i,:) = cnt(i,:) - cnt(ref,:);
    end
    
    cnt_y = cnt([27 29 31 44 46 50 52 54],:); % Exclude electrode (AF3, AF4, O1, O2, PO1, PO2)
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
FILENAME = strcat('C:\Users\유승재\Desktop\Motor Imagery EEG data\true_labels\BCICIV_eval_ds1',data_label,'_1000Hz_true_y.mat');
load(FILENAME);

true_y = downsample(true_y,10);

%%

predictions = [];
iter = 1;

chunk = 150;

predictions = zeros(1,size(filtered{1},2));

while iter + chunk <= size(filtered{1},2)
    
    cnt_c = filtered{1};
    
    E = cnt_c(:, iter:iter+chunk-1);
    
    for k = 1:3
        Z = P{k}'*E;
        % Feature vector
        tmp_ind = size(Z,1);
        Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
        var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
        fp(:,k) = log(var_vector);
    end
    
    [tt1 prediction1] = myClassifier2(fp(:,1),M_train{1,1},M_train{1,2},Q_train{1,1},Q_train{1,2},1); %%%  0 vs -1
    [tt2 prediction2] = myClassifier2(fp(:,2),M_train{2,1},M_train{2,2},Q_train{2,1},Q_train{2,2},2); %%% 0 vs +1
    [tt3 prediction3] = myClassifier2(fp(:,3),M_train{3,1},M_train{3,2},Q_train{3,1},Q_train{3,2},3); %%% -1 vs +1
    
    prediction0 = true_y(iter);
    

    if prediction2 == -10 && prediction3 == -10  
        prediction = 0;
    elseif prediction3 == 1
        prediction = 1;
    elseif prediction2 == -1
        prediction = -1;
    end
    
    for tmpp = iter:iter+chunk-1
%         predictions(:,tmpp) = [prediction0; prediction1; prediction2; prediction3];        
        predictions(:,tmpp) = prediction;
    end
    iter = iter + chunk*0.8;
end


end
% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
