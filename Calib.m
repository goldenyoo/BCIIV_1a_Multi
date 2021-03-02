% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo
%
%    Last Modified: 2020_02_26
%
% ----------------------------------------------------------------------- %
function [P, X_train, Y_train] = Calib(answer,ref)

% Input parameters
data_label = string(answer(1,1));
m = double(string(answer(2,1))); % feature vector will have length (2m)
referencing = double(string(answer(5,1))); % Non(0), CAR(1), LAP(2)
f1 = double(string(answer(3,1)));
f2 = double(string(answer(4,1)));

% Load file

FILENAME = strcat('C:\Users\유승재\Desktop\Motor Imagery EEG data\BCICIV_1_mat\BCICIV_calib_ds1',data_label,'.mat');
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

FB = [f1 f2];


feature_V = [];

for fb = 1:size(FB,1)
    low_f = FB(fb,1);
    high_f = FB(fb,2);
    
    
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

for fb = 1:size(FB,1)
    
    cnt_c = filtered{fb};
    
    % Calculate spatial filter
    a = 0; b = 0;
    C_1 = zeros(size(cnt_c,1)); C_2 = zeros(size(cnt_c,1));
    C_0 = zeros(size(cnt_c,1));
    
    % Training only for training data set
    for i = 1:length(mrk.pos)
        
        % One trial data
        E = cnt_c(:,mrk.pos(1,i):mrk.pos(1,i)+chunk);
        
        % Covariance 연산
        C = E*E'/ trace( E*E');
        
        % According to its class, divide calculated covariance
        if mrk.y(1,i) == -1
            C_1 = C_1+C;
            a = a+1;
        else
            C_2 = C_2+C;
            b = b+1;
        end
        
        E = cnt_c(:,mrk.pos(1,i)+chunk+1:mrk.pos(1,i)+2*chunk);
        C = E*E'/ trace( E*E');
        C_0 = C_0 + C;
        
    end
    
    % Average covariance of each class
    C_1 = C_1/(a);
    C_2 = C_2/(b);
    C_0 = C_0/(a+b);
    
    % composite covariace
%     C_c = C_1 + C_2 + C_0;
   C_c = C_1 + C_2;
    
    % EVD for composite covariance
    [V, D] = eig(C_c);
    
    % sort eigen vector with descend manner
%         [d, ind] = sort(abs(diag(D)),'descend');
%         D_new = diag(d);
%         V_new = V(:,ind);
    
    % whitening transformation
    whiten_tf = V*D^(-0.5);
    W = whiten_tf';
    
    % Apply whitening to each averaged covariances
    S1 = W*C_1*W';
    S2 = W*C_2*W';
    S0 = W*C_0*W';
    
    % EVD for transformed covariance
    [U, phsi] = eig(S1,S1+S2+S0);
%     [U, phsi] = eig(S1,S1+S2);
    
    
    eig_values = diag(phsi);
    N = length(eig_values);
    score = max(eig_values', 1/(1+(eig_values./(1-eig_values))*(N-1)^(2)));
    
    [d, ind] = sort(score,'descend');
    U_new = U(:,ind);
    
    % Total Projection matrix,   Z = P'*X
    P = (U_new'*W)';
    
    X_train = [];
    Y_train = [];
    % Calculate feature vector
    for i = 1:length(mrk.pos)
        
        % One trial data
        E = cnt_c(:,mrk.pos(1,i):mrk.pos(1,i)+chunk);
        
        % Project data using calculated spatial filter
        Z = P'*E;
        
        % Feature vector
        tmp_ind = size(Z,1);
        Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
        
        var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
        fp = log(var_vector);
        
        X_train = [X_train; fp'];
        
        if mrk.y(1,i) == -1
            Y_train = [Y_train; -1];
        else
            Y_train = [Y_train; 1];
        end
        
%         % One trial data
%         E = cnt_c(:,mrk.pos(1,i)+chunk:mrk.pos(1,i)+2*chunk);
%         
%         % Project data using calculated spatial filter
%         Z = P'*E;
%         
%         % Feature vector
%         tmp_ind = size(Z,1);
%         Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
%         
%         var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
%         fp = log(var_vector);
%         
%         X_train = [X_train; fp'];
%         Y_train = [Y_train; 0];
    end
end


end
% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
