% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo
%
%    Last Modified: 2020_03_26
%
% ----------------------------------------------------------------------- %
function [P,V_train,X_train] = Calib(answer,ref)

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
        cnt_y(i,:) = filtfilt(bpFilt, cnt_y(i,:));
    end
    filtered{fb} = cnt_y;
end


%%
cnt_c = filtered{1};

% Calculate spatial filter
a = 0; b = 0;
C_1 = zeros(size(cnt_c,1)); C_2 = zeros(size(cnt_c,1));
C_0 = zeros(size(cnt_c,1));

% Training only for training data set
for i = 1:length(mrk.pos)
   
        E = cnt_c(:,mrk.pos(1,i)+100:mrk.pos(1,i)+400);
        
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
        E = cnt_c(:,mrk.pos(1,i)+400:mrk.pos(1,i)+800);
        C = E*E'/ trace( E*E');
        C_0 = C_0 + C;     
    
end

% Average covariance of each class
C_0 = C_0/(a+b);
C_1 = C_1/(a);
C_2 = C_2/(b);



%%%%%%%%%%%%%%%%%%%%%%%% P_01 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% composite covariace
for i = 1:4
    if i ==1
        C_a = C_0;
        C_b = C_1;
    elseif i == 2
        C_a = C_0;
        C_b = C_2;
    elseif i ==3
        C_a = C_1;
        C_b = C_2;
    else
        C_a = C_0;
        C_b = (a*C_1+b*C_2)/(a+b);
    end
    
    C_c = C_a + C_b;
    
    % EVD for composite covariance
    [V, D] = eig(C_c);
    
    if ~isempty(find(diag(D)<0)), error("Nagative eigen value"); end
    
    % sort eigen vector with descend manner
    [d, ind] = sort(abs(diag(D)),'descend');
    D_new = diag(d);
    V_new = V(:,ind);
    
    % whitening transformation
    whiten_tf = V_new*D_new^(-0.5);
    W = whiten_tf';
    
    % Apply whitening to each averaged covariances
    Sa = W*C_a*W';
    Sb = W*C_b*W';
    
    
    % EVD for transformed covariance
    [U, phsi] = eig(Sb,Sa+Sb);
    if ~isempty(find(diag(phsi)<0)), error("Nagative eigen value"); end
  
    [d, ind] = sort(abs(diag(phsi)),'descend');
    phsi_new = diag(d);
    U_new = U(:,ind);
    
    P{i} = (U_new'*W)';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% f1 = figure;
% f2 = figure;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 1:4
    X_train_1 = [];
    X_train_2 = [];
    X_train_0 = [];

    for i = 1:length(mrk.pos)
        %     fprintf("%d\n",i);
        if mrk.y(1,i) == -1
            
            E = cnt_c(:,mrk.pos(1,i)+100:mrk.pos(1,i)+400);
            
            Z = P{k}'*E;
            % Feature vector
            tmp_ind = size(Z,1);
            Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
                        
            var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
            fp_01 = log(var_vector);
           
            X_train_1 = [X_train_1 fp_01];
            
            
        elseif mrk.y(1,i) == 1
            
            E = cnt_c(:,mrk.pos(1,i)+100:mrk.pos(1,i)+400);
            
            Z = P{k}'*E;
            % Feature vector
            tmp_ind = size(Z,1);
            Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
            
            
            var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
            fp_02 = log(var_vector);         
                        
            X_train_2 = [X_train_2 fp_02];
            
        end
        
        E = cnt_c(:,mrk.pos(1,i)+400:mrk.pos(1,i)+800);
        
        Z = P{k}'*E;
        % Feature vector
        tmp_ind = size(Z,1);
        Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];    
        
        var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
        fp_01 = log(var_vector);    
                
        X_train_0 = [X_train_0 fp_01];
    end
    if k == 1
        M_train{k,1} = mean(X_train_0,2);
        M_train{k,2} = mean(X_train_1,2);
        Q_train{k,1} = cov(X_train_0');
        Q_train{k,2} = cov(X_train_1');
        X_train{1,1} = X_train_0;
        X_train{1,2} = X_train_1;
    elseif k ==2
        M_train{k,1} = mean(X_train_0,2);
        M_train{k,2} = mean(X_train_2,2);
        Q_train{k,1} = cov(X_train_0');
        Q_train{k,2} = cov(X_train_2');
        X_train{2,1} = X_train_0;
        X_train{2,2} = X_train_2;
    elseif k == 3
        M_train{k,1} = mean(X_train_1,2);
        M_train{k,2} = mean(X_train_2,2);
        Q_train{k,1} = cov(X_train_1');
        Q_train{k,2} = cov(X_train_2');
        X_train{3,1} = X_train_1;
        X_train{3,2} = X_train_2;
    else
        M_train{k,1} = mean(X_train_0,2);
        M_train{k,2} = mean([X_train_1 X_train_2],2);
        Q_train{k,1} = cov(X_train_0');
        Q_train{k,2} = cov([X_train_1 X_train_2]');
        X_train{4,1} = X_train_0;
        X_train{4,2} = [X_train_1 X_train_2];               
    end
    
end


for k = 1:4
    
    X1 = X_train{k,1};
    X2 = X_train{k,2};
    
    M_1 = M_train{k,1};
    M_2 = M_train{k,2};
    C_1 = Q_train{k,1};
    C_2 = Q_train{k,2};
    
    Sb = (M_1-M_2)*(M_1-M_2)';
    Sw = C_1+C_2;
    tmp = pinv(Sb)*Sw;
    
    [V, D] = eig(tmp);
    [d, ind] = sort(abs(diag(D)),'descend');
    D_new = diag(d);
    V_new = V(:,ind);
    
    V_train{k,1} = V_new(:,1);     
  
 
end

end
% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
