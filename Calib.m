% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo
%
%    Last Modified: 2020_02_26
%
% ----------------------------------------------------------------------- %
function [P_01,P_02,P_012,P_12,M_r1,M_1, M_r2,M_2, M_r, M_12,M_c1,M_c2, Q_r1, Q_r2, Q_1,Q_2,Q_r,Q_12,Q_c1,Q_c2] = Calib(answer,ref)

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
cnt_c = filtered{1};

% Calculate spatial filter
a = 0; b = 0;
C_1 = zeros(size(cnt_c,1)); C_2 = zeros(size(cnt_c,1));
C_0 = zeros(size(cnt_c,1));

% Training only for training data set
for i = 1:length(mrk.pos)
    h = 100;
    while h+150 <= 400
        E = cnt_c(:,mrk.pos(1,i)+h:mrk.pos(1,i)+h+150);
        
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
        E = cnt_c(:,mrk.pos(1,i)+h+400:mrk.pos(1,i)+h+550);
        C = E*E'/ trace( E*E');
        C_0 = C_0 + C;
        
        h = h+1;
    end    
    
end

% Average covariance of each class
C_1 = C_1/(a);
C_2 = C_2/(b);
C_0 = C_0/(a+b);


%%%%%%%%%%%%%%%%%%%%%%%% P_01 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% composite covariace
C_c = C_0 + C_1 ;

% EVD for composite covariance
[V, D] = eig(C_c);

% sort eigen vector with descend manner
[d, ind] = sort(abs(diag(D)),'descend');
D_new = diag(d);
V_new = V(:,ind);

% whitening transformation
whiten_tf = V*D_new^(-0.5);
W = whiten_tf';

% Apply whitening to each averaged covariances
S0 = W*C_0*W';
S1 = W*C_1*W';

% EVD for transformed covariance
[U, phsi] = eig(S0,S1);

[d, ind] = sort(abs(diag(phsi)),'descend');
U_new = U(:,ind);

% Total Projection matrix,   Z = P'*X
P_01 = (U_new'*W)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% P_02 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% composite covariace
C_c = C_0 + C_2 ;

% EVD for composite covariance
[V, D] = eig(C_c);

% sort eigen vector with descend manner
[d, ind] = sort(abs(diag(D)),'descend');
D_new = diag(d);
V_new = V(:,ind);

% whitening transformation
whiten_tf = V*D_new^(-0.5);
W = whiten_tf';

% Apply whitening to each averaged covariances
S0 = W*C_0*W';
S2 = W*C_2*W';

% EVD for transformed covariance
[U, phsi] = eig(S0,S2);

[d, ind] = sort(abs(diag(phsi)),'descend');
U_new = U(:,ind);

% Total Projection matrix,   Z = P'*X
P_02 = (U_new'*W)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% P_012 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% composite covariace
C_c = C_0 + (a*C_1+b*C_2)/(a+b) ;

% EVD for composite covariance
[V, D] = eig(C_c);

% sort eigen vector with descend manner
[d, ind] = sort(abs(diag(D)),'descend');
D_new = diag(d);
V_new = V(:,ind);

% whitening transformation
whiten_tf = V*D_new^(-0.5);
W = whiten_tf';

% Apply whitening to each averaged covariances
S0 = W*C_0*W';
S12 = W*((a*C_1+b*C_2)/(a+b))*W';

% EVD for transformed covariance
[U, phsi] = eig(S0,S12);

[d, ind] = sort(abs(diag(phsi)),'descend');
U_new = U(:,ind);

% Total Projection matrix,   Z = P'*X
P_012 = (U_new'*W)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% P_01 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% composite covariace
C_c = C_1 + C_2 ;

% EVD for composite covariance
[V, D] = eig(C_c);

% sort eigen vector with descend manner
[d, ind] = sort(abs(diag(D)),'descend');
D_new = diag(d);
V_new = V(:,ind);

% whitening transformation
whiten_tf = V*D_new^(-0.5);
W = whiten_tf';

% Apply whitening to each averaged covariances
S1 = W*C_1*W';
S2 = W*C_2*W';

% EVD for transformed covariance
[U, phsi] = eig(S1,S2);

[d, ind] = sort(abs(diag(phsi)),'descend');
U_new = U(:,ind);

% Total Projection matrix,   Z = P'*X
P_12 = (U_new'*W)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_train_0 = [];
X_train_12 = [];
for i = 1:length(mrk.pos)
    h = 100;
    while h+150 <= 400
        E = cnt_c(:,mrk.pos(1,i)+h:mrk.pos(1,i)+h+150);
        
        Z = P_012'*E;
        % Feature vector
        tmp_ind = size(Z,1);
        Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
        var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
        fp_012 = log(var_vector);
        
        X_train_12 = [X_train_12 fp_012];
        
        % One trial data
        E = cnt_c(:,mrk.pos(1,i)+h+400:mrk.pos(1,i)+h+550);
        
        Z = P_012'*E;
        % Feature vector
        tmp_ind = size(Z,1);
        Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
        var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
        fp_012 = log(var_vector);
        
        X_train_0 = [X_train_0 fp_012];
        
        h = h+1;
    end 
end
M_r = mean(X_train_0,2);
M_12 = mean(X_train_12,2);

Q_r = zeros(2*m);
for i = 1:length(X_train_0)
    tmp = (X_train_0(:,i) - M_r)*(X_train_0(:,i) - M_r)';
    Q_r = Q_r + tmp;
end
Q_r = (1/(length(X_train_0)-1))*Q_r;

Q_12 = zeros(2*m);
for i = 1:length(X_train_12)
    tmp = (X_train_12(:,i) - M_12)*(X_train_12(:,i) - M_12)';
    Q_12 = Q_12 + tmp;
end
Q_12 = (1/(length(X_train_12)-1))*Q_12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_train_0 = [];
X_train_1 = [];
for i = 1:length(mrk.pos)
    
    if mrk.y(1,i) == -1
        h = 100;
        while h+150 <= 400
            E = cnt_c(:,mrk.pos(1,i)+h:mrk.pos(1,i)+h+150);
            
            Z = P_01'*E;
            % Feature vector
            tmp_ind = size(Z,1);
            Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
            var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
            fp_01 = log(var_vector);
            
            X_train_1 = [X_train_1 fp_01];
            
            h = h + 1;
        end
    end
    h = 100;
    while h+150 <= 400
        E = cnt_c(:,mrk.pos(1,i)+h+400:mrk.pos(1,i)+h+550);
        
        Z = P_01'*E;
        % Feature vector
        tmp_ind = size(Z,1);
        Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
        var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
        fp_01 = log(var_vector);
        
        X_train_0 = [X_train_0 fp_01];
        
        h = h+1;
    end
      
end
M_r1 = mean(X_train_0,2);
M_1 = mean(X_train_1,2);

Q_r1 = zeros(2*m);
for i = 1:length(X_train_0)
    tmp = (X_train_0(:,i) - M_r1)*(X_train_0(:,i) - M_r1)';
    Q_r1 = Q_r1 + tmp;
end
Q_r1 = (1/(length(X_train_0)-1))*Q_r1;

Q_1 = zeros(2*m);
for i = 1:length(X_train_1)
    tmp = (X_train_1(:,i) - M_1)*(X_train_1(:,i) - M_1)';
    Q_1 = Q_1 + tmp;
end
Q_1 = (1/(length(X_train_1)-1))*Q_1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_train_0 = [];
X_train_2 = [];
for i = 1:length(mrk.pos)
    
    if mrk.y(1,i) == 1
         h = 100;
        while h+150 <= 400
            E = cnt_c(:,mrk.pos(1,i)+h:mrk.pos(1,i)+h+150);
            
            Z = P_02'*E;
            % Feature vector
            tmp_ind = size(Z,1);
            Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
            var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
            fp_02 = log(var_vector);
            
            X_train_2 = [X_train_2 fp_02];
            
            h = h+1;
        end
    end
    h = 100;
    while h+150 <= 400
        E = cnt_c(:,mrk.pos(1,i)+h+400:mrk.pos(1,i)+h+550);
        
        Z = P_02'*E;
        % Feature vector
        tmp_ind = size(Z,1);
        Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
        var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
        fp_02 = log(var_vector);
        
        X_train_0 = [X_train_0 fp_02];
        
        h = h+1;
    end
      
end
M_r2 = mean(X_train_0,2);
M_2 = mean(X_train_2,2);

Q_r2 = zeros(2*m);
for i = 1:length(X_train_0)
    tmp = (X_train_0(:,i) - M_r2)*(X_train_0(:,i) - M_r2)';
    Q_r2 = Q_r2 + tmp;
end
Q_r2 = (1/(length(X_train_0)-1))*Q_r2;

Q_2 = zeros(2*m);
for i = 1:length(X_train_2)
    tmp = (X_train_2(:,i) - M_2)*(X_train_2(:,i) - M_2)';
    Q_2 = Q_2 + tmp;
end
Q_2 = (1/(length(X_train_2)-1))*Q_2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_train_1 = [];
X_train_2 = [];
for i = 1:length(mrk.pos)
    if mrk.y(1,i) == -1
        h = 100;
        while h+150 <= 400
            E = cnt_c(:,mrk.pos(1,i)+h:mrk.pos(1,i)+h+150);
            
            Z = P_12'*E;
            % Feature vector
            tmp_ind = size(Z,1);
            Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
            var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
            fp_1 = log(var_vector);
            
            X_train_1 = [X_train_1 fp_1];
            
            h = h+1;
        end
    else
         h = 100;
        while h+150 <= 400
            E = cnt_c(:,mrk.pos(1,i)+h:mrk.pos(1,i)+h+150);
            
            Z = P_12'*E;
            % Feature vector
            tmp_ind = size(Z,1);
            Z_reduce = [Z(1:m,:); Z(tmp_ind-(m-1):tmp_ind,:)];
            var_vector = diag(Z_reduce*Z_reduce')/trace(Z_reduce*Z_reduce');
            fp_2 = log(var_vector);
            
            X_train_2 = [X_train_2 fp_2];
            
            h = h+1;
        end
    end
    
      
end
M_c1 = mean(X_train_1,2);
M_c2 = mean(X_train_2,2);

Q_c1 = zeros(2*m);
for i = 1:length(X_train_1)
    tmp = (X_train_1(:,i) - M_c1)*(X_train_1(:,i) - M_c1)';
    Q_c1 = Q_c1 + tmp;
end
Q_c1 = (1/(length(X_train_1)-1))*Q_c1;

Q_c2 = zeros(2*m);
for i = 1:length(X_train_2)
    tmp = (X_train_2(:,i) - M_c2)*(X_train_2(:,i) - M_c2)';
    Q_c2 = Q_c2 + tmp;
end
Q_c2 = (1/(length(X_train_2)-1))*Q_c2;


end
% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
