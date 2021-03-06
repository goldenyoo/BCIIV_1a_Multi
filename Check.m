function [score, total ,mse, tmp, total_0, total_12, score_0, score_12,conf_mat, kappa] = Check(answer,predictions)
data_label = string(answer(1,1));   

%% Call true label
FILENAME = strcat('C:\Users\유승재\Desktop\Motor Imagery EEG data\true_labels\BCICIV_eval_ds1',data_label,'_1000Hz_true_y.mat');
load(FILENAME);


conf_mat = zeros(3,3);
true_y = downsample(true_y,10);

total = 0;
score = 0;
mse = 0;

score_0 = 0;
total_0 = 0;

score_12 = 0;
total_12 = 0;
for i = 1:size(true_y,1)
    if isnan(true_y(i))        
        continue
    elseif true_y(i)==0
        total = total + 1;
        total_0 = total_0 +1;
        if predictions(i) == 0
            score = score + 1;
            score_0 = score_0 + 1;
            conf_mat(1,1) = conf_mat(1,1) + 1; 
        else
            mse = mse + (true_y(i)-predictions(i))^2;
            if predictions(i) == -1
                conf_mat(1,2) = conf_mat(1,2) + 1;
            elseif predictions(i) == 1
                conf_mat(1,3) = conf_mat(1,3) + 1;
            end
        end
%     continue
        
    elseif true_y(i)== -1
        total = total + 1;
        total_12 = total_12 + 1;
        if predictions(i) == -1 
            score = score + 1;
            score_12 = score_12 +1;
            conf_mat(2,2) = conf_mat(2,2) + 1;
        else
            mse = mse + (true_y(i)-predictions(i))^2;
            if predictions(i) == 0
                conf_mat(2,1) = conf_mat(2,1) + 1;
            elseif predictions(i) == 1
                conf_mat(2,3) = conf_mat(2,3) + 1;
            end
        end
    elseif true_y(i)== 1
         total = total + 1;
         total_12 = total_12 + 1;
        if predictions(i) == 1 
            score = score + 1;
            score_12 = score_12 +1; 
            conf_mat(3,3) = conf_mat(3,3) + 1;
        else
            mse = mse + (true_y(i)-predictions(i))^2;
            if predictions(i) == 0
                conf_mat(3,1) = conf_mat(3,1) + 1;
            elseif predictions(i) == -1
                conf_mat(3,2) = conf_mat(3,2) + 1;
            end
        end
    end
end
mse = mse/total;

tmp = [true_y'; predictions];

N = sum(sum(conf_mat));
p0 = sum(diag(conf_mat))/N;
pe = (sum(conf_mat(:,1))*sum(conf_mat(1,:)) + sum(conf_mat(:,2))*sum(conf_mat(2,:)) + sum(conf_mat(:,3))*sum(conf_mat(3,:)))/N^2;

kappa = (p0-pe)/(1-pe);

end