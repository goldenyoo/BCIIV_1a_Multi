function [score, total] = Check(answer,predictions)
data_label = string(answer(1,1));   
m = double(string(answer(2,1))); % feature vector will have length (2m)
referencing = double(string(answer(3,1))); % Non(0), CAR(1), LAP(2)



%% Call true label
FILENAME = strcat('C:\Users\유승재\Desktop\Motor Imagery EEG data\true_labels\BCICIV_eval_ds1',data_label,'_1000Hz_true_y.mat');
load(FILENAME);

true_y = downsample(true_y,10);

total = 0;
score = 0;
for i = 1:size(true_y,1)
    if isnan(true_y(i))        
        continue
    elseif true_y(i)==0
        total = total + 1;
        if predictions(i) == 0
            score = score + 1;
        end
    else
        total = total + 1;
        if predictions(i) ~=0
            score = score + 1;
        end
    end
end
tmp = [true_y';predictions];

end