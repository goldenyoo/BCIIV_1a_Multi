% ----------------------------------------------------------------------- %
%    File_name: FBCSP.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_02_18                            
%                                                            
 % ----------------------------------------------------------------------- %

%% Get input parameter from user
close all
clear all

% Ask user for input parameters
prompt = {'Data label: ', 'Feature vector length: '};
dlgtitle = 'Input';
dims = [1 50];
definput = {'a', '2'};
answer = inputdlg(prompt,dlgtitle,dims,definput);
% Error detection
if isempty(answer), error("Not enough input parameters."); end

%% Conditions
% Rereferencing method 
ref_method = [2]; % Non(0), CAR(1), LAP(2)



% Reference electrode number
ref = 29;        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Change

%% CSP 
for i = 1:length(ref_method)
    
    answer(3,1) = {ref_method(i)};
    
    fprintf('Data_Label: %s\n',string(answer(1,1)));
    fprintf('Re-referencing: %d\n',ref_method(i));
    
    [interest_freq_band,interest_P, training_data,training_label] = Calib(answer,ref);
    
    for k = 1:size(interest_freq_band,1)
        fprintf('Filter bank: %d %d\n',interest_freq_band(k,1),interest_freq_band(k,2));
    end
    
    [predictions] = Eval(answer,interest_freq_band,interest_P, training_data,training_label,ref);
    [score, total] = Check(answer,predictions);
    fprintf('%d / %d\n',score,total);
end

% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
