% ----------------------------------------------------------------------- %
%    File_name: multiCSP.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_03_26                           
%                                                            
 % ----------------------------------------------------------------------- %

%% Get input parameter from user
clc
close all
clear all

% Ask user for input parameters
prompt = {'Data label: ', 'Feature vector length: ','Low freq: ','High freq: '};
dlgtitle = 'Input';
dims = [1 50];
definput = {'a', '2','8','30'};
answer = inputdlg(prompt,dlgtitle,dims,definput);
% Error detection
if isempty(answer), error("Not enough input parameters."); end

%% Conditions
% Rereferencing method 
ref_method = [0 1 2]; % Non(0), CAR(1), LAP(2)git 

% Reference electrode number
ref = 33;        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Change

%% CSP 
for i = 1:length(ref_method)
    
    answer(5,1) = {ref_method(i)};
    
    [P,V_train,X_train] = Calib(answer,ref);
   
    [predictions] = Eval(answer,P,V_train,X_train); 
    [score, total, mse, tmp,total_0, total_12, score_0, score_12,conf_mat, kappa] = Check(answer,predictions);
    
    fprintf('\nData_Label: %s\n',string(answer(1,1)));
    fprintf('Re-referencing: %d\n',ref_method(i));
    
    
    fprintf('MSE: %f\n',mse);
       
    
    fprintf('\nTotal: %d / %d\n',score,total);
    fprintf("SCORE: %f\n",100*score/total); 
    
    fprintf('\nTotal_0: %d / %d\n',score_0,total_0);
    fprintf("SCORE: %f\n",100*score_0/total_0); 
    
    fprintf('\nTotal_12: %d / %d\n',score_12,total_12);
    fprintf("SCORE: %f\n",100*score_12/total_12); 
    
    fprintf("\nKappa value: %f\n\n", kappa);
    
    disp(conf_mat);
end

% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
