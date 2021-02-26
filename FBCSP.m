% ----------------------------------------------------------------------- %
%    File_name: FBCSP.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_02_26                           
%                                                            
 % ----------------------------------------------------------------------- %

%% Get input parameter from user
close all
clear all

% Ask user for input parameters
prompt = {'Data label: ', 'Feature vector length: ','Low freq: ','High freq: '};
dlgtitle = 'Input';
dims = [1 50];
definput = {'a', '2','8','12'};
answer = inputdlg(prompt,dlgtitle,dims,definput);
% Error detection
if isempty(answer), error("Not enough input parameters."); end

%% Conditions
% Rereferencing method 
ref_method = [0 1 2]; % Non(0), CAR(1), LAP(2)

% Reference electrode number
ref = 29;        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Change

%% CSP 
for i = 1:length(ref_method)
    
    answer(5,1) = {ref_method(i)};
    
    [P, X_train, Y_train] = Calib(answer,ref);
   
    [predictions] = Eval(answer, P, X_train, Y_train); 
    [score, total, mse, tmp] = Check(answer,predictions);
    
    fprintf('\nData_Label: %s\n',string(answer(1,1)));
    fprintf('Re-referencing: %d\n',ref_method(i));
    fprintf('%d / %d\n',score,total);
    fprintf('MSE: %f\n',mse);
    fprintf("SCORE: %f\n",100*score/total);    
end

% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
