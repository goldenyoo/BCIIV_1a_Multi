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
    
    [P_01,P_02,P_012,P_12,M_r1,M_1, M_r2,M_2, M_r, M_12,M_c1,M_c2, Q_r1, Q_r2, Q_1,Q_2,Q_r,Q_12,Q_c1,Q_c2] = Calib(answer,ref);
   
    [predictions] = Eval(answer, P_01,P_02,P_012,P_12,M_r1,M_1, M_r2,M_2, M_r, M_12,M_c1,M_c2, Q_r1, Q_r2, Q_1,Q_2,Q_r,Q_12,Q_c1,Q_c2); 
    [score, total, mse, tmp,total_0, total_12, score_0, score_12] = Check(answer,predictions);
    
    fprintf('\nData_Label: %s\n',string(answer(1,1)));
    fprintf('Re-referencing: %d\n',ref_method(i));
    
    
    fprintf('MSE: %f\n',mse);
       
    
    fprintf('\nTotal: %d / %d\n',score,total);
    fprintf("SCORE: %f\n",100*score/total); 
    
    fprintf('\nTotal_0: %d / %d\n',score_0,total_0);
    fprintf("SCORE: %f\n",100*score_0/total_0); 
    
    fprintf('\nTotal_12: %d / %d\n',score_12,total_12);
    fprintf("SCORE: %f\n",100*score_12/total_12); 
end

% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
