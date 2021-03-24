function [check, result] = myClassifier(fp,M1,M2,Q1,Q2,k)
    
Sb = (M1-M2)*(M1-M2)';
Sw = Q1+Q2;
tmp = pinv(Sb)*Sw;

[V, D] = eig(tmp);
[d, ind] = sort(abs(diag(D)),'descend');
D_new = diag(d);
V_new = V(:,ind);

new_data = V_new'*fp;
data1 = V_new'*M1;
data2 = V_new'*M2;

tmp1 = data1 - new_data;
tmp2 = data2 - new_data;

if sum(tmp1.*tmp1) < sum(tmp2.*tmp2)
    if k==1
        result = 0;
    elseif k==2
        result = -1;
    else
        result = 1;
    end
else
    if k==1
        result = -10;
    elseif k==2
        result = -10;
    else
        result = -10;
    end
end

check = abs((min(abs(tmp1))) - (min(abs(tmp2))));

end
