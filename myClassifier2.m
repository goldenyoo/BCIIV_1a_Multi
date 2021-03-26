function [check, result] = myClassifier2(fp,X1,X2,V,k)
    
insert = V'*fp;

data1 = V'*X1;
data2 = V'*X2;

tmp1 = insert - data1;
tmp2 = insert - data2;

ttmp1 = tmp1.*tmp1;
ttmp2 = tmp2.*tmp2;

h1 = ((4/(3*length(data1)))^0.2)*std(data1);
h2 = ((4/(3*length(data2)))^0.2)*std(data2);
phi_1 = (1/sqrt(2*pi))*exp(-ttmp1/(2*h1^2));
phi_2 = (1/sqrt(2*pi))*exp(-ttmp2/(2*h2^2));

map1 = sum(phi_1)/length(data1);
map2 = sum(phi_2)/length(data2);

if map1 > map2
    if k == 1
        result = 0;
    elseif k == 2
        result = 0;
    elseif k ==3
        result = -1;
    else
        result  = 0;
    end
else
    if k == 1
        result = -1;
    elseif k==2
        result = 1;
    elseif k==3
        result = 1;
    else
        result = 12;
    end
end
check = map1 - map2;
end
