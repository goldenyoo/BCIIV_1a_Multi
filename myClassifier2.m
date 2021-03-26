function [check, result] = myClassifier2(fp,p_1,p_2,V,k)
    
insert = V'*fp;

map1 = double(subs(p_1,insert));
map2 = double(subs(p_2,insert));

if map1 > map2
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
check = map1 - map2;
end
