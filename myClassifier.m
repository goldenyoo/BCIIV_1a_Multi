function [check, result] = myClassifier(fp,Mr,Ml,Qr,Ql,k)
    

    check = (0.5*(fp - Mr)'*pinv(Qr)*(fp - Mr) + 0.5*log(det(Qr))) - (0.5*(fp - Ml)'*pinv(Ql)*(fp - Ml) + 0.5*log(det(Ql))); 
 
    if check > 0
        if k ==1
            result = -10;  % l
        elseif k==2
            result = -10;
        else
            result = -10;
        end
    else
        if k ==1
            result = 0;  % l
        elseif k==2
            result =-1;
        else
            result = 1;
        end
    end
end
