function [check, result] = myClassifier(fp,Mr,Ml,Qr,Ql)
    
      
    check = (0.5*(fp - Mr)'*pinv(Qr)*(fp - Mr) + 0.5*log(det(Qr))) - (0.5*(fp - Ml)'*pinv(Ql)*(fp - Ml) + 0.5*log(det(Ql))); 
 
    if check > 0
        result = 3;  % l
    else
        result = 4;   % r
    end
end
