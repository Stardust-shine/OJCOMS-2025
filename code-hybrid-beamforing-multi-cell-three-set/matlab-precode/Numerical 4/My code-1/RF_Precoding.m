function [W_RF_all,F_RF,num_ite] = RF_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,threshold)
W_RF_all=zeros(num_users*user_num_an,num_user_stream);
for user=1:num_users
    W_RF_all((user-1)*user_num_an+1:user*user_num_an,:)=[eye(num_user_stream);zeros(user_num_an-num_user_stream,num_user_stream)];
end
error=10;
norm_sum=0;
num_ite=0;
while(num_ite<500)
    P=0;
    for user=1:num_users
        h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
        w_rf_k=W_RF_all((user-1)*user_num_an+1:user*user_num_an,:);
        P=P+h_k'*w_rf_k*w_rf_k'*h_k;
    end
    [U,S,~]=svd(P);
    eigvec=sum(S);
    [~, index]=sort(eigvec,2,'descend');
    F_RF=U(:,index(1:bs_num_rf));
    for user=1:num_users
        h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
        Q_k = h_k*F_RF*F_RF'*h_k';
        [U,S,~]=svd(Q_k);
        eigvec=sum(S);
        [~, index]=sort(eigvec,2,'descend');
        w_rf_k=U(:,index(1:num_user_stream));
        W_RF_all((user-1)*user_num_an+1:user*user_num_an,:)=w_rf_k;
    end
    Norm=Norm_cal(H,W_RF_all,F_RF,num_users,user_num_an);
    error=Norm-norm_sum;
    norm_sum=Norm;
    num_ite=num_ite+1;
end

end


function [Norm_sum] = Norm_cal(H,W_RF_all,F_RF,num_users,user_num_an)
Norm_sum=0;
for user=1:num_users
    h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
    w_rf_k=W_RF_all((user-1)*user_num_an+1:user*user_num_an,:);
    h_k_hat=w_rf_k'*h_k*F_RF;
    Norm_sum = Norm_sum + h_k_hat;
end
Norm_sum = norm(Norm_sum,'fro');
end
