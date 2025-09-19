function [rate_all] = data_rate(num_test,num_users,user_num_an, num_user_stream,H_all,W_RF_all,F_RF_all,F_BB_all,W_BB_all,sigma_dB)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
rate_all=[];
sigma=10^(-sigma_dB/10);
for itr=1:num_test
    rate=0;
    H=squeeze(H_all(itr,:,:));
    W_RF=squeeze(W_RF_all(itr,:,:));
    F_RF=squeeze(F_RF_all(itr,:,:));
    F_BB=squeeze(F_BB_all(itr,:,:));
    W_BB=squeeze(W_BB_all(itr,:,:));
    for user=1:num_users
        H_k=H((user-1)*user_num_an+1:user*user_num_an,:);
        f_rf=F_RF;
        f_bb=F_BB(:,(user-1)*num_user_stream+1:user*num_user_stream);
        w_rf=W_RF((user-1)*num_user_stream+1:user*num_user_stream,:);
        w_bb=W_BB(:,(user-1)*num_user_stream+1:user*num_user_stream);
        v_k=f_rf*f_bb;
        w_k=w_rf.'*w_bb;
        sum_inf_value=sum_inf(F_RF,F_BB,num_users,num_user_stream,user);
        Q_k=w_k'*H_k*sum_inf_value*H_k'*w_k+sigma*w_k'*w_k;
        rate_k=log2(abs(det(eye(num_user_stream)+Q_k^(-1)*w_k'*H_k*v_k*v_k'*H_k'*w_k)));
        rate=rate+rate_k;
    end
    rate_all=[rate_all rate];
end
end

function [sum] = sum_inf(F_RF,F_BB,num_users,num_user_stream,k)
sum=0;
for L=1:num_users
    if L==k
        sum=sum+0;
    else
        f_rf_inf=F_RF;
        f_bb_inf=F_BB(:,(L-1)*num_user_stream+1:L*num_user_stream);
        sum = sum + (f_rf_inf*f_bb_inf)*(f_rf_inf*f_bb_inf)';
    end
end

end