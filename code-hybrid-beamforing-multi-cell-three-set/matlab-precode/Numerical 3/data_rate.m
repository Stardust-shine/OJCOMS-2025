function [rate_all] = data_rate(num_test,num_users,user_num_an, num_user_stream,H_all,T_all,R_all,sigma_dB)
rate_all=[];
sigma=10^(-sigma_dB/10);
for itr=1:num_test
    rate=0;
    H=squeeze(H_all(itr,:,:));
    T=squeeze(T_all(itr,:,:));
    R=squeeze(R_all(itr,:,:));
    for user=1:num_users
        H_k=H((user-1)*user_num_an+1:user*user_num_an,:);
        v_k=T(:,(user-1)*num_user_stream+1:user*num_user_stream);
        w_k=R((user-1)*num_user_stream+1:user*num_user_stream,:);
        w_k=w_k.';
        sum_inf_value=sum_inf(T,num_users,num_user_stream,user);
        Q_k=w_k'*H_k*sum_inf_value*H_k'*w_k+sigma*w_k'*w_k;
        rate_k=log2(abs(det(eye(num_user_stream)+Q_k^(-1)*w_k'*H_k*v_k*v_k'*H_k'*w_k)));
        rate=rate+rate_k;
    end
    rate_all=[rate_all rate];
end
end

function [sum] = sum_inf(T,num_users,num_user_stream,k)
sum=0;
for L=1:num_users
    if L==k
        sum=sum+0;
    else
        t_k=T(:,(L-1)*num_user_stream+1:L*num_user_stream);
        sum = sum + t_k*t_k';
    end
end

end