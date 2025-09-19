function [sum_rate] = cal_rate(num_users,num_user_stream,Lamda, Sigma_all)
sum_rate= 0;
for user=1:num_users
    lamda_k = Lamda(1, (user-1)*num_user_stream+1:user*num_user_stream);
    sigma_k = Sigma_all(1, (user-1)*num_user_stream+1:user*num_user_stream);
    rate_k=log2(det(eye(num_user_stream) + diag(lamda_k) * diag(sigma_k) ));
    sum_rate = sum_rate + rate_k;
end
end

