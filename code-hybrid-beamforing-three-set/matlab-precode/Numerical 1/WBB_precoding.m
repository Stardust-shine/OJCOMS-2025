function [W_BB] = WBB_precoding(H,num_users, user_num_an, num_user_stream,sigma_dB,W_RF,F_RF,F_BB)
snr=10^(-sigma_dB/10);
H_tilde = zeros(num_users*num_user_stream,num_users*num_user_stream);
W_BB=zeros(num_users*num_user_stream,num_user_stream);
for user=1:num_users
    h = H((user-1)*user_num_an+1:user*user_num_an,:);
    w_rf=W_RF((user-1)*num_user_stream+1:user*num_user_stream,:);
    h_tilde=conj(w_rf)*h*F_RF;
    H_tilde((user-1)*num_user_stream+1:user*num_user_stream,:)=h_tilde;
end

for user=1:num_users
    w_rf=W_RF((user-1)*num_user_stream+1:user*num_user_stream,:);
    h_tilde_k = H_tilde((user-1)*num_user_stream+1:user*num_user_stream,:);
    A=(h_tilde_k*F_BB*F_BB'*h_tilde_k' + (num_users*num_user_stream/snr)*conj(w_rf)*w_rf.')^(-1);
    w_bb=A*h_tilde_k*F_BB(:,(user-1)*num_user_stream+1:user*num_user_stream);
    W_BB((user-1)*num_user_stream+1:user*num_user_stream,:)=w_bb;
end
end

