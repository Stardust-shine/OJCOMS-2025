function [W_BB] = WBB_precoding(H,num_users, user_num_an, num_user_stream,snr_dB,W_RF,F_RF)
% F_BB=zeros(num_users*num_user_stream,num_users*num_user_stream);
snr=10^(-snr_dB/10);
H_EQ = zeros(num_users*num_user_stream,num_users*num_user_stream);
W_BB=zeros(num_user_stream, num_users*num_user_stream);
for user=1:num_users
    h = H((user-1)*user_num_an+1:user*user_num_an,:);
    w_rf=W_RF((user-1)*num_user_stream+1:user*num_user_stream,:);
    h_eq=conj(w_rf)*h*F_RF;
    H_EQ((user-1)*num_user_stream+1:user*num_user_stream,:)=h_eq;
end

for user=1:num_users
    H_tilde=H_EQ;
    H_tilde((user-1)*num_user_stream+1:user*num_user_stream,:)=[];
    B_tilde=(H_tilde'*H_tilde + (num_user_stream/snr)*eye(num_users*num_user_stream))^(-1);
    
    h_eq_k = H_EQ((user-1)*num_user_stream+1:user*num_user_stream,:);
    [U_tilde,~,~] = svd(h_eq_k*B_tilde); 
    w_bb=U_tilde(:,1:num_user_stream);
    % w_rf=W_RF((user-1)*num_user_stream+1:user*num_user_stream,:);
    % w_bb=sqrt(num_user_stream)*w_bb/norm(w_bb*w_rf,'fro');
    % W_BB=[w_bb_1,w_bb_2,...,w_bb_K];
    W_BB(:,(user-1)*num_user_stream+1:user*num_user_stream)=w_bb;
end
% F_BB = norm_frf_fbb*F_BB/norm(F_RF*F_BB,'fro');
end

