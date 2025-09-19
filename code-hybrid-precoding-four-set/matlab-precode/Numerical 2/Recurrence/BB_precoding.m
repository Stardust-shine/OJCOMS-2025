function [F_BB] = BB_precoding(H,num_users, user_num_an, bs_num_rf,num_user_stream,snr_dB,W_RF,F_RF,norm_frf_fbb)
%UNTITLED4 此处显示有关此函数的摘要

%   此处显示详细说明
% F_BB=zeros(num_users*num_user_stream,num_users*num_user_stream);
snr=10^(-snr_dB/10);
H_EQ = zeros(num_users*num_user_stream,num_users*num_user_stream);
F_BB=zeros(bs_num_rf,num_users*num_user_stream);
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
    [~,~,V_tilde] = svd(h_eq_k*B_tilde); 
    
    B_k_ = B_tilde*V_tilde(:,1:num_user_stream);
    B_k_=B_k_/norm(B_k_,'fro');
    
    % f_bb=norm_frf_fbb*B_k_/norm(F_RF * B_k_,'fro');
    % F_BB=[f_bb_1,f_bb_2,...,f_bb_K];
    F_BB(:,(user-1)*num_user_stream+1:user*num_user_stream)=B_k_;
end
for user=1:num_users
    B_k_=F_BB(:,(user-1)*num_user_stream+1:user*num_user_stream);
    f_bb=norm_frf_fbb*B_k_/norm(F_RF * B_k_,'fro');
    F_BB(:,(user-1)*num_user_stream+1:user*num_user_stream)=f_bb;
end
% F_BB = norm_frf_fbb*F_BB/norm(F_RF*F_BB,'fro');
end

