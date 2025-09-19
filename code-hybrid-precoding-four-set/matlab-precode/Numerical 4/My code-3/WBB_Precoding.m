function [W_BB] = WBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,W_RF_all,F_RF, P_max)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
H_hat_all=zeros(num_users*num_user_stream,bs_num_rf);
for user=1:num_users
    h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
    w_rf_k=W_RF_all((user-1)*user_num_an+1:user*user_num_an,:);
    H_hat_all((user-1)*num_user_stream+1:user*num_user_stream,:)=w_rf_k'*h_k*F_RF;
end

W_BB=[];
for user=1:num_users
    h_hat_k=H_hat_all((user-1)*num_user_stream+1:user*num_user_stream,:);
    H_tilde=H_hat_all;
    H_tilde((user-1)*num_user_stream+1:user*num_user_stream,:)=[];
    
    [~,S,V_tilde]=svd(H_tilde);
    s = diag(S);
    L_k_tilde=nnz(s);
    V_tilde_k_0=V_tilde(:,L_k_tilde+1:bs_num_rf);
    
    [U,~,~]=svd(h_hat_k*V_tilde_k_0);
    
    W_BB=[W_BB;U];
end

end
