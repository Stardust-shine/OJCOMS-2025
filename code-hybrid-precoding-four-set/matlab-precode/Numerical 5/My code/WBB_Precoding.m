function [W_BB_diag, W_BB] = WBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,num_user_rf, W_RF_all,F_RF, P_max)
H_hat_all=zeros(num_users*num_user_rf,bs_num_rf);
W_BB_diag=cell(num_users,1);
W_BB=[];
for user=1:num_users
    h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
    w_rf_k=W_RF_all((user-1)*user_num_an+1:user*user_num_an,:);
    H_hat_all((user-1)*num_user_rf+1:user*num_user_rf,:)=w_rf_k'*h_k*F_RF;
end

for user=1:num_users
    h_hat_k=H_hat_all((user-1)*num_user_rf+1:user*num_user_rf,:);
    H_tilde=H_hat_all;
    H_tilde((user-1)*num_user_rf+1:user*num_user_rf,:)=[];
    
    [~,S,V_tilde]=svd(H_tilde);
    s = diag(S);
    L_k_tilde=nnz(s);
    V_tilde_k_0=V_tilde(:,L_k_tilde+1:bs_num_rf);
    
    [U,~,~]=svd(h_hat_k*V_tilde_k_0);
    W_BB_diag{user, 1}=U(:,1:num_user_stream);
    W_BB=[W_BB;U(:,1:num_user_stream)];
end

W_BB_diag=blkdiag(W_BB_diag{:});
end
