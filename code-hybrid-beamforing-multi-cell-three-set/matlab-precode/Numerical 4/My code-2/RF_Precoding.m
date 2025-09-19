function [W_RF_all,F_RF] = RF_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf)
[W_RF_all] = W_RF_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf);
H_int=[];
for user=1:num_users
    w_rf_k=W_RF_all((user-1)*user_num_an+1:user*user_num_an,:);
    h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
    H_int=[H_int;w_rf_k'*h_k];
end
H_int = H_int';
F_RF =H_int./abs(H_int);
end
