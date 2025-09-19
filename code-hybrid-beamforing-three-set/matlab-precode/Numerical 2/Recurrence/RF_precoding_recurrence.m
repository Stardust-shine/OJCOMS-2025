function [W_RF,F_RF] = RF_precoding_recurrence(H,num_users, user_num_an, bs_num_an, num_user_stream,norm_wrf,norm_frf,tt)
% H [num_users*user_num_an,bs_num_an]
% H_G = zeros(bs_num_an,num_users*num_user_stream);
W_RF=tt;
H_G=[];
for user=1:num_users
    h = H((user-1)*user_num_an+1:user*user_num_an,:);
    w_rf=W_RF((user-1)*num_user_stream+1:user*num_user_stream,:);
    H_G=[H_G (conj(w_rf)*h).'];
end
H_G=H_G';
F_RF=norm_frf*H_G./abs(H_G);
end