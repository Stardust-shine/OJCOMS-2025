function [W_RF,F_RF] = RF_precoding(H,num_users, user_num_an, bs_num_an, num_user_stream,norm_wrf,norm_frf,tt)
W_RF=zeros(num_users*num_user_stream,user_num_an);
H_G=[];
for user=1:num_users
    h = H((user-1)*user_num_an+1:user*user_num_an,:);
    [U,~,~] = svd(h); 
    U=U(:,1:num_user_stream);
    w_rf=norm_wrf*U./abs(U);
    W_RF((user-1)*num_user_stream+1:user*num_user_stream,:)=w_rf.';
    H_G=[H_G (w_rf'*h).'];
end
H_G=H_G';
F_RF=norm_frf*H_G./abs(H_G);
end

