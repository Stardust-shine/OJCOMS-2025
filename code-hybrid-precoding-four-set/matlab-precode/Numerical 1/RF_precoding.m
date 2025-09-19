function [W_RF,F_RF] = RF_precoding(H,num_users, user_num_an, bs_num_an, num_user_stream,norm_wrf,norm_frf)
% H [num_users*num_user_an,num_bs_an]
W_RF=zeros(num_users*num_user_stream,user_num_an);
F_RF=zeros(bs_num_an,num_users*num_user_stream);
for user=1:num_users
    h = H((user-1)*user_num_an+1:user*user_num_an,:);
    [U,~,V] = svd(h); 
    % U:[num_user_an,_] V:[num_bs_an,_]
    U=U(:,1:num_user_stream);
    V=V(:,1:num_user_stream);
    f_rf=norm_frf*V./abs(V);
    w_rf=norm_wrf*U./abs(U);
    W_RF((user-1)*num_user_stream+1:user*num_user_stream,:)=w_rf.';
    F_RF(:,(user-1)*num_user_stream+1:user*num_user_stream)=f_rf;
end

end

