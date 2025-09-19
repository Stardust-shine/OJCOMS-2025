function [F_BB] = FBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,W_RF_all,F_RF, P_max,sigma_dB,norm_frf_fbb)
H_hat_all=zeros(num_users*num_user_stream,bs_num_rf);
for user=1:num_users
    h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
    w_rf_k=W_RF_all((user-1)*user_num_an+1:user*user_num_an,:);
    H_hat_all((user-1)*num_user_stream+1:user*num_user_stream,:)=w_rf_k'*h_k*F_RF;
end

snr=10^(sigma_dB/10);
sigma_all=[];
F_BB=[];
for user=1:num_users
    h_hat_k=H_hat_all((user-1)*num_user_stream+1:user*num_user_stream,:);
    H_tilde=H_hat_all;
    H_tilde((user-1)*num_user_stream+1:user*num_user_stream,:)=[];
    
    [~,S,V_tilde]=svd(H_tilde);
    s = diag(S);
    L_k_tilde=nnz(s);
    if L_k_tilde+1>bs_num_rf
        V_tilde_k_0=V_tilde(:,bs_num_rf);
    else
        V_tilde_k_0=V_tilde(:,L_k_tilde+1:bs_num_rf);
    end
    
    [~,S,V]=svd(h_hat_k*V_tilde_k_0);
    s = diag(S);
    s=s';
    sigma_all=[sigma_all s(1:num_user_stream)];
    V_k_1 = V(:,1:num_user_stream);
    
    F_BB=[F_BB V_tilde_k_0*V_k_1];
end

% sigma_all = sigma_all.^2;
sigma_all = (snr/(num_users*num_user_stream))*sigma_all.^2;
[Lamda] = water_filling(sigma_all, norm_frf_fbb,num_users,num_user_stream);

F_BB=F_BB*sqrt(diag(Lamda));

end

function [Lamda] = water_filling(sigma_all, norm_frf_fbb,num_users,num_user_stream)
norm = norm_frf_fbb^2;
Lamda=zeros(1, num_users*num_user_stream);

water_level_min=0.0000001;
water_level_max=max(sigma_all);
thre=0.00001;

water_level=(water_level_max + water_level_min)/2;
error=100;
num_itr=0;

while(num_itr<500)
    sum=0;
    for k=1:num_users*num_user_stream
        sum=sum + max(1/water_level-1/sigma_all(k),0);
    end
    error = abs(sum-norm);
    if sum >norm
        water_level_min=water_level;
        water_level = (water_level_min+water_level_max)/2;
    else
        water_level_max=water_level;
        water_level = (water_level_min+water_level_max)/2;
    end
    num_itr = num_itr +1;
end

for k=1:num_users*num_user_stream
    Lamda(k)=max(1/water_level-1/sigma_all(k),0);
end

end

