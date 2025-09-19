rate_all=zeros(1,num_test);
for sample=1:num_test
    h=reshape(H_com(sample,1:num_users*user_num_an,1:bs_num_an),num_users*user_num_an,bs_num_an);
    F_RF=reshape(F_RF_all(sample,1:bs_num_an,1:bs_num_rf),bs_num_an,bs_num_rf);
    F_BB=reshape(squeeze(F_BB_all(sample,:,:)),bs_num_rf,num_users);
    w_rf=squeeze(W_RF_all(sample,:,:));
    w_rf = reshape(w_rf,user_num_an,num_users);
    w_rf=w_rf.';
    rate=0;
    for user=1:num_users
        signal=(1/num_users)*abs(conj(w_rf(user,:))*h((user-1)*user_num_an+1:user*user_num_an,:)*F_RF*F_BB(:,user))^2;
        inf_u=0;
        for uu = 1:num_users
            inf_u = inf_u + abs(conj(w_rf(user,:))*h((user-1)*user_num_an+1:user*user_num_an,:)*F_RF*F_BB(:,uu))^2;
        end
        inf_u = (1/num_users)*inf_u;
        inf_u = (inf_u - signal) + sigma;
        rate = rate + log2(abs(1+signal/inf_u));
    end
    rate_all(1,sample)=rate;
end
mean(rate_all)