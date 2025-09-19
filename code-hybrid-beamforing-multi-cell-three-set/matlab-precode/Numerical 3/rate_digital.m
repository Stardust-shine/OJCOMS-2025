rate_all=zeros(1,num_test);
V=T_all;
U=R_all;
for sample=1:num_test
    h=reshape(H_com(sample,1:num_users*user_num_an,1:bs_num_an),num_users*user_num_an,bs_num_an);
    u=squeeze(U(sample,:, :));
    v=squeeze(V(sample,:, :));
    rate=0;
    for user=1:num_users
        signal=abs(conj(u(user,:))*h((user-1)*user_num_an+1:user*user_num_an,:)*v(:,user))^2;
        inf_u=0;
        for uu = 1:num_users
            inf_u = inf_u + abs(conj(u(user,:))*h((user-1)*user_num_an+1:user*user_num_an,:)*v(:,uu))^2;
        end
        inf_u = (inf_u - signal) + sigma;
        rate = rate + log2(abs(1+signal/inf_u));
    end
    rate_all(1,sample)=rate;
end
mean(rate_all)