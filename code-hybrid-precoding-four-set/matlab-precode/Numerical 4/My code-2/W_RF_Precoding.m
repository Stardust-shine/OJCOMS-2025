function [W_RF_all] = W_RF_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf)
D=zeros(user_num_an,user_num_an);
W_RF_all=zeros(num_users*user_num_an,num_user_stream);

for n_ms=1:user_num_an
    omega=(n_ms-1)*2*pi/user_num_an;
    d=(1/sqrt(user_num_an))*exp(sqrt(-1)*(0:1:user_num_an-1)*omega);
    D(n_ms,:)=d;
end

for user=1:num_users
    h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
    norm_F=zeros(1,user_num_an);
    for k=1:user_num_an
        norm_F(1,k)=norm(conj(D(k,:))*h_k, 1)^2;
    end
    [~, index]=sort(norm_F,2,'descend');
    index_selected=index(1:num_user_stream);
    w_rf_k=D(index_selected,:);
    w_rf_k=w_rf_k.';
    W_RF_all((user-1)*user_num_an+1:user*user_num_an,:)=w_rf_k;
end

end

