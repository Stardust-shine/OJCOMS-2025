function [W,W_RF_all,F_new] = RF_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,num_user_rf, bs_num_rf, num_iter)
%  UNTITLED12 此处显示有关此函数的摘要
%  此处显示详细说明
W_RF_all=zeros(num_users*user_num_an,num_user_rf);
for k=1:num_users
    h_k=H((k-1)*user_num_an+1:k*user_num_an,:);
    A_k=h_k*h_k';
    
    w_k_initial=randn(user_num_an,num_user_rf) + sqrt(-1) * randn(user_num_an,num_user_rf) ;
    w_k_initial = (1/sqrt(user_num_an)) * w_k_initial./abs(w_k_initial);
    
    w_k_new=update_wrf_k(A_k,user_num_an,num_user_rf,w_k_initial,num_iter);
    W_RF_all((k-1)*user_num_an+1:k*user_num_an,:)=w_k_new;
end
W_RF_diag=cell(num_users,1);

for k=1:num_users
    W_RF_diag{k,1}=W_RF_all((k-1)*user_num_an+1:k*user_num_an,:);
end

W=blkdiag(W_RF_diag{:});
A=H'*W*W'*H;
F=randn(bs_num_an, bs_num_rf) + sqrt(-1) * randn(bs_num_an, bs_num_rf) ;
F = (1/sqrt(bs_num_an)) * F./abs(F);

F_new =update_f_rf(A,F,bs_num_an,bs_num_rf,num_iter);
end

function [sum] =cal_sum(a, b, ele, d)
sum=0;
for L=1:d
    if L==ele
        sum=sum+0;
    else
        sum = sum+ a*b(L,1);
    end
end
end

function [w_k_new] =update_wrf_k(A_k,user_num_an,num_user_rf,w_k,num_iter)
w_k_new=zeros(size(w_k));
for ite=1:num_iter
    for j=1:num_user_rf
        w_k_hat=w_k;
        w_k_hat(:,j)=[];
        C_j=w_k_hat'*A_k*w_k_hat;
        G_j=A_k-A_k*w_k_hat*C_j^(-1)*w_k_hat'*A_k;
        for i=1:user_num_an
            sum=cal_sum(G_j(i, j), w_k(:,j), i, user_num_an);
            sum=sum/abs(sum);
            w_k_new(i, j)=(1/sqrt(user_num_an)) * sum;
        end
    end
end
end

function [F] =update_f_rf(A,F,bs_num_an,bs_num_rf,num_iter)
for ite=1:num_iter
    for j=1:bs_num_rf
        F_j_hat=F;
        F_j_hat(:,j)=[];
        D_j=F_j_hat'*A*F_j_hat;
        E_j=A-A*F_j_hat*D_j^(-1)*F_j_hat'*A;
        for i=1:bs_num_an
            sum=cal_sum(E_j(i, j), F(:,j), i, bs_num_an);
            sum=sum/abs(sum);
            F(i,j)=(1/sqrt(bs_num_an)) * sum;
        end
    end
end
end