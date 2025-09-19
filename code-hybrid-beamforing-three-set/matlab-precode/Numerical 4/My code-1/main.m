num_users=40;
user_num_an=1;
bs_num_an=128;
num_user_stream=1;

P_max=1;
sigma_dB=10;
sigma=P_max * 10^(-sigma_dB/10);

num_bs=1;
bs_num_rf =64;
num_test=500;
threshold=0.1;
norm_f_rf=1/sqrt(bs_num_an);
norm_w_rf=1/sqrt(user_num_an);
norm_frf_fbb=sqrt(num_users);

F_BB_all=zeros(num_test,bs_num_rf,num_users*num_user_stream);
F_RF_all=zeros(num_test,bs_num_an, bs_num_rf);
W_RF_all=zeros(num_test,num_user_stream,num_users*user_num_an);
W_BB_all=zeros(num_test, num_user_stream, num_users*num_user_stream);

H_com=squeeze(H_all(:,1,:,:))+sqrt(-1)*squeeze(H_all(:,2,:,:));
for sample=1:num_test
    H = squeeze(H_com(sample,:,:));
    [w_rf,f_rf,num_ite] = RF_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,threshold);
    f_rf=norm_f_rf*f_rf./abs(f_rf);
    w_rf =norm_w_rf*w_rf./abs(w_rf);
    [f_bb] = FBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,w_rf,f_rf,P_max,sigma_dB,norm_frf_fbb);
    [w_bb] = WBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,w_rf,f_rf, P_max);
    
    
    W_RF_all(sample,:,:)=w_rf.';
    F_RF_all(sample,:,:)=f_rf;
    F_BB_all(sample,:,:)=f_bb;
    w_bb=w_bb.';
    W_BB_all(sample,:,:)=w_bb;
end

rate_all=data_rate(num_test,num_users,user_num_an, num_user_stream,H_com,W_RF_all,F_RF_all,F_BB_all,W_BB_all,P_max,sigma);
mean(rate_all)




