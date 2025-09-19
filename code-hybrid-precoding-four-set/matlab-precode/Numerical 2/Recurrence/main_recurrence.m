num_users=18;
user_num_an=16;
bs_num_an=128;
num_user_stream=2;
P_max=1;
sigma_dB=-10;
sigma=P_max * 10^(-sigma_dB/10);
num_bs=1;
bs_num_rf = num_users*num_user_stream;
num_test=1000;

norm_wrf=1/sqrt(user_num_an);
norm_frf=1/sqrt(bs_num_an);
norm_frf_fbb=sqrt(P_max);

F_BB_all=zeros(num_test,bs_num_rf,num_users*num_user_stream);
F_RF_all=zeros(num_test,bs_num_an, bs_num_rf);
W_RF_all=zeros(num_test,num_users*num_user_stream,user_num_an);
W_BB_all=zeros(num_test, num_user_stream, num_users*num_user_stream);
for sample=1:num_test
    H=squeeze(H_com(sample,:,:));
    tt=squeeze(TT(num_test,:,:));
    [w_rf,f_rf] = RF_precoding_recurrence(H,num_users, user_num_an, bs_num_an, num_user_stream,norm_wrf,norm_frf,tt);
    f_rf=f_rf.';
    [f_bb] = BB_precoding(H,num_users, user_num_an, bs_num_rf,num_user_stream,sigma_dB,w_rf,f_rf,norm_frf_fbb);
    [w_bb] = WBB_precoding(H,num_users, user_num_an, num_user_stream,sigma_dB,w_rf,f_rf);
    W_RF_all(sample,:,:)=w_rf;
    F_RF_all(sample,:,:)=f_rf;
    F_BB_all(sample,:,:)=f_bb;
    W_BB_all(sample,:,:)=w_bb;
end

rate_all=data_rate(num_test,num_users,user_num_an, num_user_stream,H_com,W_RF_all,F_RF_all,F_BB_all,W_BB_all,sigma_dB);
mean(rate_all)