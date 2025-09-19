num_users=2;
user_num_an=2;
bs_num_an=8;
num_user_stream=1;
P_max=1;
sigma_dB=10;
sigma=P_max * 10^(-sigma_dB/10);


num_bs=1;
bs_num_rf = 6;
num_test=1000;

norm_f_rf=1;
norm_w_rf=1;
norm_fbb_frf=sqrt(1);

% H_com=zeros(num_test,num_users*user_num_an,bs_num_an);
% for i=1:num_test
%     H_com(i,:,:)=(randn(num_users*user_num_an,bs_num_an) + sqrt(-1)*randn(num_users*user_num_an,bs_num_an))/sqrt(2);
% end

F_BB_all=zeros(num_test,bs_num_rf,num_users*num_user_stream);
F_RF_all=zeros(num_test,bs_num_an, bs_num_rf);
W_RF_all=zeros(num_test,num_user_stream,num_users*user_num_an);
W_BB_all=zeros(num_test, num_user_stream, num_users*num_user_stream);

rate = 0;
for sample=1:num_test
    H = squeeze(H_com(sample,:,:));
    [w_rf,f_rf] = RF_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf);
    f_rf=norm_f_rf*f_rf./abs(f_rf);
    w_rf =norm_w_rf*w_rf./abs(w_rf);
    [f_bb, lamda, sigma_all] = FBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,w_rf,f_rf,P_max,sigma_dB,norm_fbb_frf);
    [w_bb] = WBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,w_rf,f_rf, P_max);
    
    [sum_rate] = cal_rate(num_users,num_user_stream,lamda,sigma_all);
    rate= rate+ sum_rate;
    
    W_RF_all(sample,:,:)=w_rf.';
    F_RF_all(sample,:,:)=f_rf;
    F_BB_all(sample,:,:)=f_bb;
    w_bb=w_bb.';
    W_BB_all(sample,:,:)=w_bb;
end
rate/num_test

rate_all=data_rate(num_test,num_users,user_num_an, num_user_stream,H_com,W_RF_all,F_RF_all,F_BB_all,W_BB_all,P_max,sigma);
mean(rate_all)




