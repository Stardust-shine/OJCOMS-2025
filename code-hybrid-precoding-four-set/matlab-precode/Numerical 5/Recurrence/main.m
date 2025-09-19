num_users=3;
user_num_an=4;
bs_num_an=16;
num_user_stream=1;
num_user_rf=1;

% P_max=1;
sigma_dB=10;
sigma=1;
P_max=10^(sigma_dB/10);


num_bs=1;
bs_num_rf = num_users*num_user_rf;
num_test=30;
num_iter=100;

% H_com=zeros(num_test,num_users*user_num_an,bs_num_an);
% for i=1:num_test
%     H_com(i,:,:)=(randn(num_users*user_num_an,bs_num_an) + sqrt(-1)*randn(num_users*user_num_an,bs_num_an))/sqrt(2);
% end

F_BB_all=zeros(num_test,bs_num_rf,num_users*num_user_stream);
F_RF_all=zeros(num_test,bs_num_an, bs_num_rf);
W_RF_all=zeros(num_test,num_users*user_num_an,num_user_rf);
W_BB_all=zeros(num_test, num_users*num_user_rf, num_user_stream);

rate = 0;
for sample=1:num_test
    H = squeeze(H_com(sample,:,:));
    [w_rf_block,w_rf, f_rf] = RF_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,num_user_rf, bs_num_rf,num_iter);
    [f_bb, lamda, sigma_all] = FBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,num_user_rf, w_rf,f_rf,P_max,sigma_dB);
    [w_bb_block, w_bb] = WBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream, bs_num_rf,num_user_rf, w_rf,f_rf, P_max);
    
    [sum_rate] = cal_rate(num_users, num_user_rf, num_user_stream,sigma_dB,H,w_rf_block,f_rf,f_bb,w_bb_block);
    rate= rate+ sum_rate;
    
    W_RF_all(sample,:,:)=w_rf;
    F_RF_all(sample,:,:)=f_rf;
    F_BB_all(sample,:,:)=f_bb;
    W_BB_all(sample,:,:)=w_bb;
end
rate/num_test




