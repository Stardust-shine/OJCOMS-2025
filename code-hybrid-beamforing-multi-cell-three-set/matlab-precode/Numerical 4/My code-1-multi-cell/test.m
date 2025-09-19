num_users=3;
user_num_an=2;
bs_num_an=8;
num_user_stream=1;

P_max=1;
sigma_dB=10;
sigma=P_max * 10^(-sigma_dB/10);

% sigma_dB=10;
% sigma=1;
% P_max=10^(sigma_dB/10);

num_bs=1;
bs_num_rf = 6;
num_test=1000;
threshold=0.1;

norm_f_rf=1;
norm_w_rf=1;
norm_frf_fbb=sqrt(P_max);

F_BB_all=zeros(num_test,bs_num_rf,num_users*num_user_stream);
F_RF_all=zeros(num_test,bs_num_an, bs_num_rf);
W_RF_all=zeros(num_test,num_user_stream,num_users*user_num_an);
W_BB_all=zeros(num_test, num_user_stream, num_users*num_user_stream);

for sample=1:num_test
    H = squeeze(H_com(sample,:,:));
    w_rf=squeeze(W_rf(sample,:,:));
    f_rf=squeeze(F_rf(sample,:,:));
    f_bb=reshape(squeeze(F_bb(sample,:,:)), bs_num_rf, num_users);
    
    
    W_RF_all(sample,:,:)=w_rf.';
    F_RF_all(sample,:,:)=f_rf;
    F_BB_all(sample,:,:)=f_bb;
end
rate_all=data_rate(num_test,num_users,user_num_an, num_user_stream,H_com,W_RF_all,F_RF_all,F_BB_all,W_BB_all,P_max,sigma);
mean(rate_all)
