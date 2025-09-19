num_iteration=50;
num_users=4;
user_num_an=2;
bs_num_an=4;
num_user_stream=1;
P_max=1;
sigma_dB=15;
sigma=P_max * 10^(-sigma_dB/10);
num_bs=1;
bs_num_rf = num_users*num_user_stream;
num_test=100;
H_com=zeros(num_test,num_users*user_num_an,bs_num_an);
for i=1:num_test
    H_com(i,:,:)=(randn(num_users*user_num_an,bs_num_an) + sqrt(-1)*randn(num_users*user_num_an,bs_num_an))/sqrt(2);
end
T_all = zeros(num_test, bs_num_an,num_users*num_user_stream);
R_all = zeros(num_test, num_users*num_user_stream,user_num_an);
for i=1:num_test
    H = squeeze(H_com(i,:,:));
    [T,R] = Precoding(H,num_iteration,num_users,user_num_an,bs_num_an,num_user_stream,sigma_dB,P_max);
    T_all(i,:,:)=T;
    R_all(i,:,:)=R;
end

[rate_all] = data_rate(num_test,num_users,user_num_an, num_user_stream,H_com,T_all,R_all,sigma_dB);
mean(rate_all)




