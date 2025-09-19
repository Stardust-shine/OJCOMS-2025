num_users=5;
user_num_an=1;
bs_num_an=16;
num_user_stream=1;
num_cell=1;

P_max=1;
sigma_dB=-1 * (-174 + 10 * log10(20 * 10^6) - 30);
sigma=P_max * 10^(-sigma_dB/10);


num_bs=1;
bs_num_rf =8;
num_test=1000;

cd('..')
cd('..')
cd('..')
cd('./data')
H =h5read('Cell' + string(num_cell) + '_TX' + string(bs_num_an) + '_UE' + string(num_users) + '.mat', '/H_cell_test');
H=permute(H, [5, 4, 2, 3, 1]);
H_com=H(:,:,:,:,1) +sqrt(-1) * H(:,:,:,:,2);
H_com=squeeze(H_com);

cd('..')
cd(['./result', '/KCell'] + string(num_cell) + '_TX' + string(bs_num_an) + '_UE' + string(num_users))
result=load('result.mat');
F_bb=result.F_bb;
F_rf=result.F_rf;
F_bb=squeeze(F_bb);
F_rf=squeeze(F_rf);
F_bb=F_bb(:,:,1) + sqrt(-1)*F_bb(:,:,2);
F_rf=F_rf(:,:,:,1) + sqrt(-1)*F_rf(:,:,:,2);
F_bb=reshape(F_bb, [num_test, bs_num_rf, num_users]);
% F_bb=permute(F_bb, [1, 3, 2]);

fbb=squeeze(F_bb(1,:,:));
frf=squeeze(F_rf(1,:,:));

cd('..')
cd('..')
cd('./matlab-precode/Numerical 4/My code-1-multi-cell')

W_RF_all=ones(num_test,num_user_stream,num_users*user_num_an);
W_BB_all=ones(num_test, num_user_stream, num_users*num_user_stream);

rate_all=data_rate(num_test,num_users,user_num_an, num_user_stream,H_com,W_RF_all,F_rf,F_bb,W_BB_all,P_max,sigma);

mean(rate_all)
