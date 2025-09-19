num_users=5;
user_num_an=1;
bs_num_an=16;
num_user_stream=1;
num_cell_max=10;
num_cell=10;
cell_indx=1;

P_max=1;
sigma_dB=-1 * (-174 + 10 * log10(20 * 10^6) - 30);
sigma=P_max * 10^(-sigma_dB/10);

cd('..')
cd('..')
cd('..')
cd('./data')
H =h5read('Cell' + string(num_cell_max) + '_TX' + string(bs_num_an) + '_UE' + string(num_users) + '.mat', '/H_cell_test');
H=permute(H, [5, 4, 2, 3, 1]);
H_com=H(:,:,:,:,1) +sqrt(-1) * H(:,:,:,:,2);
H_com = H_com(:, cell_indx:cell_indx+num_cell-1, :, :);

cd('..')
cd('./matlab-precode/Numerical 4/My code-1-multi-cell')
num_bs=1;
bs_num_rf =8;
num_test=1000;
threshold=0.01;

norm_f_rf=1/sqrt(bs_num_an);
norm_w_rf=1/sqrt(user_num_an);
norm_frf_fbb=sqrt(num_users);

F_BB_all=zeros(num_test,num_cell, bs_num_rf,num_users*num_user_stream, 2);
F_RF_all=zeros(num_test,num_cell, bs_num_an, bs_num_rf, 2);
F_BB_all_1=zeros(num_test,bs_num_rf,num_users*num_user_stream);
F_RF_all_1=zeros(num_test,bs_num_an, bs_num_rf);
W_RF_all=zeros(num_test,num_user_stream,num_users*user_num_an);
W_BB_all=zeros(num_test, num_user_stream, num_users*num_user_stream);

% H_com=squeeze(H_all(:,:,:,:,1))+sqrt(-1)*squeeze(H_all(:,:,:,:,2));
for sample=1:num_test
    for c=1:num_cell
        H = squeeze(H_com(sample,c, :,:));
        [w_rf,f_rf,num_ite] = RF_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,threshold);
        f_rf=norm_f_rf*f_rf./abs(f_rf);
        w_rf =norm_w_rf*w_rf./abs(w_rf);
        [f_bb] = FBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,w_rf,f_rf,P_max,sigma_dB,norm_frf_fbb);
        [w_bb] = WBB_Precoding(H,num_users,user_num_an,bs_num_an,num_user_stream,bs_num_rf,w_rf,f_rf, P_max);


        W_RF_all(sample,:,:)=w_rf.';
        F_RF_all(sample,c,:,:, 1)=real(f_rf);
        F_RF_all(sample,c,:,:, 2)=imag(f_rf);
        F_BB_all(sample,c,:,:, 1)=real(f_bb);
        F_BB_all(sample,c,:,:, 2)=imag(f_bb);
        F_RF_all_1(sample,:,:)=f_rf;
        F_BB_all_1(sample,:,:)=f_bb;
        w_bb=w_bb.';
        W_BB_all(sample,:,:)=w_bb;
    end
end
% rate_all=data_rate(num_test,num_users,user_num_an, num_user_stream,squeeze(H_com),W_RF_all,F_RF_all_1,F_BB_all_1,W_BB_all,P_max,sigma);
% mean(rate_all)

cd('..')
cd('..')
cd('..')
cd('./data')
save('solution_Cell' + string(num_cell) + '_TX' + string(bs_num_an) + '_UE' + string(num_users) + '.mat')




