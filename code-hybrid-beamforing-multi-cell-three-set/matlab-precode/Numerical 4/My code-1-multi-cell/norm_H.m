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

H_real=real(H_com);
H_imag=imag(H_com);

mean_real=sum(H_real, 'all')/size(H_real, 1)/size(H_real, 2)/size(H_real, 3);
var_real=sqrt(sum((H_real-mean_real).^2, 'all')/size(H_real, 1)/size(H_real, 2)/size(H_real, 3));
norm_real=(H_real-mean_real)/var_real;

mean_imag=sum(H_imag, 'all')/size(H_imag, 1)/size(H_imag, 2)/size(H_imag, 3);
var_imag=sqrt(sum((H_imag-mean_imag).^2, 'all')/size(H_imag, 1)/size(H_imag, 2)/size(H_imag, 3));
norm_imag=(H_imag-mean_imag)/var_imag;


