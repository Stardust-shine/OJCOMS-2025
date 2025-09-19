function [Rate] = cal_rate(num_users,num_user_rf, num_user_stream,snr_dB,H,w_rf,f_rf,f_bb,w_bb)
%UNTITLED8 此处显示有关此函数的摘要
%   此处显示详细说明
snr=10^(snr_dB/10);
I=eye(num_users*num_user_stream);
alpha=snr/(num_users*num_user_rf);
Rate=log2(abs(det(I + alpha * w_bb'*w_rf'*H*f_rf*f_bb*f_bb'*f_rf'*H'*w_rf*w_bb)));
end

