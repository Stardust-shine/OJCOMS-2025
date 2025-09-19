function [F_BB] = BB_precoding(H,num_users, user_num_an, num_user_stream,snr_dB,W_RF,F_RF,norm_frf_fbb)
F_BB=zeros(num_users*num_user_stream,num_users*num_user_stream);
snr=10^(-snr_dB/10);
H_tilde = zeros(num_users*num_user_stream,num_users*num_user_stream);
for user=1:num_users
    h = H((user-1)*user_num_an+1:user*user_num_an,:);
    w_rf=W_RF((user-1)*num_user_stream+1:user*num_user_stream,:);
    h_tilde=conj(w_rf)*h*F_RF;
    H_tilde((user-1)*num_user_stream+1:user*num_user_stream,:)=h_tilde;
end

for user=1:num_users
    H_bar=H_tilde;
    H_bar((user-1)*num_user_stream+1:user*num_user_stream,:)=[];
    h_tilde_k = H_tilde((user-1)*num_user_stream+1:user*num_user_stream,:);
    A=h_tilde_k'*h_tilde_k;
    B=(num_users*num_user_stream^2/snr)*eye(num_users*num_user_stream)+H_bar'*H_bar;
    [V,D] = eig(A,B);
    lamda=sum(D); % [1,num_users*num_user_stream]
    [~, index]=sort(lamda,2,'descend');
    P=V(:,index(1:num_user_stream)); % [num_users*num_user_stream, num_user_stream]
    f_bb=norm_frf_fbb*P/norm(F_RF * P,'fro');
    %f_bb=P;
    % F_BB=[f_bb_1,f_bb_2,...,f_bb_K];
    F_BB(:,(user-1)*num_user_stream+1:user*num_user_stream)=f_bb;
end
% F_BB = norm_frf_fbb*F_BB/norm(F_RF*F_BB,'fro');
end

