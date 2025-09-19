function [T_all,R_all] = Precoding(H,num_iteration,num_users,user_num_an,bs_num_an,num_user_stream,snr_dB,P_max)
% H: [K*N_R, N_T]
H_tilde_all=zeros(num_users*num_user_stream,bs_num_an);
snr=10^(-snr_dB/10);
R_all=zeros(num_users*num_user_stream,user_num_an);
alpha=num_users*user_num_an/snr;
T_all=zeros(bs_num_an,num_users*num_user_stream);
for ite=1:num_iteration-1
    for user=1:num_users
        h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
        if ite==1
            h_tilde_k=H;
            h_tilde_k((user-1)*user_num_an+1:user*user_num_an,:) = [];
        else
            h_tilde_k=H_tilde_all;
            h_tilde_k((user-1)*num_user_stream+1:user*num_user_stream,:) = [];
        end
        T_k = (h_tilde_k'*h_tilde_k + alpha*eye(bs_num_an))^(-1);
        [U_k,~,~]=svd(h_k*T_k);
        R_k=U_k(:,1:num_user_stream);
        R_k=R_k';
        % H_tilde_all((user-1)*num_user_stream+1:user*num_user_stream,:)=R_k*h_k;
        R_all((user-1)*num_user_stream+1:user*num_user_stream,:)=R_k;
    end
    for user=1:num_users
        h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
        R_k=R_all((user-1)*num_user_stream+1:user*num_user_stream,:);
        H_tilde_all((user-1)*num_user_stream+1:user*num_user_stream,:)=R_k*h_k;
    end
end


SUM=0;
for user=1:num_users
    R_k=R_all((user-1)*num_user_stream+1:user*num_user_stream,:);
    h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
    SUM = SUM + h_k'*R_k'*R_k*h_k;
end


T_k_all=zeros(num_users*bs_num_an,num_user_stream);
for user=1:num_users
    R_k=R_all((user-1)*num_user_stream+1:user*num_user_stream,:);
    h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
    t_k = (SUM + (alpha*num_user_stream/user_num_an) * eye(bs_num_an))^(-1)*h_k'*R_k';
    T_k_all((user-1)*bs_num_an+1:user*bs_num_an,:)=t_k;
end


SUM_norm=0;
for user=1:num_users
    t_k = T_k_all((user-1)*bs_num_an+1:user*bs_num_an,:);
    SUM_norm = SUM_norm + norm(t_k, 'fro')^2;
end

for user=1:num_users
    t_k = sqrt(P_max/SUM_norm) * T_k_all((user-1)*bs_num_an+1:user*bs_num_an,:);
    T_all(:,(user-1)*num_user_stream+1:user*num_user_stream)=t_k;
end

% for user=1:num_users
%     h_k=H((user-1)*user_num_an+1:user*user_num_an,:);
%     t_k=T_all(:,(user-1)*num_user_stream+1:user*num_user_stream);
%     H_eff_k=[];
%     H_eff_k=[H_eff_k h_k*t_k];
%     for j=1:num_users
%         if j ~=user
%             t_j=T_all(:,(j-1)*num_user_stream+1:j*num_user_stream);
%             H_eff_k = [H_eff_k h_k*t_j];
%         end
%     end
%     r_k=(H_eff_k'*H_eff_k +alpha*eye(num_users*num_user_stream))^(-1)*H_eff_k';
%     R_all((user-1)*num_user_stream+1:user*num_user_stream,:)=r_k((user-1)*num_user_stream+1:user*num_user_stream,:);
% end


end

