% Machine Learning-Enabled Joint Antenna Selection and Precoding Design: From Offline Complexity to Online Performance
% 用SCA解决DC
% 输入 信道H（用户数K，天线数N）（原文是行向量），信噪比SNR（实数），数据率要求eta（K,1）
% 输出 预编码W(N,K)
function W = SCAalgoRF(H,SNR,eta,FRF)
    [K,N] = size(H);
    sigma2 = 10^(-SNR/10);
    %%
    %初始化W_hat和u_hat
    % 求bigW
    bigH = zeros(N,N,K);
    for k = 1:K
        bigH(:,:,k) = H(k,:)'*H(k,:);
    end
    den = 1./(2.^eta-1);
    cvx_begin quiet sdp
        cvx_solver sedumi
        variable bigW(N,N,K) hermitian;
        expressions sum_tr_bigHbigW(K,1) sum_tr_bigW;
        subject to
            for k = 1:K
                for i = 1:K
                    if i ~= k
                        sum_tr_bigHbigW(k) = sum_tr_bigHbigW(k) + real(trace(bigH(:,:,k)*bigW(:,:,i)));
                    end
                end
            end
            for k = 1:K % K constraints
                real(trace(bigH(:,:,k)*bigW(:,:,k)))*den(k) >= sum_tr_bigHbigW(k) + sigma2;
            end
            for k = 1:K
                sum_tr_bigW  = sum_tr_bigW + real(trace(FRF*bigW(:,:,k)*FRF'));
                bigW(:,:,k) >= 0;
            end
            sum_tr_bigW <= 1; % one constraint
    cvx_end
    
    if strcmp(cvx_status,'Solved')==0 %不可解
        W = FRF*zeros(N,K);
        return
    end
    
    %求W_hat和u_hat
    W_hat = zeros(N,K);
    for k = 1:K
        [V,D] = eig(bigW(:,:,k));
        W_hat(:,k) = sqrt(D(N,N)) * V(:,N);
    end
    Q = abs(H*W_hat).^2;
    D = eye(K).* Q;
    u_hat = sum(D,2)./(sigma2+sum(Q,2)-sum(D,2));
    
    %%
    %Algo2:反复解问题(14)
    Xold = 0;
    err = 10000;
    epsilon = 1e-2;
    eta_bar = 2.^eta-1;
    while err > epsilon
        u_hat_reci = 1./u_hat;
        cvx_begin quiet
            cvx_solver sedumi
            variable W(N,K) complex;
            variable u(K,1) nonnegative;
            expression sum_wkHkwi_ex(K,1);
            maximize(sum(log(1+u)));
            subject to
                u>=eta_bar;%(11b)
                norm(FRF*W,'fro')<=1;%(11c)
                for k = 1:K
                    for i = 1:K
                        if i ~= k
                            sum_wkHkwi_ex(k) = sum_wkHkwi_ex(k) + W(:,i)'*H(k,:)'*H(k,:)*W(:,i);
                        end
                    end
                end
                for k = 1:K %(13)原文有问题
%                     sum_wkHkwi_ex(k) + sigma2 <= u_hat_reci(k)*real(W(:,k)'*H(k,:)'*H(k,:)*W_hat(:,k)) ...
%                         + u_hat_reci(k)*real(W(:,k)'*Hc(k,:)'*Hc(k,:)*W_hat(:,k)) ...
%                         - u(k)*u_hat_reci(k)*u_hat_reci(k)*real(W_hat(:,k)'*H(k,:)'*H(k,:)*W_hat(:,k)) ...
%                         + u_hat_reci(k)*real(W_hat(:,k)'*H(k,:)'*H(k,:)*W_hat(:,k)) ...
%                         - u_hat_reci(k)*real(W_hat(:,k)'*Hc(k,:)'*Hc(k,:)*W_hat(:,k));   
                    sum_wkHkwi_ex(k) + sigma2 <= u_hat_reci(k)*(2-u(k)*u_hat_reci(k))*real(W_hat(:,k)'*H(k,:)'*H(k,:)*W_hat(:,k)) ...
                        +2*u_hat_reci(k)*real(W_hat(:,k)'*H(k,:)'*H(k,:)*(W(:,k))) ...
                        -2*u_hat_reci(k)*real(W_hat(:,k)'*H(k,:)'*H(k,:)*(W_hat(:,k)));
                end
        cvx_end
        
        err = abs(sum(log2(1+u))-Xold);
        u_hat = u;
        W_hat = W;
        Xold = sum(log2(1+u));
    end


    W = FRF*W;
end
