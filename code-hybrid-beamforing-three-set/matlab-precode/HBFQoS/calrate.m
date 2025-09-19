function R = calrate(H,W,K,SNR,eta)


sigma2 = 10^(-SNR/10);
Q = abs(H*W).^2;
D = eye(K).* Q;
sinr_set = sum(D,2)./(sigma2+sum(Q,2)-sum(D,2));
R = log2(1+sinr_set);
if any(R<eta-1e4)
    R = 0;
    return
end
R = sum(R);
end