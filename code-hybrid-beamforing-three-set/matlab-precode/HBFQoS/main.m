clear
clc

K = 2;
NRF = 6;
N = 8;
SNR = 10;

cd('..')
cd('..')
cd('./data')
H_test=h5read("setH_K" + string(K) +"_N" + string(N) + "_Ncl8_Nray10_test.mat", '/H');
H_test=permute(H_test, [4, 2, 1, 3]);
H_com =H_test(:,:,:,1) + 1j*H_test(:,:,:,2);
cd('..')
cd('./matlab-precode/HBFQoS')

Time = 1000;
QoS=3;

R1 = zeros(Time,1);
Rate_user=zeros(Time, K);
eta = QoS*ones(K,1); 

for ti = 1:Time

    % H = sqrt(1/2)*(randn(K,N)+1j*randn(K,N));
    H=double(squeeze(H_com(ti, :, :)));

    [U,~,~] = svd(H');  
    FRF = U(:,1:NRF);
    FRF = FRF./abs(FRF);

    W = SCAalgoRF(H*FRF,SNR,eta,FRF);
    R1(ti) = calrate(H,W,K,SNR,eta);

    sigma2 = 10^(-SNR/10);
    Q = abs(H*W).^2;
    D = eye(K).* Q;
    sinr_set = sum(D,2)./(sigma2+sum(Q,2)-sum(D,2));
    R = log2(1+sinr_set);
    Rate_user(ti, :)=R;

    sum(abs(W).^2,'all');
    ti
    
end
Rate_user(isnan(Rate_user)) = 0;
R1(isnan(R1)) = 0;
ratio=sum(Rate_user>QoS, 'all')/Time/K;
performance=mean(R1);


