clear
close all
clc

%% Figure 3 : Plotting N
theta=-pi:0.01:pi;
r=1;
N_theta=zeros(length(theta),2);
for i=1:length(theta)
    N_theta(i,:)=[r*theta(i)/tan(theta(i)) r*theta(i)];
end

figure
plot(N_theta(:,1),N_theta(:,2))
title('N - Figure(3)')
axis([-4 1 -3 3])
grid on

%% Figure 4
N_x=200;
x=logspace(-3,0,N_x);
alpha_s_Mz_2=0.118;
Mz_2=91.187^2;
N_f=4;
beta_0=11-(2/3)*N_f;
beta_1=102-(38/3)*N_f;
beta_2=2857/2-(5033/18)*N_f+(325/54)*N_f^2;
b_1=beta_1/beta_0;
b_2=beta_2/beta_0;
Q_2=10^4;
Q_0_2=2;
[alpha_s_LO_Q_2,alpha_s_NLO_Q_2,alpha_s_NNLO_Q_2,g_s_LO,g_s_NLO,g_s_NNLO] = alpha_s(beta_0,b_1,b_2,Q_2,Q_0_2,alpha_s_Mz_2,Mz_2)

M=16; % relative accuracy

gama_x=zeros(N_x,1);
for i=1:N_x
    r=2*M/(5*log(1/x(i)));
    sum=0;
    for k=1:M-1
        theta_k=k*pi/M;
        N_theta_k=r*theta_k*(cot(theta_k)+1i);
        sig_theta_k=theta_k+(theta_k*cot(theta_k)-1)*cot(theta_k);
        sum=sum+real(x(i)^(-N_theta_k)*gamma(abs((N_theta_k)))*(1+1i*sig_theta_k));
    end
    gama_x(i)=(r/M)*(0.5*gamma(r)*x(i)^(-r)+sum);
end

figure
semilogx(x,gama_x)
axis([0.001 1 0 2])