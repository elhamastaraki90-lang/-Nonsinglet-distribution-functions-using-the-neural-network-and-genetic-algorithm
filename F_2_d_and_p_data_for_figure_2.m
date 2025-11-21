clear
clc

load 'data_figure_2_NMC_BCDMS.mat';
%% NMC Data
x_NMC=[x_NMC_p;x_NMC_d];
Q_NMC=[Q_2_NMC_p;Q_2_NMC_d];
bt=zeros(length(x_NMC),1);
for i=1:length(x_NMC)
    if Q_NMC(i)>3
        bt(i)=1;
    end
end
figure
loglog(x_NMC(bt==1),Q_NMC(bt==1),'b+') % NMC Points
ylabel('Q^2[GeV^2]')
xlabel('x')
hold on

Par_p=[-0.02778,2.926,1.0362,-1.840,8.123,-13.074,6.215,0.285,-2.694,0.0188,0.0274,-1.413,9.366,-37.79,47.10];
Par_d=[-0.04858,2.863,0.8367,-2.532,9.145,-12.504,5.473,-0.008,-2.227,0.0551,0.0570,-1.509,8.553,-31.20,39.98];

Q_0_2=20;
Gama_2=0.250^2;
F_2_p_NMC=zeros(length(x_NMC),1);
F_2_d_NMC=zeros(length(x_NMC),1);
for i=1:x_NMC(end)
    A=(x_NMC_p(i)^Par_p(1))*((1-x_NMC_p(i))^Par_p(2))*(Par_p(3)+Par_p(4)*(1-x_NMC_p(i))+Par_p(5)*((1-x_NMC_p(i))^2)+Par_p(6)*((1-x_NMC_p(i))^3)+Par_p(7)*((1-x_NMC_p(i))^4));
    B=Par_p(8)+Par_p(9)*x_NMC_p(i)+Par_p(10)/(x_NMC_p(i)+Par_p(11));
    C=Par_p(12)*x_NMC_p(i)+Par_p(13)*x_NMC_p(i)^2+Par_p(14)*x_NMC_p(i)^3+Par_p(15)*x_NMC_p(i)^4;
    F_2_p_NMC(i)=A*(log(Q_NMC_p(i)^2/Gama_2)/log(Q_0^2/Gama_2))^B*(1+C/Q_NMC_p(i)^2);
    
    A=(x_NMC_d(i)^Par_p(1))*((1-x_NMC_d(i))^Par_p(2))*(Par_p(3)+Par_p(4)*(1-x_NMC_d(i))+Par_p(5)*((1-x_NMC_d(i))^2)+Par_p(6)*((1-x_NMC_d(i))^3)+Par_p(7)*((1-x_NMC_d(i))^4));
    B=Par_p(8)+Par_p(9)*x_NMC_d(i)+Par_p(10)/(x_NMC_d(i)+Par_p(11));
    C=Par_p(12)*x_NMC_d(i)+Par_p(13)*x_NMC_d(i)^2+Par_p(14)*x_NMC_d(i)^3+Par_p(15)*x_NMC_d(i)^4;
    F_2_d_NMC(i)=A*(log(Q_NMC_d(i)^2/Gama_2)/log(Q_0^2/Gama_2))^B*(1+C/Q_NMC_d(i)^2);
end
%% BCDMS Data
x_BCDMS=[x_BCDMS_p;x_BCDMS_d];
Q_BCDMS=[Q_2_BCDMS_p;Q_2_BCDMS_d];

loglog(x_BCDMS,Q_BCDMS,'rx') % BCDMS Points
ylabel('Q^2[GeV^2]')
xlabel('x')
hold on
loglog(x_NMC(bt==0),Q_NMC(bt==0),'rs') % Excluded Points
hold off
legend('NMC','BCDMS','Excluded','Location','north')

F_2_p_BCDMS=zeros(length(x_BCDMS_p),1);
F_2_d_BCDMS=zeros(length(x_BCDMS_d),1);
for i=1:x_BCDMS(end)
    A=(x_BCDMS_p(i)^Par_p(1))*((1-x_BCDMS_p(i))^Par_p(2))*(Par_p(3)+Par_p(4)*(1-x_BCDMS_p(i))+Par_p(5)*((1-x_BCDMS_p(i))^2)+Par_p(6)*((1-x_BCDMS_p(i))^3)+Par_p(7)*((1-x_BCDMS_p(i))^4));
    B=Par_p(8)+Par_p(9)*x_BCDMS_p(i)+Par_p(10)/(x_BCDMS_p(i)+Par_p(11));
    C=Par_p(12)*x_BCDMS_p(i)+Par_p(13)*x_BCDMS_p(i)^2+Par_p(14)*x_BCDMS_p(i)^3+Par_p(15)*x_BCDMS_p(i)^4;
    F_2_p_BCDMS(i)=A*(log(Q_BCDMS_p(i)^2/Gama_2)/log(Q_0^2/Gama_2))^B*(1+C/Q_BCDMS_p(i)^2);
    
    A=(x_BCDMS_d(i)^Par_p(1))*((1-x_BCDMS_d(i))^Par_p(2))*(Par_p(3)+Par_p(4)*(1-x_BCDMS_d(i))+Par_p(5)*((1-x_BCDMS_d(i))^2)+Par_p(6)*((1-x_BCDMS_d(i))^3)+Par_p(7)*((1-x_BCDMS_d(i))^4));
    B=Par_p(8)+Par_p(9)*x_BCDMS_d(i)+Par_p(10)/(x_BCDMS_d(i)+Par_p(11));
    C=Par_p(12)*x_BCDMS_d(i)+Par_p(13)*x_BCDMS_d(i)^2+Par_p(14)*x_BCDMS_d(i)^3+Par_p(15)*x_BCDMS_d(i)^4;
    F_2_d_BCDMS(i)=A*(log(Q_BCDMS_d(i)^2/Gama_2)/log(Q_0^2/Gama_2))^B*(1+C/Q_BCDMS_d(i)^2);
end

%% Table 1 Data
% NMC
x_NMC_without_Excluded=x_NMC(bt==1);
Q_NMC_without_Excluded=Q_NMC(bt==1);

min_x_NMC=min(x_NMC_without_Excluded);
max_x_NMC=max(x_NMC_without_Excluded);

min_Q_NMC=min(Q_NMC_without_Excluded);
max_Q_NMC=max(Q_NMC_without_Excluded);

N_dat_NMC=length(x_NMC_without_Excluded);
disp(['NMC with ',num2str(N_dat_NMC),' data points'])
disp(['x range : ',num2str(min_x_NMC),'-',num2str(max_x_NMC)])
disp(['Q^2 range : ',num2str(min_Q_NMC),'-',num2str(max_Q_NMC)])
% BCDMS
min_x_BCDMS=min(x_BCDMS);
max_x_BCDMS=max(x_BCDMS);

min_Q_BCDMS=min(Q_BCDMS);
max_Q_BCDMS=max(Q_BCDMS);

N_dat_BCDMS=length(x_BCDMS);
disp(['BCDMS with ',num2str(N_dat_BCDMS),' data points'])
disp(['x range : ',num2str(min_x_BCDMS),'-',num2str(max_x_BCDMS)])
disp(['Q^2 range : ',num2str(min_Q_BCDMS),'-',num2str(max_Q_BCDMS)])

save('F_2_p_d','F_2_p_NMC','F_2_d_NMC','F_2_p_BCDMS','F_2_d_BCDMS')