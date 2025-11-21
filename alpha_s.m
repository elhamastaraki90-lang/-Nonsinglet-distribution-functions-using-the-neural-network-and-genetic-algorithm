function [a_s_LO_Q_2,a_s_NLO_Q_2,a_s_NNLO_Q_2,g_s_LO,g_s_NLO,g_s_NNLO] = alpha_s(beta_0,b_1,b_2,Q_2,Q_0_2,alpha_s_Mz_2,Mz_2)
    % Calculation of alpha_s_Q^2 for LO(Leading Order),NLO(Next-to-Leading Order),NNLO(Next-to-Next-to-Leading Order)
    
    a_s_LO_Q_2=alpha_s_Mz_2/(1+beta_0*alpha_s_Mz_2*log(Q_2/Mz_2));
    a_s_LO_Q_0_2=alpha_s_Mz_2/(1+beta_0*alpha_s_Mz_2*log(Q_0_2/Mz_2));

    a_s_NLO_Q_2=a_s_LO_Q_2*(1-b_1*a_s_LO_Q_2*log(1+beta_0*alpha_s_Mz_2*log(Q_2/Mz_2)));
    a_s_NLO_Q_0_2=a_s_LO_Q_0_2*(1-b_1*a_s_LO_Q_0_2*log(1+beta_0*alpha_s_Mz_2*log(Q_0_2/Mz_2)));
    
    a_s_NNLO_Q_2=a_s_LO_Q_2*(1+a_s_LO_Q_2*(a_s_LO_Q_2-alpha_s_Mz_2)*(b_2-b_1^2)+a_s_NLO_Q_2*b_1*log(a_s_NLO_Q_2/alpha_s_Mz_2));
    a_s_NNLO_Q_0_2=a_s_LO_Q_0_2*(1+a_s_LO_Q_0_2*(a_s_LO_Q_0_2-alpha_s_Mz_2)*(b_2-b_1^2)+a_s_NLO_Q_0_2*b_1*log(a_s_NLO_Q_0_2/alpha_s_Mz_2));
    
    % Calculation of Gamma_s for LO(Leading Order),NLO(Next-to-Leading Order),NNLO(Next-to-Next-to-Leading Order)
    gama_0_N=1;
    gama_1_N=1;
    gama_2_N=1;
        
    U_NS_1_N=-(gama_1_N-b_1*gama_0_N)/beta_0;
    U_NS_2_N=-(gama_2_N/beta_0-b_1*U_NS_1_N-b_2*gama_0_N/beta_0-U_NS_1_N^2)/2;
    
    g_s_LO=((a_s_LO_Q_2/a_s_LO_Q_0_2)^(-gama_0_N/beta_0))*(1-(a_s_LO_Q_2-a_s_LO_Q_0_2)*U_NS_1_N/(4*pi)+...
        (a_s_LO_Q_2^2-a_s_LO_Q_0_2^2)*U_NS_2_N/(16*pi^2)-a_s_LO_Q_2*a_s_LO_Q_0_2*U_NS_1_N^2);
    
    g_s_NLO=((a_s_NLO_Q_2/a_s_NLO_Q_0_2)^(-gama_0_N/beta_0))*(1-(a_s_NLO_Q_2-a_s_NLO_Q_0_2)*U_NS_1_N/(4*pi)+...
        (a_s_NLO_Q_2^2-a_s_NLO_Q_0_2^2)*U_NS_2_N/(16*pi^2)-a_s_NLO_Q_2*a_s_NLO_Q_0_2*U_NS_1_N^2);
    
    g_s_NNLO=((a_s_NNLO_Q_2/a_s_NNLO_Q_0_2)^(-gama_0_N/beta_0))*(1-(a_s_NNLO_Q_2-a_s_NNLO_Q_0_2)*U_NS_1_N/(4*pi)+...
        (a_s_NNLO_Q_2^2-a_s_NNLO_Q_0_2^2)*U_NS_2_N/(16*pi^2)-a_s_NNLO_Q_2*a_s_NNLO_Q_0_2*U_NS_1_N^2);
end

