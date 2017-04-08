function [fmeasure, roc] = cv_binary_classification(w, test_set, prior, is)
% <============ HEADER =============>
% @brief    : returns metrics regarding classification on a test set
%             computes the 
%                       F_measure =  2*P*R/(P+R)
%                       ROC value = (TP,VP)
% @params   : w <- learned parameter vector
%             test_set 
%             prior : prior for {+1} class
%             is <- input size in ds 

% <============ HEADER =============>

% compute estimates 
yest = compute_output('logistic_sigmoid', w(1:is), w(is+1),test_set(:,1:is), 'linear');
t_test = test_set(:,is+1);
t_est = round(yest+0.05);

% computes TP and FP 
TP = sum(double(t_test  & t_est));
FP = sum(double(~t_test & t_est));

% computes recall and precision, fmeasure and roc value 
if (sum(double(t_test==1))) > 0
    R = TP / sum(double(t_test==1));
    P = TP / (TP + FP);
    roc.TP = TP / sum(double(t_test==1)) ; 
    roc.FP = FP / sum(double(t_test==0));
    fmeasure = 2*R*P/(R+P);
else
    fmeasure = 1;
    roc.TP = 1;
    roc.FP = 0;
end



end