function irls_cv(dataset, is, fold)
% <============ HEADER =============>
% @brief    : F-fold CV for irls : compute F-measure and plots ROC curve
% @params   : dataset
%             is <- input vectors dimensionality
%            fold <- F-fold number
% @returns  : plots 
% <============ HEADER =============>

%% hyper-parameters 
tt_ratio = 0.6;
opt_names = {'unpenalized','L2'};
l2_penalties = [0.01, 0.1, 1];
n = 1+size(l2_penalties,2);
f_measures = zeros(n+1,fold);
roc_points = zeros(n+1,2);

%% run 
iter = 1;
for l=1:size(opt_names,2)
   switch (string(opt_names(l)))
       case 'L2'
           hp = l2_penalties;
       otherwise
           hp = 1;
   end 
   for k=1:size(hp,2)
      opt = struct('name',string(opt_names(l)),'hp',hp(k));
      for f=1:fold
         [train_set,test_sest] = sample_train_test(dataset,tt_ratio);
         [w, prior, ~] = irls(train_set, is, opt);
         [fmeasure, roc] = cv_binary_classification(w, test_sest, prior, is); 
         f_measures(iter,f) = fmeasure;
         roc_points(iter,:) = roc_points(iter,:) + [roc.TP,roc.FP]/fold;
      end
      iter = iter +1;
   end
end


for f=1:fold
    [train_set,test_sest] = sample_train_test(dataset,tt_ratio);
    [w,~] = laplax(train_set,is);
    [fmeasure, roc] = cv_binary_classification(w, test_sest, 1, is);
    f_measures(iter,f) = fmeasure;
    roc_points(iter,:) = roc_points(iter,:) + [roc.TP,roc.FP]/fold;
end


%% plots 
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,2,1); boxplot(f_measures','Labels',{'IRLS','RIDGE : 0.01','RIDGE = 0.1','RIDGE = 1','Laplace'});
xlabel('Method');
ylabel('F-measure statistics');
title('10-fold cross validation using F-measure');

subplot(1,2,2); 
for i = 1:n+1
    plot(roc_points(i,2),roc_points(i,1),'o','MarkerSize',20,'MarkerEdgeColor',[rand rand rand], 'MarkerFaceColor',[rand rand rand]); hold on;
end

axis([0 1 0 1]);
xlabel('False Positive'); ylabel('True Positive');
title('ROC curve');
l = legend('IRLS','RIDGE, $\lambda_2 = 0.01$', 'RIDGE, $\lambda_2 = 0.1$','RIDGE, $\lambda_2 = 1$','Laplace');
set(l,'Interpreter','latex');


end