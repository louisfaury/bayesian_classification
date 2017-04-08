function irls_cv(dataset, is, fold)
% <============ HEADER =============>
% @brief    : F-fold CV for irls : compute F-measure and plots ROC curve
% @params   : dataset
%             is <- input vectors dimensionality
%            fold <- F-fold number
% @returns  : plots 
% <============ HEADER =============>

%% hyper-parameters 
tt_ratio = 0.9;
opt_names = {'unpenalized','L1','L2'};
l1_penalties = [0.1, 1, 2];
l2_penalties = [0.2, 2, 5];
n = 1+size(l1_penalties,2)+size(l2_penalties,2);
f_measures = zeros(n,fold);
roc_points = zeros(n,2);

%% run 
iter = 1;
for l=1:size(opt_names,2)
   switch (string(opt_names(l)))
       case 'L1'
           hp = l1_penalties;
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


%% plots 
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,2,1); boxplot(f_measures','Labels',{'IRLS','LASSO = 0.1','LASSO = 1','LASSO = 2','RIDGE = 0.2','RIDGE = 2','RIDGE = 5'});
xlabel('Method');
ylabel('F-measure statistics');
title('10-fold cross validation using F-measure');

subplot(1,2,2); 
for i = 1:n
    plot(roc_points(i,2),roc_points(i,1),'o','MarkerSize',20,'MarkerEdgeColor',[rand rand rand], 'MarkerFaceColor',[rand rand rand]); hold on;
end
axis([0 1 0 1]);
l = legend('IRLS','LASSO, $\lambda_1 = 0.1$','LASSO, $\lambda_1 = 1$','LASSO, $\lambda_1 = 2$', 'RIDGE, $\lambda_2 = 0.2$', 'RIDGE, $\lambda_2 = 2$','RIDGE, $\lambda_2 = 5$');
set(l,'Interpreter','latex');


end