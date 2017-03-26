function irls_solve_cv(dataset, is, fold)
% <============ HEADER =============>
% @brief    : F-fold CV for irls : compute F-measure and plots ROC curve
% @params   : dataset
%             is <- input vectors dimensionality
%            fold <- F-fold number
% @returns  : plots 
% <============ HEADER =============>

%% hyper-parameters 
tt_ratio = 0.9;
opt_names = ['unpenalized','L1','L2'];
l1_penalties = [1, 2, 5];
l2_penalties = [5, 10, 20];
f_measures = zeros(size(opt_names,2)+size(l1_penalties,2)+size(l2_penalties,2),fold);
roc_points = zeros(size(opt_names,12)+size(l1_penalties,2)+size(l2_penalties,2),fold);

%% run 
for l=1:size(opt_names,2
    iter = 1;
   switch (opt(l))
       case 'L1'
           hp = l1_penalties;
       case 'L2'
           hp = l2_penalties;
       otherwise
           hp = 1;
   end 
   for k=1:size(hp,2)
      opt = struct('name',opt_names(l),'hp',hp(k));
      for f=1:fold
         [training,testing] = sample_train_test(dataset,tt_ratio);
         [w, prior, ~] = irls(training, is, opt);
         [fmeasure, roc] = cv_binary_classification(w, testing, prior); %% TODO 
         f_measures(iter,f) = fmeasure;
         roc_points(iter,f) = roc;
      end
      iter = iter +1;
   end
   
end

end