function [training_ds, testing_ds] = sample_train_test(ds, tt_ratio)
% <============ HEADER =============>
% @brief    : sample training and testing subsets from dataset
% @params   : name <- dataset's name
%             tt_ratio <- ratio for training size vs testing size
%             ds <- dataset (table type)
% @returns  : training_ds <- sampled training set
%             testing_ds <- sampled training set
% <============ HEADER =============>

m = ceil(size(ds,1)*tt_ratio);
[training_ds, idx] = datasample(ds, m, 'Replace', false);
testing_ds = ds;
testing_ds(idx,:) = [];

end
