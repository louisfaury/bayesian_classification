function [training_ds, testing_ds] = sample_train_test(ds, tt_ratio)
% <============ HEADER =============>
% @brief    : sample training and testing subsets from dataset
% @params   : name <- dataset's name
%             tt_ratio <- ratio for training size vs testing size
%             ds <- dataset (table type)
% @returns  : training_ds <- sampled training set
%             testing_ds <- sampled training set
%             in_size <- dimensionality of inputs
% <============ HEADER =============>


%normalized_data = [table2array(ds(:,1:in_size)) , strcmp(ds{:,in_size+1},'malign')];
m = ceil(size(ds,1)*tt_ratio);
[training_ds, idx] = datasample(normalized_data, m, 'Replace', false);
testing_ds = normalized_data;
testing_ds(idx,:) = [];

end
