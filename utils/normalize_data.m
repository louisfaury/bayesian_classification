function [n_ds, input_size] = normalize_data(name, ds)
% <============ HEADER =============>
% @brief    : normalize the mixed data retrieved in dataset (string go to integer)
% @params   : name <- dataset's name
%             ds <- dataset (table type)
% @returns  : n_ds <- normalized dataset
%             input_size <- size of input in dataset
% <============ HEADER =============>

switch (name)
    case 'bcw'
        input_size = 9;
        n_ds = [table2array(ds(:,1:input_size)) , strcmp(ds{:,input_size+1},'malign')];
    case 'ipls'
        input_size = 10;
        n_ds = [table2array(ds(:,1)) , strcmp(ds{:,2},'Male'), table2array(ds(:,3:input_size)),double(ds{:,input_size+1}==2)];
        n_ds(:,1:input_size) = n_ds(:,1:input_size) ./ sqrt(var(n_ds(:,1:input_size)));
    case 'pidd'
        input_size = 8;
        n_ds = table2array(ds(:,1:input_size+1));
        n_ds(:,1:input_size) = (n_ds(:,1:input_size)-mean(n_ds(:,1:input_size))) ./ sqrt(var(n_ds(:,1:input_size)));
        n_ds(randperm(size(n_ds,1),400),:) = [];
    case 'piddr'
        input_size = 8;
        n_ds = table2array(ds(:,1:input_size+1));
    case 'synth'
        input_size = 2;
        n_ds = table2array(ds(:,1:input_size+1));
    case 'park'
        input_size = 22;
        n_ds = table2array(ds(:,1:input_size+1));
        n_ds(:,1:input_size) = (n_ds(:,1:input_size)-mean(n_ds(:,1:input_size))) ./ sqrt(var(n_ds(:,1:input_size)));
        
        
    otherwise
        error('Unknown dataset');
end




end