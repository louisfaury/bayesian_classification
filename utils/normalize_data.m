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
    otherwise
        error('Unknwon dataset');
end

n_ds = [table2array(ds(:,1:input_size)) , strcmp(ds{:,input_size+1},'malign')];

end