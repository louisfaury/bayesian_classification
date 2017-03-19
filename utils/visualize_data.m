function visualize_data(dataset,input_size)
% <============ HEADER =============>
% @brief    : visualization tools for given dataset : 
%                   -> pca
%                   -> histogram plots of mean values 
% @params   : dataset
%             inut_size <- size of input vector in the dataset
% <============ HEADER =============>


%% PCA 
data = dataset(:,1:input_size);
empirical_cov = (data-mean(data))'*(data-mean(data));
[V,D] = eigs(empirical_cov,input_size,'sm');
retained_dim = 2;
%explained_variance = trace(D(1:retained_dim,1:retained_dim))/ trace(D); %=0.76
proj_matrix = V(:,1:retained_dim)';
proj_data = data*proj_matrix';

%plots
if (retained_dim==2)
    figure;
    hold on;
    for i=1:size(dataset,1)
        if (dataset(i,input_size+1))
            color = [0.9, 0.1, 0.2];
            m = plot(proj_data(i,1),proj_data(i,2),'.','MarkerSize',15,'Color',color);
        else
            color = [0.1, 0.9, 0.2];
            b = plot(proj_data(i,1),proj_data(i,2),'.','MarkerSize',15,'Color',color);
        end
    end
end

title('PCA analysis for dataset');
xlabel('$e_1$','Interpreter','latex');
ylabel('$e_2$','Interpreter','latex');
legend([m,b],'Malign','Benign');


end