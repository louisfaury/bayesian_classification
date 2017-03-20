function proj_matrix = visualize_data(dataset,input_size)
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
proj_data = (data-mean(data))*proj_matrix';

%plots
if (retained_dim==2)
    figure('units','normalized','outerposition',[0 0 1 1])
    hold on;
    for i=1:size(dataset,1)
        if (dataset(i,input_size+1))
            color = [1, 0.5, 0.4];
            subplot(1,2,1); hold on; m = plot(proj_data(i,1),proj_data(i,2),'.','MarkerSize',15,'Color',color);
        else
            color = [0.4, 0.5, 0.9];
            subplot(1,2,1); hold on; b = plot(proj_data(i,1),proj_data(i,2),'.','MarkerSize',15,'Color',color);
        end
    end
end

title('PCA analysis for dataset');
xlabel('$e_1$','Interpreter','latex');
ylabel('$e_2$','Interpreter','latex');
legend([m,b],'Malign','Benign');


%% Hist plots 
m = [];
b = [];

for i=1:size(dataset,1)
    % really unefficient but only way to have nice legends
    for j = 1:input_size
       if (dataset(i,input_size+1))
           m = [m;[j,dataset(i,j)+0.2*randn]];
       else
           b = [b;[j,dataset(i,j)+0.2*randn]];
       end
    end
end

subplot(1,2,2); hold on; scatter(m(:,1),m(:,2),60,'MarkerFaceColor',[1, 0.5, 0.4],'MarkerEdgeColor',[1, 0.5, 0.4],'MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);
subplot(1,2,2); hold on; scatter(b(:,1),b(:,2),60,'MarkerFaceColor',[0.4, 0.5, 0.9],'MarkerEdgeColor',[0.4, 0.5, 0.9],'MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);
legend('Malign','Benign');
xlabel('Input index');
title('Scatter plot representation of data');
end