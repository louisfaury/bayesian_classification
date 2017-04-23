function visualize_data(dataset,input_size)
% <============ HEADER =============>
% @brief    : visualization tools for given dataset : 
%                   -> pca
%                   -> histogram plots of mean values 
% @params   : dataset
%             input_size <- size of input vector in the dataset
% <============ HEADER =============>


%% PCA 
data = dataset(:,1:input_size);
data = data ./ sqrt(var(data));
empirical_cov = (data-mean(data))'*(data-mean(data));
[V,D] = eigs(empirical_cov,input_size,'sm');
retained_dim = 2;
%explained_variance = trace(D(1:retained_dim,1:retained_dim))/ trace(D); %70%
proj_matrix = V(:,1:retained_dim)';
proj_data = (data-mean(data))*proj_matrix';

%plots
if (retained_dim==2)
    figure('units','normalized','outerposition',[0 0 1 1])
    hold on;
    for i=1:size(dataset,1)
        if (dataset(i,input_size+1))
            color = [0.9, 0.2, 0.3];
            subplot(1,2,1); hold on; m = plot(proj_data(i,1),proj_data(i,2),'.','MarkerSize',15,'Color',color);
        else
            color = [0.1, 0.9, 0.4];
            subplot(1,2,1); hold on; b = plot(proj_data(i,1),proj_data(i,2),'.','MarkerSize',15,'Color',color);
        end
    end
end

title('PCA analysis for dataset');
xlabel('$e_1$','Interpreter','latex');
ylabel('$e_2$','Interpreter','latex');
legend([m,b],'Positive','Negative');


%% Hist plots 
 ixp = find(dataset(:,input_size+1));
 ixn = find(~dataset(:,input_size+1));
 figure('units','normalized','outerposition',[0 0 1 1]);
 subplot(6,4,1); histogram(dataset(ixp,1),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,1),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,2); histogram(dataset(ixp,2),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,2),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,3); histogram(dataset(ixp,3),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,3),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,4); histogram(dataset(ixp,4),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,4),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,5); histogram(dataset(ixp,5),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,5),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,6); histogram(dataset(ixp,6),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,6),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,7); histogram(dataset(ixp,7),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,7),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,8); histogram(dataset(ixp,8),11,'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,8),11,'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,9); histogram(dataset(ixp,9),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,9),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,10); histogram(dataset(ixp,10),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,10),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,11); histogram(dataset(ixp,11),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,11),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,12); histogram(dataset(ixp,12),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,12),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,13); histogram(dataset(ixp,13),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,13),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,14); histogram(dataset(ixp,14),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,14),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,15); histogram(dataset(ixp,15),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,15),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,16); histogram(dataset(ixp,16),11,'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,16),11,'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,17); histogram(dataset(ixp,17),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,17),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,18); histogram(dataset(ixp,18),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,18),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,19); histogram(dataset(ixp,19),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,19),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,20); histogram(dataset(ixp,20),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,20),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,21); histogram(dataset(ixp,21),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,21),'FaceColor','green'); legend('Positive','Negative');
 subplot(6,4,22); histogram(dataset(ixp,22),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,22),'FaceColor','green'); legend('Positive','Negative');

 
 


end