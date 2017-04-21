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
%explained_variance = trace(D(1:retained_dim,1:retained_dim))/ trace(D); %=0.76
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
 subplot(3,3,1); histogram(dataset(ixp,1),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,1),'FaceColor','green'); title('Numbers of pregnancy'); legend('Positive','Negative');
 subplot(3,3,2); histogram(dataset(ixp,2),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,2),'FaceColor','green'); title('Plasma Glucose Concentration'); legend('Positive','Negative');
 subplot(3,3,3); histogram(dataset(ixp,3),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,3),'FaceColor','green'); title('Diastolic blood pressure'); legend('Positive','Negative');
 subplot(3,3,4); histogram(dataset(ixp,4),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,4),'FaceColor','green'); title('Triceps skin fold thickness'); legend('Positive','Negative');
 subplot(3,3,5); histogram(dataset(ixp,5),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,5),'FaceColor','green'); title('2-Hour serum insulin'); legend('Positive','Negative');
 subplot(3,3,6); histogram(dataset(ixp,6),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,6),'FaceColor','green'); title('Body mass index'); legend('Positive','Negative');
 subplot(3,3,7); histogram(dataset(ixp,7),'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,7),'FaceColor','green'); title('iabetes pedigree function'); legend('Positive','Negative');
 subplot(3,3,8); histogram(dataset(ixp,8),11,'FaceColor','red','FaceAlpha',0.8); hold on; histogram(dataset(ixn,8),11,'FaceColor','green'); title('Age'); legend('Positive','Negative');


end