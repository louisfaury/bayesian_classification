function visualize_solution(w, ds, input_size, lc, opt)
% <============ HEADER =============>
% @brief    : plots the decision boundary in projected PCA space  
% @params   : w <- learned parameter vector
%             ds <- dataset
%             in_size <- input size in ds 
%             lc <- learning curve array for given method 
%             opt <- penalization info for titles
% <============ HEADER =============>

% Plot learning curve 
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,2,1); plot(lc,'LineWidth',3,'Color',[0.3 0.3 0.8]);
title('Iterative Reweighted Least Square Learning Curve');
xlabel('Iterations');
ylabel('Loss (cross-entropy)');
legend('IRLS learning curve');

% projection rule  %TODO : watch out 0 term for learned proj. vector 
data = ds(:,1:input_size);
empirical_cov = (data-mean(data))'*(data-mean(data));
[V,~] = eigs(empirical_cov,input_size,'sm');
retained_dim = 2;
proj_matrix = V(:,1:retained_dim)';
proj_data = (data-mean(data))*proj_matrix';
proj_w = proj_matrix*w;
% 
if (retained_dim==2)
    hold on;
    for i=1:size(data,1)
        if (ds(i,input_size+1))
            color = [0.9, 0.2, 0.3];
            subplot(1,2,2); hold on; m = plot(proj_data(i,1),proj_data(i,2),'.','MarkerSize',15,'Color',color);
        else
            color = [0.1, 0.9, 0.4];
            subplot(1,2,2); hold on; b = plot(proj_data(i,1),proj_data(i,2),'.','MarkerSize',15,'Color',color);
        end
    end
end
x1 = [-10;5];
x2 = -proj_w(1)/proj_w(2) * x1;

subplot(1,2,2); hold on; h = plot(x1,x2,'-.','LineWidth',3,'Color',[0.4, 0.5, 0.9]);

title('PCA analysis for dataset');
xlabel('$e_1$','Interpreter','latex');
ylabel('$e_2$','Interpreter','latex');
legend([m,b,h],'Malign','Benign','Decision boundary : IRLS');



% plots 

