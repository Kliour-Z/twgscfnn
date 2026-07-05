%% method 3 先划分测试训练集再计算相似度，用logistic模型将特征融合
clc ,clear,close all
% 代码参数设置
method = 3;
a = 0.8; % 训练集占比
nn = 1; % 并行池数量

%% 数据准备
% 示例数据
% UAI = [1 0 0 0 0 0 0;0 1 1 0 1 1 0;0 1 1 0 1 0 0;0 0 0 1 0 0 1;0 1 1 0 1 1 0;0 1 0 0 1 1 0;0 0 0 1 0 0 1];% 获取训练集的修正邻接矩阵UAI
% A = UAI - eye(size(UAI,1));
finame ='USAir';
load(finame+".mat",'net');
A = full(net);
UAI = A + eye(size(A,1));
UAI = sparse(UAI);
clear net
N = size(A,1);
if isequal(A, A')
    disp('矩阵是对称矩阵');
else
    disp('矩阵不是对称矩阵');
    return
end
if max(max(A)) ~= 1
    disp('矩阵有问题');
    return
else
    disp('继续');
end

%% 划分训练测试集
tic
S = sparse(A);
[~, ~, train_edges, test_edges] = split_graph_undirected(S, a);
clear("S")
%% 只用训练集生成防止泄露(注释掉就恢复原来的)
% G = graph(train_edges(:,1), train_edges(:,2) ,[], N);
% A = adjacency(G);
% UAI = adjacency(G)+eye(size(A,1));
%% 计算粒概念空间
[AllGranularconcept,concepttime] = GranularConceptualSpace(UAI,N,nn);
disp('计算概念空间的时间：')
disp(concepttime)




%% 计算全局相似度
katz_centrality = FastKatzCentrality(A, 0.1, 1);
% katz_centrality = KatzCentrality(A, 0.1, 0, 200); % Katz中心性
katz_normal = katz_centrality / max(katz_centrality); % 百分比归一化
time1 = toc;
clear A
%% logistic模型训练部分 %%
% 按照train_edges节点的顺序提取AllGranularconcept需要用的对象集和正域
obj1_T = AllGranularconcept(train_edges(:,1),1);
obj2_T = AllGranularconcept(train_edges(:,2),1);
pos1_T = AllGranularconcept(train_edges(:,1),2);
pos2_T = AllGranularconcept(train_edges(:,2),2);
% 按照train_edges节点的顺序提取katz_normal中需要用的节点katz值
k1_T = katz_normal(train_edges(:,1));
k2_T = katz_normal(train_edges(:,2));
% 并行设置
myCluster = parcluster('Processes');
delete(myCluster.Jobs);
delete(gcp('nocreate'));  % 关闭现有池
parpool('local', nn); % 限制为nn个进程
tic
parfor i = 1:size(train_edges,1)
    if isequal(cell2mat(obj1_T(i,:)),cell2mat(obj2_T(i,:)))
            conceptsimilarity_T = 1;
    else
            conceptsimilarity_T= concept_similarity( ...
                cell2mat(pos1_T(i,:)), cell2mat(pos2_T(i,:)));
    end
    k_train(i,:) = max(k1_T(i), k2_T(i));
    conceptsimilarity_train(i,:) = conceptsimilarity_T;
    if i==floor(size(train_edges,1)/4)
        disp('25%')
    elseif i==floor(size(train_edges,1)/2)
        disp('50%')
    elseif i==floor(size(train_edges,1))
        disp('100%')
    end
end
% 训练集特征构建
X_train = [conceptsimilarity_train, k_train]; % N×2矩阵
y_train = train_edges(:,3);                   % N×1二值标签
% 使用L2正则化逻辑回归
model = fitclinear(X_train, y_train, ...
    'Learner', 'logistic', ...
    'Regularization', 'lasso', ...
    'Lambda', 0.01); % 正则化强度
time2 = toc;
% 获取学习到的参数
w = [model.Beta(1),model.Beta(2),model.Bias];
% 输出参数
disp('训练时间：')
disp(time2)
disp('参数：'); 
disp('w1,w2,bias:'); disp(w); 


%% 测试 %%
% 按照test_edges节点的顺序提取AllGranularconcept需要用的对象集和正域
obj1 = AllGranularconcept(test_edges(:,1),1);
obj2 = AllGranularconcept(test_edges(:,2),1);
pos1 = AllGranularconcept(test_edges(:,1),2);
pos2 = AllGranularconcept(test_edges(:,2),2);
% 按照test_edges节点的顺序提取katz_normal中需要用的节点katz值
k1 = katz_normal(test_edges(:,1));
k2 = katz_normal(test_edges(:,2));
clear AllGranularconcept katz_centrality katz_normal
tic
parfor i = 1:size(test_edges,1)
    if isequal(cell2mat(obj1(i,:)),cell2mat(obj2(i,:)))
            conceptsimilarity = 1;
    else
            conceptsimilarity= concept_similarity( ...
                cell2mat(pos1(i,:)), cell2mat(pos2(i,:)));
    end
    % 测试集特征构建
    k_test(i,:) = max(k1(i), k2(i));
    conceptsimilarity_test(i,:) = conceptsimilarity;
end
% 测试集特征构建
X_test = [conceptsimilarity_test, k_test];  % N×2矩阵
y_test = test_edges(:,3);                   % N×1二值标签
% 测试集预测
[test_labels, test_scores] = predict(model, X_test);
test_probs = test_scores(:,2); 
ptime = toc;
disp('测试集计算时间：')
disp(ptime)
totoaltime = time1 + time2 + ptime + concepttime;

%% 性能评估,指标计算
[X, Y, T, AUC] = perfcurve(y_test, test_probs, 1);
% 绘制ROC曲线
figure(1);
plot(X, Y, 'b-', 'LineWidth', 2, 'DisplayName', 'ROC Curve');
hold on;
plot([0 1], [0 1], 'k--', 'DisplayName', 'Random Guess');
% 标注阈值点及最佳点
thresholds = [0.2, 0.5, 0.8];
colors = {'r*', 'g*', 'm*'};
for i = 1:length(thresholds)
    [~, idx] = min(abs(T - thresholds(i)));
    plot(X(idx), Y(idx), colors{i}, 'MarkerSize', 10, 'DisplayName', sprintf('%.1f Threshold', thresholds(i)));
end
youden_index = Y - X;
[~, optimal_idx] = max(youden_index);
optimal_threshold = T(optimal_idx);
plot(X(optimal_idx), Y(optimal_idx), 'ko', 'MarkerSize', 10, 'DisplayName', 'Optimal Threshold');
% 图形美化
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(['ROC Curve | AUC=' num2str(AUC, '%.4f')]);
legend('show', 'Location', 'southeast');
grid on; set(gca, 'FontSize', 12);
hold off;
% 基于最佳阈值计算指标
figure(2)
pred_labels = double(test_probs >= optimal_threshold);
conf_mat = confusionmat(y_test, pred_labels, 'Order', [1, 0]);
disp(confusionchart(conf_mat));
% 统计量计算（含鲁棒性检查）
TP = conf_mat(1,1); FN = conf_mat(1,2);
FP = conf_mat(2,1); TN = conf_mat(2,2);
accuracy = (TP + TN) / (TP + TN + FP + FN);
precision = TP / (TP + FP + eps);
recall = TP / (TP + FN + eps);
f1 = 2 * (precision * recall) / (precision + recall + eps);
mse = mean((test_probs - y_test).^2);
% 输出结果
fprintf('=== 性能评估 ===\n');
fprintf('AUC值: %.4f\n', AUC);
fprintf('最佳阈值: %.3f\n', optimal_threshold);
fprintf('准确率 Accuracy: %.4f\n', accuracy);
fprintf('精确率 Precision: %.4f\n', precision);
fprintf('召回率 Recall: %.4f\n', recall);
fprintf('F1值: %.4f\n', f1);
fprintf('MSE: %.4f\n', mse);
fprintf('时间: %.4f\n', totoaltime);
%% 保存输出和相应指标
targetFolder = 'output_logistic'; % 替换为你的目标路径
mkdir(targetFolder);
similarity_4 = round(test_probs, 4);  % 保留 4 位小数
savePath = targetFolder + "\" +finame + "_testp.txt";
writematrix(similarity_4,savePath);
savePath = targetFolder + "\" +finame + "_testy.txt";
writematrix(test_edges(:,3),savePath);
% 指定目标文件夹和文件名
filename = "optimal_threshold" + finame;
save(fullfile(targetFolder, filename), 'optimal_threshold', '-v7.3');
% 将指标写入txt
fileID = fopen(fullfile(targetFolder,'results_method3.txt'), 'a');
fprintf(fileID, '%s\n',finame + ":");
fprintf(fileID, 'w: %.4f\n', w);
fprintf(fileID, 'precision: %.4f\n', precision);
fprintf(fileID, 'accuracy: %.4f\n', accuracy);
fprintf(fileID, 'recall: %.4f\n', recall);
fprintf(fileID, 'F1: %.4f\n', f1);
fprintf(fileID, 'MSE: %.4f\n', mse);
fprintf(fileID, 'AUC: %.4f\n', AUC);
fprintf(fileID, 'time: %.4f\n', totoaltime);
fprintf(fileID, '\n');