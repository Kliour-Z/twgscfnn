%% method9 概念认知学习+随机森林
function method9(finame)
% 代码参数设置
method = "TWGSCRF";
L = 0.8; % 训练集占比
nn = 1; % 并行池数量
numofpath = 10;
lenthofpath = 10;
targetFolder = "D:\all\result\output_" + method;
mkdir(targetFolder);
%% 数据准备

load(finame + ".mat", 'net');
A = full(net);
clear net
N = size(A,1);

%% 划分训练测试集
S = sparse(A);
[~, ~, train_edges, test_edges] = split_graph_undirected(S, L);
overlap = intersect(test_edges, train_edges, 'rows');
clear("S")

%% 生成训练集对应的概念空间
train_pos = train_edges(train_edges(:,3) == 1, :);
G = graph(train_pos(:,1), train_pos(:,2), [], N);
A = adjacency(G);
UAI = adjacency(G)+eye(size(A,1));
[AllGranularconcept,concepttime] = GranularConceptualSpace(UAI,N,nn);
disp('计算概念空间的时间：')
disp(concepttime)
% 检测是否数据泄露
if ~isempty(overlap)
    error('训练与测试负样本存在重叠');
end
% 并行设置
myCluster = parcluster('Processes');
delete(myCluster.Jobs);
delete(gcp('nocreate'));  % 关闭现有池
parpool('local', nn); % 限制为nn个进程
%% 计算度生成路径，这里每个路径重复经过的节点只算一次
% 提取所有节点的正域
tic
pos = AllGranularconcept(:,2);
pathcount = zeros(N,N);
count = 0;
parfor i = 1:N
    nodedegree(i,1) = length(cell2mat(pos(i,:))) ;% 这里算出来的度为真实度加1  
    temp2 = [];
    for j = 1:numofpath
        temp = zeros(N,1);
        jpath = pathgeneration(i, lenthofpath, pos); 
        temp(jpath) = 1;
        pathcount(:,i) = pathcount(:,i) + temp; 
        temp2 = union(temp2,jpath);
    end
    pathnodefori{i} = temp2;
end
pathcount = sum(pathcount,2) - numofpath*ones(N,1);
pathtime2 = toc;

sumdegree = sum(nodedegree-1);

%% 训练
tic
% 计算训练集特征
obj1_T = AllGranularconcept(train_edges(:,1),1);
obj2_T = AllGranularconcept(train_edges(:,2),1);
pos1_T = AllGranularconcept(train_edges(:,1),2);
pos2_T = AllGranularconcept(train_edges(:,2),2);
conceptsimilarity_T = zeros(size(train_edges,1),1);
index_T = [];
parfor i = 1:size(train_edges,1)
    node1_T = cell2mat(obj1_T(i,:));
    node2_T = cell2mat(obj2_T(i,:));
    node1_T = node1_T(1);
    node2_T = node2_T(1);
    if max(pathcount(node1_T),pathcount(node2_T)) == 0
        PI1_T(i,:)=0;
    else
        PI1_T(i,:) = abs(pathcount(node1_T)-pathcount(node2_T))/max(pathcount(node1_T),pathcount(node2_T));
    end
    PI2_T(i,:) = ((size(pathnodefori{node1_T},1)+size(pathnodefori{node2_T},1))/size(union(pathnodefori{node1_T},pathnodefori{node2_T}),1))-1;
    competition_T(i,:) = abs(length(cell2mat(pos1_T(i,1))) -length(cell2mat(pos2_T(i,1))))/max(length(pos1_T(i,1)),length(pos2_T(i,1)));
    influence_T(i,:) = length(cell2mat(pos1_T(i,1)))*length(cell2mat(pos2_T(i,1)))/sumdegree;
    intersection_T(i,:) = numel(intersect(cell2mat(pos1_T(i,1)), cell2mat(pos2_T(i,1)))); 
end

% 归一化训练集特征并保存参数
[PI1_norm_T, min_PI1, max_PI1] = minmaxNorm_vec(PI1_T);
[PI2_norm_T, min_PI2, max_PI2] = minmaxNorm_vec(PI2_T);
[intersection_norm_T, min_intersection, max_intersection] = minmaxNorm_vec(intersection_T);
[influence_norm_T, min_influence, max_influence] = minmaxNorm_vec(influence_T);
[competition_norm_T, min_competition, max_competition] = minmaxNorm_vec(competition_T);

% 构建训练特征矩阵
X_train = [intersection_norm_T, PI2_norm_T, influence_norm_T, competition_norm_T, PI1_norm_T];
Y_train = train_edges(:,3);

%% === 仅修改此部分：使用随机森林替代贝叶斯优化 ===
fprintf('训练随机森林模型...\n');
numTrees = 100; % 树的数量
minLeafSize = 5; % 最小叶节点大小

% 训练随机森林模型
rf_model = TreeBagger(numTrees, X_train, Y_train, ...
    'Method', 'classification', ...
    'MinLeafSize', minLeafSize, ...
    'OOBPrediction', 'off', ... % 关闭袋外误差以加快速度
    'NumPredictorsToSample', 'all'); % 使用所有特征

%% 计算测试集特征
obj1 = AllGranularconcept(test_edges(:,1),1);
obj2 = AllGranularconcept(test_edges(:,2),1);
pos1 = AllGranularconcept(test_edges(:,1),2);
pos2 = AllGranularconcept(test_edges(:,2),2);

tic
conceptsimilarity = zeros(size(test_edges,1),1);
index = [];
parfor i = 1:size(test_edges,1)
    node1 = cell2mat(obj1(i,:));
    node2 = cell2mat(obj2(i,:));
    node1 = node1(1);
    node2 = node2(1);
    if max(pathcount(node1),pathcount(node2)) == 0
        PI1(i,:)=0;
    else
        PI1(i,:) = abs(pathcount(node1)-pathcount(node2))/max(pathcount(node1),pathcount(node2));
    end
    PI2(i,:) = ((size(pathnodefori{node1},1)+size(pathnodefori{node2},1))/size(union(pathnodefori{node1},pathnodefori{node2}),1))-1;
    competition(i,:) = abs(length(cell2mat(pos1(i,1))) -length(cell2mat(pos2(i,1))))/max(length(pos1(i,1)),length(pos2(i,1)));
    influence(i,:) = length(cell2mat(pos1(i,1)))*length(cell2mat(pos2(i,1)))/sumdegree;
    intersection(i,:) = numel(intersect(cell2mat(pos1(i,1)), cell2mat(pos2(i,1))));
    % 若在同一个概念里，记录索引
    if isequal(node1,node2)
            index = [index,i];
    end
end

% 使用训练集的归一化参数
PI1_norm = minmaxNorm_vec(PI1, min_PI1, max_PI1);
PI2_norm = minmaxNorm_vec(PI2, min_PI2, max_PI2);
intersection_norm = minmaxNorm_vec(intersection, min_intersection, max_intersection);
influence_norm   = minmaxNorm_vec(influence, min_influence, max_influence);
competition_norm   = minmaxNorm_vec(competition, min_competition, max_competition);

% 构建测试集特征矩阵
X_test = [intersection_norm, PI2_norm, influence_norm, competition_norm, PI1_norm];

% === 随机森林预测 ===
[~, scores] = predict(rf_model, X_test);
test_probs = scores(:, 2); % 正类概率

% 特殊处理：同一个概念中的边概率设为1
test_probs(index) = 1;

ptime = toc;
disp('测试集计算时间：')
disp(ptime)

%% 保存测试集输出（完全保持原样）


savePath = targetFolder + "\" +finame + "_testp.txt";
writematrix(test_probs, savePath);
savePath = targetFolder + "\" +finame + "_testy.txt";
writematrix(test_edges(:,3), savePath);
y_test = test_edges(:,3);

%% 性能评估,指标计算（完全保持原样）
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

fileID = fopen(fullfile(targetFolder,"results_matlab"+method+".txt"), 'a');
fprintf(fileID, '%s\n',finame + ":");
fprintf(fileID, 'precision: %.4f\n', precision);
fprintf(fileID, 'accuracy: %.4f\n', accuracy);
fprintf(fileID, 'recall: %.4f\n', recall);
fprintf(fileID, 'F1: %.4f\n', f1);
fprintf(fileID, 'MSE: %.4f\n', mse);
fprintf(fileID, 'AUC: %.4f\n', AUC);
fprintf(fileID, '\n');
end
%% 归一化函数定义（保持原样）
function [norm_x, min_val, max_val] = minmaxNorm_vec(x, min_val, max_val)
    if nargin == 1
        min_val = min(x);
        max_val = max(x);
    end
    norm_x = (x - min_val) / (max_val - min_val + eps);
    % 保留min_val和max_val用于后续归一化
end