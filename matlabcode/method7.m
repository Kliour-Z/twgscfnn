function method7(finame)
%% method7 线性模型+sigmoid
% 代码参数设置
method = "TWGSCLINE(test)";
L = 0.8; % 训练集占比
nn = 1; % 并行池数量
numofpath = 30;
lenthofpath = 10;
targetFolder = "D:\all\result\output_" + method;
mkdir(targetFolder);
rng(1213); 
%% 数据准备
% 示例数据
% UAI1 = [1 1 0 0 1 0 0 0;1 1 1 0 0 1 0 0;0 1 1 1 0 0 0 0;0 0 1 1 0 0 0 0;1 0 0 0 1 1 0 0;0 1 0 0 1 1 1 0; 0 0 0 0 0 1 1 1;0 0 0 0 0 0 1 1];
% UAI = [1 0 0 0 0 0 0;0 1 1 0 1 1 0;0 1 1 0 1 0 0;0 0 0 1 0 0 1;0 1 1 0 1 1 0;0 1 0 0 1 1 0;0 0 0 1 0 0 1];% 获取训练集的修正邻接矩阵UAI
% A = UAI - eye(size(UAI,1));
load(finame+".mat",'net');
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
        temp2 = union(temp2,jpath)
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
% for i = 1:size(test_edges,1)
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
PI1_norm_T = minmaxNorm_vec(PI1_T,min(PI1_T),max(PI1_T));
PI2_norm_T = minmaxNorm_vec(PI2_T,min(PI2_T),max(PI2_T));
intersection_norm_T = minmaxNorm_vec(intersection_T,min(intersection_T),max(intersection_T));
influence_norm_T = minmaxNorm_vec(influence_T,min(influence_T),max(influence_T));
competition_norm_T = minmaxNorm_vec(competition_T,min(competition_T),max(competition_T));
 features = struct(...
        'intersection', intersection_norm_T, ...
        'PI2', PI2_norm_T, ...
        'influence', influence_norm_T, ...
        'competition', competition_norm_T, ...
        'PI1', PI1_norm_T);

% 贝叶斯优化寻找最优参数
fprintf('开始贝叶斯优化...\n');
vars = [
    optimizableVariable('a', [0.1,0.8]), ...
    optimizableVariable('b', [0.1,0.8]), ...
    optimizableVariable('c', [0.1,0.8]), ...
    optimizableVariable('d', [0.1,0.8]), ...
    optimizableVariable('e', [0.1,0.8])
];
objectiveFcn = @(params) constrained_loss(params, features, train_edges(:,3));
% 运行贝叶斯优化
optimizer = bayesopt(objectiveFcn, vars, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 5, ...
    'UseParallel', true, ...
    'PlotFcn', []);

%% 获取最优参数
best_params = bestPoint(optimizer);
param_vec = [best_params.a, best_params.b, best_params.c, best_params.d,best_params.e];
param_vec_normalized = param_vec / sum(param_vec); % 归一化参数
% param_vec_normalized = param_vec;
a = param_vec_normalized(1);
b = param_vec_normalized(2);
c = param_vec_normalized(3);
d = param_vec_normalized(4);
e = param_vec_normalized(5);
fprintf('优化完成! 最优参数: a=%.4f, b=%.4f, c=%.4f, d=%.4f,e=%.4f', a, b, c, d, e);

%% 定义交叉熵函数
function loss = constrained_loss(params, features, labels)
    % 提取参数
    param_vec = [params.a, params.b, params.c, params.d,params.e];
    
    % 归一化参数以满足 a+b+c+d=1
    param_vec = param_vec / sum(param_vec);
    a = param_vec(1); b = param_vec(2); c = param_vec(3); d = param_vec(4); e = param_vec(5); 
    
    % 计算加权相似度
    p = a * features.intersection + ...
        b * features.PI2 + ...
        c * features.influence + ...
        d * features.competition+ ...
        e * features.PI1;
    
    % Sigmoid转换获得概率
    p = 1 ./ (1 + exp(-p));

    % 防止log(0)
    p = max(min(p, 1-1e-10), 1e-10);
    
    % 计算交叉熵
    loss = -mean(labels .* log(p) + (1-labels) .* log(1-p));
    
    % 添加惩罚项确保参数为正
    penalty = 0;
    if any(param_vec < 0)
        penalty = 1e6 * sum(abs(param_vec(param_vec < 0))); % 大惩罚
    end
    
    % 添加L2正则化
    lambda = 0.15;
    reg = lambda * sum(param_vec.^2);
    
    loss = loss + penalty + reg;
end

%% 计算测试集相似度

% 按照test_edges节点的顺序提取AllGranularconcept需要用的对象集和正域
obj1 = AllGranularconcept(test_edges(:,1),1);
obj2 = AllGranularconcept(test_edges(:,2),1);
pos1 = AllGranularconcept(test_edges(:,1),2);
pos2 = AllGranularconcept(test_edges(:,2),2);


tic
conceptsimilarity = zeros(size(test_edges,1),1);
index = [];
parfor i = 1:size(test_edges,1)
% for i = 1:size(test_edges,1)
    node1 = cell2mat(obj1(i,:));
    node2 = cell2mat(obj2(i,:));
    node1 = node1(1) ;
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
    % 若在同一个概念里，相似度相同
    if isequal(node1,node2)
            index = [index,i];
    end
end
PI1_norm = minmaxNorm_vec(PI1,min(PI1),max(PI1));
PI2_norm = minmaxNorm_vec(PI2,min(PI2),max(PI2));
intersection_norm = minmaxNorm_vec(intersection,min(intersection),max(intersection));
influence_norm   =  minmaxNorm_vec(influence,min(influence),max(influence));
competition_norm   =  minmaxNorm_vec(competition,min(competition),max(competition));

%计算最终概率
conceptsimilarity = a*intersection_norm+b*PI2_norm +c*influence_norm+d*competition_norm+e*PI1_norm;
conceptsimilarity(index') = 1;

 % 计算最终概率
 % similarity =conceptsimilarity;
similarity = 1 ./ (1 + exp(-conceptsimilarity));

ptime = toc;
disp('测试集计算时间：')
disp(ptime)
%% 保存测试集输出
 % 保留 4 位小数
savePath = targetFolder + "\" +finame + "_testp.txt";
writematrix(similarity,savePath);
savePath = targetFolder + "\" +finame + "_testy.txt";
writematrix(test_edges(:,3),savePath);
y_test=test_edges(:,3);
test_probs = similarity;
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