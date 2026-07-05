function time = Katz(finame)

load("D:\data\data\"+finame+".mat",'net');
A = full(net);
clear net
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
N = size(A,1);
tic
S = sparse(A);
[~, ~, train_edges, test_edges] = split_graph_undirected(S, 0.8);
clear S
% 构建训练图邻接矩阵
train_pos = train_edges(train_edges(:,3) == 1, :);
G = graph(train_pos(:,1), train_pos(:,2), [], size(A,1));
train_A = adjacency(G);

% 初始化结果
testp = zeros(size(test_edges, 1), 1);
testy = zeros(size(test_edges, 1), 1);

% Katz算法参数设置
beta = 0.05;  % 衰减因子，通常取小于最大特征值倒数的值
max_iter = 10; % 最大迭代次数（路径长度）

% 计算Katz相似度矩阵
% Katz(i,j) = βA + β²A² + β³A³ + ... 
katz_matrix = sparse(N, N);
A_power = train_A; % 初始化为邻接矩阵

for k = 1:max_iter
    % 添加当前路径长度的贡献
    katz_matrix = katz_matrix + beta^k * A_power;
    
    % 计算下一阶邻接矩阵
    if k < max_iter
        A_power = A_power * train_A; % A^(k+1)
    end
end

% 对于测试集中的每条边，获取其Katz分数
for i = 1:size(test_edges, 1)
    node_i = test_edges(i, 1);
    node_j = test_edges(i, 2);
    
    % 获取Katz分数
    katz_score = full(katz_matrix(node_i, node_j));
    
    testp(i) = katz_score;
    testy(i) = A(node_i, node_j); % 真实标签
end

% 归一化
min_score = min(testp);
max_score = max(testp);
if max_score > min_score
    testp = (testp - min_score) / (max_score - min_score);
else
    testp = zeros(size(testp));
end
time = toc;
% 保存结果
targetFolder = 'D:\data\output_Katz';
mkdir(targetFolder);
prob_rounded = round(testp, 4);
writematrix(prob_rounded, fullfile(targetFolder, finame + "_testp.txt"));
writematrix(testy, fullfile(targetFolder, finame + "_testy.txt"));
[~, ~, ~, AUC] = perfcurve(testy, prob_rounded, 1);
fprintf('AUC值: %.4f\n', AUC);
disp(AUC)
% 将指标写入txt
fileID = fopen(fullfile(targetFolder,"resultmatlab_Katz.txt"), 'a');
fprintf(fileID, '%s\n',finame + ":");
fprintf(fileID, 'AUC: %.4f\n', AUC);
end