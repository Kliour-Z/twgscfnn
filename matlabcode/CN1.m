%% CN实现链路预测
function time = CN1(finame)
load(finame+".mat",'net');
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
tic
S = sparse(A);
[~, ~, train_edges, test_edges] = split_graph_undirected(S, 0.8);
clear S
% 构建图
train_pos = train_edges(train_edges(:,3) == 1, :);
G = graph(train_pos(:,1), train_pos(:,2), [], size(A,1));

% 初始化测试集的概率和真实标签
testp = zeros(size(test_edges, 1), 1);
testy = zeros(size(test_edges, 1), 1);

% 计算测试集每条边的 CN 值作为概率
for i = 1:size(test_edges, 1)
    node_i = test_edges(i, 1);
    node_j = test_edges(i, 2);
    % 找到节点 i 和 j 的共同邻居
    neighbors_i = neighbors(G, node_i);
    neighbors_j = neighbors(G, node_j);
    common_neighbors = intersect(neighbors_i, neighbors_j);
    % CN 值即为共同邻居的数量
    cn_value = length(common_neighbors);
    % 将 CN 值作为链路预测概率
    testp(i) = cn_value;
    % 检查测试集中的边是否真实存在
    testy(i) = A(node_i, node_j);
end
if isequal(testy,test_edges(:,3))
    disp("没有问题")
end
% 将CN分数归一化为概率 [0, 1]
min_score = min(testp);
max_score = max(testp);
if max_score > min_score
    testp = (testp - min_score) / (max_score - min_score);
else
    testp = zeros(size(testp));  % 所有分数相同
end
time = toc;
disp("time:")
disp(time)
% 将结果保存为 txt 文件
targetFolder = 'output_CN(test)'; % 替换为你的目标路径
mkdir(targetFolder);
similarity_4 = round(testp, 4);  % 保留 4 位小数
savePath = targetFolder + "\" + finame + "_testp.txt";
writematrix(similarity_4,savePath);
savePath = targetFolder + "\" + finame + "_testy.txt";
writematrix(test_edges(:,3),savePath);
[~, ~, ~, AUC] = perfcurve(testy, similarity_4, 1);
fprintf('AUC值: %.4f\n', AUC);
disp(AUC)
fileID = fopen(fullfile(targetFolder,"resultmatlab_CN1.txt"), 'a');
fprintf(fileID, '%s\n',finame + ":");
fprintf(fileID, 'AUC: %.4f\n', AUC);
end