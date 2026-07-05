% Jaccard
function time = Jaccard(finame)

load(finame + ".mat",'net');
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
% 构建训练图
train_pos = train_edges(train_edges(:,3) == 1, :);
G = graph(train_pos(:,1), train_pos(:,2), [], size(A,1));

% 初始化结果
testp = zeros(size(test_edges, 1), 1);
testy = zeros(size(test_edges, 1), 1);

% 计算测试集每条边的Jaccard值
for i = 1:size(test_edges, 1)
    node_i = test_edges(i, 1);
    node_j = test_edges(i, 2);
    
    % 获取节点邻居
    neighbors_i = neighbors(G, node_i);
    neighbors_j = neighbors(G, node_j);
    
    % 计算交集和并集
    intersection = numel(intersect(neighbors_i, neighbors_j));
    union_size = numel(neighbors_i) + numel(neighbors_j) - intersection;
    
    % Jaccard值：|N(i) ∩ N(j)| / |N(i) ∪ N(j)|
    if union_size > 0
        jaccard_value = intersection / union_size;
    else
        jaccard_value = 0;
    end
    
    testp(i) = jaccard_value;
    testy(i) = A(node_i, node_j); % 真实标签
end
if isequal(testy,test_edges(:,3))
    disp("没有问题")
end
% Jaccard值已在[0,1]区间，但为了与其他算法一致，仍进行归一化
min_score = min(testp);
max_score = max(testp);
if max_score > min_score
    testp = (testp - min_score) / (max_score - min_score);
else
    testp = zeros(size(testp));
end
time = toc;
% 保存结果
targetFolder = 'output_Jaccard';
mkdir(targetFolder);
prob_rounded = round(testp, 4);
writematrix(prob_rounded, fullfile(targetFolder, finame + "_testp.txt"));
writematrix(testy, fullfile(targetFolder, finame + "_testy.txt"));
end