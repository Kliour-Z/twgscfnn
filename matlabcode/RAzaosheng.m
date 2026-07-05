% RA_LinkPrediction.m
function time = RAzaosheng(finame,p)

load(finame + ".mat",'net');
A = full(net);
clear net
tic
S = sparse(A);
[~, ~, train_edges, test_edges] = split_graph_undirected(S, 0.8);
%% 随机删除训练集边增加噪声
% 找到所有边（label=1）
edge_idx = find(train_edges(:,3) == 1);
num_edges = length(edge_idx);
num_remove = round(num_edges * p);
idx_to_flip = edge_idx(randperm(num_edges, num_remove));
% 把这些边的标签从 1 改成 0
train_edges(idx_to_flip, 3) = 0;
clear S

% 构建训练图
train_pos = train_edges(train_edges(:,3) == 1, :);
G = graph(train_pos(:,1), train_pos(:,2), [], size(A,1));

% 初始化结果
testp = zeros(size(test_edges, 1), 1);
testy = zeros(size(test_edges, 1), 1);

% 计算测试集每条边的RA值
for i = 1:size(test_edges, 1)
    node_i = test_edges(i, 1);
    node_j = test_edges(i, 2);
    
    % 获取节点邻居
    neighbors_i = neighbors(G, node_i);
    neighbors_j = neighbors(G, node_j);
    
    % 找到共同邻居
    common_neighbors = intersect(neighbors_i, neighbors_j);
    
    % 计算RA值：∑(1/degree(z)) for z in common_neighbors
    ra_value = 0;
    for z = common_neighbors'
        deg_z = degree(G, z);
        ra_value = ra_value + 1/deg_z;
    end

    if isempty(ra_value) 
       ra_value = 0;
    end
    testp(i) = ra_value;
    testy(i) = A(node_i, node_j); % 真实标签

end
if isequal(testy,test_edges(:,3))
    disp("没有问题")
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
targetFolder = "D:\all\result\output_RAzaosheng"+p;
mkdir(targetFolder);
prob_rounded = round(testp, 4);
writematrix(prob_rounded, fullfile(targetFolder, finame + "_testp.txt"));
writematrix(testy, fullfile(targetFolder, finame + "_testy.txt"));
[~, ~, ~, AUC] = perfcurve(testy, prob_rounded, 1);
fprintf('AUC值: %.4f\n', AUC);
disp(AUC)
fileID = fopen(fullfile(targetFolder,"resultmatlab_RA.txt"), 'a');
fprintf(fileID, '%s\n',finame + ":");
fprintf(fileID, 'AUC: %.4f\n', AUC);
end