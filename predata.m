clc,clear
data = readmatrix('econ-mahindas.txt');
edges = data(:,1:2);
% 对每条边排序，确保无向边只有一对
edges = sort(edges, 2);  % 对每行进行排序

% 去重：删除重复的边
edges = unique(edges, 'rows');

num_edges = size(edges, 1);
nodes_in_edges = unique(edges(:));  % 获取所有边中的唯一节点
num_nodes = length(nodes_in_edges);  % 计算节点数量

% 创建映射，将节点编号调整为连续的 1, 2, 3, ...
node_map = containers.Map(nodes_in_edges, 1:num_nodes);

% 使用映射更新边索引
i = zeros(2 * num_edges, 1);
j = zeros(2 * num_edges, 1);
val = ones(2 * num_edges, 1);

i(1:num_edges) = cell2mat(values(node_map, num2cell(edges(:, 1))));
j(1:num_edges) = cell2mat(values(node_map, num2cell(edges(:, 2))));
i(num_edges+1:2*num_edges) = cell2mat(values(node_map, num2cell(edges(:, 2))));
j(num_edges+1:2*num_edges) = cell2mat(values(node_map, num2cell(edges(:, 1))));

% 创建稀疏矩阵
net = sparse(i, j, val, num_nodes, num_nodes);
net = spdiags(zeros(size(net,1),1), 0, net);
% 保存为 MAT 文件
save('econ-mahindas.mat', 'net');
