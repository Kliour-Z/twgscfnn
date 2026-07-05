clc, clear
% 读取文件中的边数据
fileID = fopen('citeseer_cites.txt', 'r');  % 打开文件
edges = {};  % 用于存储边数据
tline = fgetl(fileID);  % 读取文件的第一行
while ischar(tline)
    edge_data = strsplit(tline);  % 按空格分割
    % 如果第一列是空格（无效），则只保留后面的列
    if isempty(strtrim(edge_data{1}))  % 检查第一列是否为空
        edge_data = edge_data(2:end);  % 跳过第一列，保留后面的列
    end
    edges = [edges; edge_data(:,1:2)];  % 将每行分割并添加到edges数组
    tline = fgetl(fileID);  % 读取下一行
end
fclose(fileID);  % 关闭文件

% 清除空值或 NaN 节点（如果有）
edges = edges(~cellfun(@(x) any(isnan(x)), edges(:, 1)), :);  % 去掉NaN行

% 创建节点编号映射
% 将所有节点合并为一个唯一节点列表
unique_nodes = unique(edges(:));  
node_map = containers.Map(unique_nodes, 1:length(unique_nodes));  % 创建映射

% 4. 将边数据重新映射到连续编号
mapped_edges = zeros(size(edges));
for i = 1:size(edges, 1)
    mapped_edges(i, 1) = node_map(edges{i, 1});  % 映射第一个节点
    mapped_edges(i, 2) = node_map(edges{i, 2});  % 映射第二个节点
end
% 5. 去除重复边
% 使用 sorted 形式，确保无向图边的顺序一致
mapped_edges_sorted = sort(mapped_edges, 2);  % 按每条边的节点大小排序
mapped_edges_sorted = unique(mapped_edges_sorted, 'rows');  % 去除重复的边
% 6. 创建无向图的稀疏矩阵
n_nodes = length(unique_nodes);  % 节点总数
i = [mapped_edges_sorted(:, 1); mapped_edges_sorted(:, 2)];
j = [mapped_edges_sorted(:, 2); mapped_edges_sorted(:, 1)];
val = ones(size(i));  % 权重为1

% 创建稀疏矩阵，确保对称性
net = sparse(i, j, val, n_nodes, n_nodes);
net = spdiags(zeros(size(net,1),1), 0, net);
% 保存为 MAT 文件
save('citeseer_cites.mat', 'net');