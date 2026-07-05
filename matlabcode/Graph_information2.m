clc, clear, close all
finame = 'PP-Pathways_ppi';
load(finame + ".mat", 'net');
A = net;
clear net

% 优化分析函数
[N, E, AVD, AVC, AVP] = OptimizedGraphAnalysis(A);

targetFolder = 'F:\作业\Graph_information';
mkdir(targetFolder);
fileID = fopen(fullfile(targetFolder, 'Graph_information.txt'), 'a');
fprintf(fileID, '%s\n', finame + ":");
fprintf(fileID, 'N: %.4f ', N);
fprintf(fileID, 'E: %.4f ', E);
fprintf(fileID, 'AVD: %.4f ', AVD);
fprintf(fileID, 'AVC: %.4f ', AVC);
fprintf(fileID, 'AVP: %.4f \n', AVP);

function [n, m, avg_degree, avg_clustering, avg_path] = OptimizedGraphAnalysis(A)
    % 转换为稀疏矩阵以节省内存
    A = sparse(A);
    
    % 确保矩阵对称且无自环
    if ~issymmetric(A)
        A = spones(A + A');  % 强制对称化
    end
    A = A - diag(diag(A));  % 移除自环
    
    % 1. 节点数
    n = size(A, 1);
    
    % 2. 边数 (无向图每条边计数两次)
    m = full(sum(A(:))) / 2;
    
    % 3. 平均度数
    avg_degree = 2 * m / n;
    
    % 4. 聚类系数计算
    avg_clustering = ClusteringCoefficient(A, n);
    
    % 5. 平均最短路径计算
    if n == 0
        avg_path = NaN;
    elseif n == 1
        avg_path = 0;
    else
        avg_path = StableAveragePath(A, n);
    end
end

function coeff = ClusteringCoefficient(A, n)
    % 优化聚类系数计算
    if n > 60000
        sample_size = min(60000, n);
        sample_nodes = randperm(n, sample_size);
        clustering_coeffs = zeros(sample_size, 1);
        
        for idx = 1:sample_size
            i = sample_nodes(idx);
            neighbors = find(A(i, :));
            k = numel(neighbors);
            if k < 2
                clustering_coeffs(idx) = 0;
            else
                % 向量化计算三角形数量
                subgraph = A(neighbors, neighbors);
                num_edges = full(sum(subgraph(:))) / 2;
                max_possible = k * (k - 1) / 2;
                clustering_coeffs(idx) = num_edges / max_possible;
            end
        end
        coeff = mean(clustering_coeffs);
    else
        % 全图计算
        clustering_coeffs = zeros(n, 1);
        for i = 1:n
            neighbors = find(A(i, :));
            k = numel(neighbors);
            if k < 2
                clustering_coeffs(i) = 0;
            else
                subgraph = A(neighbors, neighbors);
                num_edges = full(sum(subgraph(:))) / 2;
                max_possible = k * (k - 1) / 2;
                clustering_coeffs(i) = num_edges / max_possible;
            end
        end
        coeff = mean(clustering_coeffs);
    end
end

function avg_path = StableAveragePath(A, n)
    % 查找连通分量
    [comp_labels, comp_sizes] = FindConnectedComponents(A);
    num_components = numel(comp_sizes);
    
    % 按分量大小排序
    [sorted_sizes, sort_idx] = sort(comp_sizes, 'descend');
    sorted_labels = sort_idx;
    
    % 存储各分量的平均路径和权重
    comp_paths = zeros(1, num_components);
    comp_weights = zeros(1, num_components);
    
    % 总节点对数
    total_pairs = 0;
    
    for comp_id = 1:num_components
        comp_nodes = find(comp_labels == sorted_labels(comp_id));
        comp_size = sorted_sizes(comp_id);
        
        if comp_size == 1
            comp_paths(comp_id) = 0;
            comp_weights(comp_id) = 0;
            continue;
        end
        
        % 计算分量内节点对数 (无向图)
        comp_pairs = comp_size * (comp_size - 1) / 2;
        comp_weights(comp_id) = comp_pairs;
        total_pairs = total_pairs + comp_pairs;
        
        % 根据分量大小选择计算方法
        if comp_size <= 5000
            % 小分量：完整计算
            comp_paths(comp_id) = FullComponentPath(A, comp_nodes);
        else
            % 大分量：使用稳定的采样方法
            comp_paths(comp_id) = SampledComponentPath(A, comp_nodes, comp_size);
        end
    end
    
    % 计算整体加权平均路径
    if total_pairs > 0
        weighted_sum = sum(comp_paths .* comp_weights);
        avg_path = weighted_sum / total_pairs;
    else
        avg_path = 0;
    end
end

function comp_avg = FullComponentPath(A, nodes)
    % 完整计算连通分量内的平均路径
    comp_size = numel(nodes);
    total_distance = 0;
    
    % 为分量创建子图
    subgraph = A(nodes, nodes);
    
    % 使用Floyd-Warshall算法计算所有节点对的最短路径
    D = FloydWarshall(full(subgraph));
    
    % 计算所有节点对的距离和
    for i = 1:(comp_size-1)
        for j = (i+1):comp_size
            if D(i, j) < inf
                total_distance = total_distance + D(i, j);
            end
        end
    end
    
    % 计算分量内平均路径
    comp_pairs = comp_size * (comp_size - 1) / 2;
    comp_avg = total_distance / comp_pairs;
end

function comp_avg = SampledComponentPath(A, nodes, comp_size)
    % 稳定的采样方法计算大分量内的平均路径
    total_distance = 0;
    total_pairs = 0;
    
    % 采样设置 - 确保足够的采样密度
    sample_size = min(ceil(sqrt(comp_size)) * 10, comp_size);
    sample_size = max(sample_size, 100);  % 至少采样100个节点
    sample_nodes = nodes(randperm(comp_size, sample_size));
    
    % 计算采样节点之间的所有路径
    for i = 1:sample_size
        source = sample_nodes(i);
        distances = OptimizedBFS(A, source);
        
        % 计算到其他采样节点的距离
        for j = 1:sample_size
            if i ~= j
                target = sample_nodes(j);
                d = distances(target);
                if d < inf
                    total_distance = total_distance + d;
                    total_pairs = total_pairs + 1;
                end
            end
        end
    end
    
    % 计算采样平均路径
    comp_avg = total_distance / total_pairs;
end

function [labels, sizes] = FindConnectedComponents(A)
    % 使用优化的BFS查找连通分量
    n = size(A, 1);
    labels = zeros(n, 1, 'uint32');
    visited = false(n, 1);
    comp_count = 0;
    
    for i = 1:n
        if ~visited(i)
            comp_count = comp_count + 1;
            % 执行BFS
            queue = zeros(1, n);
            front = 1;
            rear = 1;
            queue(rear) = i;
            rear = rear + 1;
            visited(i) = true;
            labels(i) = comp_count;
            comp_size = 1;
            
            while front < rear
                current = queue(front);
                front = front + 1;
                
                % 获取邻居（列访问更高效）
                neighbors = find(A(:, current));
                for j = 1:numel(neighbors)
                    neighbor = neighbors(j);
                    if ~visited(neighbor)
                        visited(neighbor) = true;
                        labels(neighbor) = comp_count;
                        queue(rear) = neighbor;
                        rear = rear + 1;
                        comp_size = comp_size + 1;
                    end
                end
            end
            sizes(comp_count) = comp_size;
        end
    end
end

function D = FloydWarshall(A)
    % Floyd-Warshall算法计算所有节点对的最短路径
    n = size(A, 1);
    D = A;
    
    % 初始化距离矩阵
    D(D == 0) = inf;
    D(1:n+1:end) = 0;  % 对角线设为0
    
    for k = 1:n
        for i = 1:n
            for j = 1:n
                if D(i, k) + D(k, j) < D(i, j)
                    D(i, j) = D(i, k) + D(k, j);
                end
            end
        end
    end
end

function distances = OptimizedBFS(A, source)
    % 高效的BFS实现
    n = size(A, 1);
    distances = inf(n, 1, 'single');
    visited = false(n, 1);
    
    % 预分配队列
    queue = zeros(n, 1, 'uint32');
    front = 1;
    rear = 1;
    
    queue(rear) = source;
    rear = rear + 1;
    visited(source) = true;
    distances(source) = 0;
    
    while front < rear
        current = queue(front);
        front = front + 1;
        current_dist = distances(current);
        
        % 获取邻居（列访问更高效）
        neighbors = find(A(:, current));
        for j = 1:length(neighbors)
            neighbor = neighbors(j);
            if ~visited(neighbor)
                visited(neighbor) = true;
                distances(neighbor) = current_dist + 1;
                queue(rear) = neighbor;
                rear = rear + 1;
            end
        end
    end
end