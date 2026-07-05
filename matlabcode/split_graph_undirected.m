function [train_matrix, test_matrix, train_edges, test_edges] = split_graph_undirected(A, a)
    %% 划分无向图的训练集和测试集（彻底解决负样本重叠问题）
    if ~isequal(A, A')
        error('输入矩阵必须是无向图的对称矩阵');
    end
    [N, M] = size(A);
    if N ~= M
        error('邻接矩阵必须是方阵');
    end
    
    % 提取正样本边（i<=j）
    [rows, cols] = find(triu(A, 1));
    pos_pairs = [rows, cols];
    num_pos = length(rows);
    clear A
    % 划分正样本
    rng('default');
    indices = randperm(num_pos);
    split = round(a * num_pos);
    train_pos_idx = indices(1:split);
    test_pos_idx = indices(split+1:end);
    
    % 训练/测试正样本
    train_pos_i = rows(train_pos_idx);
    train_pos_j = cols(train_pos_idx);
    test_pos_i = rows(test_pos_idx);
    test_pos_j = cols(test_pos_idx);
    
    % 直接为训练集和测试集分别生成负样本
    % 训练负样本（排除所有正样本）
    [train_neg_i, train_neg_j] = generate_undirected_negative_samples(pos_pairs, 2*length(train_pos_i), N);
    
    % 测试负样本（排除所有正样本+训练负样本）
    test_exclude_pairs = [pos_pairs; train_neg_i, train_neg_j];
    [test_neg_i, test_neg_j] = generate_undirected_negative_samples(test_exclude_pairs, 2*length(test_pos_i), N);
    
    % 构建带标签的边列表
    train_edges = [train_pos_i, train_pos_j, ones(length(train_pos_i), 1);
                   train_neg_i, train_neg_j, zeros(length(train_neg_i), 1)];
    test_edges = [test_pos_i, test_pos_j, ones(length(test_pos_i), 1);
                  test_neg_i, test_neg_j, zeros(length(test_neg_i), 1)];
    
    % 打乱顺序
    train_edges = train_edges(randperm(size(train_edges, 1)), :);
    test_edges = test_edges(randperm(size(test_edges, 1)), :);
    
    % 创建稀疏矩阵
    train_matrix = sparse([train_pos_i; train_pos_j], [train_pos_j; train_pos_i], 1, N, N);
    test_matrix = sparse([test_pos_i; test_pos_j], [test_pos_j; test_pos_i], 1, N, N);
end

function [neg_i, neg_j] = generate_undirected_negative_samples(exclude_pairs, num_neg_total, N)
    % 创建哈希表存储排除的边（格式: "i,j" 字符串, i < j）
    exclude_dict = containers.Map();
    for k = 1:size(exclude_pairs, 1)
        i = min(exclude_pairs(k, :));
        j = max(exclude_pairs(k, :));
        key = sprintf('%d,%d', i, j);
        exclude_dict(key) = true;
    end
    
    neg_pairs = zeros(num_neg_total, 2);
    count = 0;
    max_attempts = num_neg_total * 100; % 避免无限循环
    attempt = 0;
    
    while count < num_neg_total && attempt < max_attempts
        attempt = attempt + 1;
        i = randi(N);
        j = randi(N);
        if i == j, continue; end % 跳过自环
        % 统一为 i < j
        if i > j, [i, j] = deal(j, i); end
        key = sprintf('%d,%d', i, j);
        
        % 检查是否在排除集
        if ~isKey(exclude_dict, key)
            count = count + 1;
            neg_pairs(count, :) = [i, j];
            exclude_dict(key) = true; % 避免重复采样
        end
    end
    
    if count < num_neg_total
        warning('仅生成 %d/%d 个负样本', count, num_neg_total);
        neg_pairs = neg_pairs(1:count, :);
    end
    
    neg_i = neg_pairs(:, 1);
    neg_j = neg_pairs(:, 2);
end