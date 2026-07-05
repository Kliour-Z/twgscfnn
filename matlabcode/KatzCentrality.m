function katz_centrality = KatzCentrality(A, beta, label, m)
    % 输入:
    % A - 邻接矩阵
    % beta - 控制中心性的参数
    % label - 计算方式 0 迭代 1 求逆
    % m - 迭代次数
    % 输出:
    % katz_centrality - Katz中心性向量
    
    % 计算特征值和特征向量
    [~, eigenvalues] = eig(A);
    % 提取对角线上的特征值
    eigenvalues = diag(eigenvalues);
    % 找到最大特征值
    max_eigenvalue = max(eigenvalues);
    % 随机取alpha - 衰减因子，通常小于1/(最大特征值)
    alpha = 0.9 *(1/max_eigenvalue);
    %% 迭代法计算
    if label == 0
        n = size(A, 1);
        K = ones(n, 1) * beta;
        for iter = 1:m % 最大迭代次数
            K_new = alpha * A * K + ones(n, 1) * beta;
            if norm(K_new - K, 2) < 1e-8 % 收敛阈值
                break;
            end
            K = K_new;
        end
        katz_centrality = K;
    end

    %% 求逆法计算
    if label == 1
        n = size(A, 1); % 获取节点数量
        I = eye(n); % 创建单位矩阵
        katz_centrality = (I - alpha * A') \ (beta * ones(n, 1));
    end
end