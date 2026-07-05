function v_norm = minmaxNorm_vec(v,v_min,v_max)
% 将列向量归一化到[0,1]区间
% 输入: v - 列向量 (n×1)
% 输出: v_norm - 归一化后的列向量

    
    % 处理常量向量 (所有元素相等)
    if v_max == v_min
        % 常量向量归一化为0.5，或保持原值(0)
        if v_min == 0
            v_norm = zeros(size(v));
        else
            v_norm = 0.5 * ones(size(v));
        end
    else
        v_norm = (v - v_min) / (v_max - v_min);
    end
end
