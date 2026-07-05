% 计算对象集合的正概念和负概念
% function [A1, A2] = computePNConcepts(UAI, objects)
%     % 该函数作用相当于F算子[A1,A2]相当于f(X)
%     UAI = sparse(UAI);
%     A1 = find(all(UAI(objects, :) == 1, 1)); % 计算正概念（共同具有的属性）
%     A2 = find(all(UAI(objects, :) == 0, 1)); % 计算负概念（共同不具有的属性）
% end
%% 用子矩阵和稀疏矩阵的方法优化
function [A1, A2] = computePNConcepts(UAI, objects)
    % 计算正概念和负概念
    % A1: 共同具有的属性
    % A2: 共同不具有的属性

    % 选择子矩阵 UAI(objects, :)，然后分别找出满足正概念和负概念的列索引
    subUAI = UAI(objects, :); % 仅选择与objects相关的行

    % 使用稀疏矩阵提高效率，如果 subUAI 主要是0的矩阵
    subUAI = sparse(subUAI); % 将子矩阵转为稀疏矩阵

    % 计算正概念（每列全为1的属性）
    A1 = find(all(subUAI == 1, 1)); % 每列全为1

    % 计算负概念（每列全为0的属性）
    A2 = find(all(subUAI == 0, 1)); % 每列全为0
end