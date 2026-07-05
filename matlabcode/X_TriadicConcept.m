% % 计算三支粒概念的对象集
%% 最初的算法，时间内存开销都很大
% function [isValid,X] = verifyTriadicConcept(UAI, objects, A1, A2)
%     % 该函数的左右类似于G算子，X相当于g(A1,A2)
%     % 获取具有A1属性的共同对象
%     UAI = sparse(UAI);
%     hasA1 = find(all(UAI(:, A1) == 1, 2))';
%     % 获取不具有A2属性的共同对象
%     notHasA2 = find(all(UAI(:, A2) == 0, 2))';
%     X = intersect(hasA1, notHasA2);
%     % 检查交集是否等于对象集X
%     isValid = isequal(sort(X), sort(objects));
% end
%% 速度快，有最初的一半的内存开销
function  X = X_TriadicConcept(UAI, A1, A2)
    hasA1 = all(UAI(:, A1) == 1, 2);  % 向量化计算 A1 条件
    candidateRows = find(hasA1);      % 仅对满足 A1 的行检查 A2
    if isempty(candidateRows)
        X = [];
        error("对象概念的对象集为空")
    else
        notHasA2 = all(UAI(candidateRows, A2) == 0, 2); % 仅检查候选行
        X = candidateRows(notHasA2)';
    end
end
%% 内存占用非常小，速度还行
% function  X = X_TriadicConcept(UAI, A1, A2)
%     m = size(UAI, 1);
%     mask = false(m, 1);
%     for i = 1:m
%         % 检查 A1 条件，若失败则跳过 A2 检查
%         if all(UAI(i, A1) == 1) && all(UAI(i, A2) == 0) % 此处 && 为标量短路逻辑与
%             mask(i) = true;
%         end
%     end
%     X = find(mask);
% end


% function [isValid, X] = verifyTriadicConcept(UAI, objects, A1, A2)
%     % 确保输入为稀疏矩阵
%     UAI = sparse(UAI);
% 
%     % 检查 A1 属性：每行在 A1 列中是否全为 1（稀疏矩阵中非零即视为1）
%     hasA1 = (sum(UAI(:, A1), 2) == length(A1));  % 利用稀疏矩阵的快速列求和
% 
%     % 检查 A2 属性：每行在 A2 列中是否全为 0
%     notHasA2 = (sum(UAI(:, A2), 2) == 0);        % 直接判断列和是否为0
% 
%     % 合并条件
%     mask = hasA1 & notHasA2;
%     X = find(mask);
% 
%     % 验证结果
%     isValid = isequal(sort(X), sort(objects));
% end