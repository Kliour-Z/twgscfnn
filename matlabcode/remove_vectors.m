function result = remove_vectors(original, to_remove)
% REMOVE_VECTOR_ELEMENTS 从原始向量中移除指定元素
%   original: 原始向量
%   to_remove: 需要移除的元素组成的向量
%   result: 移除后的结果向量

% 使用ismember函数找出原始向量中存在于to_remove中的元素
idx = ismember(original, to_remove);

% 保留不在to_remove中的元素
temp = original(~idx);

%%  如果移除后为空集，则只移除最新的节点
if isempty(temp)
    idx = ismember(original, to_remove(:,end));
    result = original(~idx);
else
    result = original(~idx);
end

end