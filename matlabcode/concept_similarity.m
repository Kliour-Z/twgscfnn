function similarity = concept_similarity(vector1, vector2)
    % 计算概念相似度
    vector1(vector1 == 0) = []; 
    vector2(vector2 == 0) = [];
    intersection = numel(intersect(vector1, vector2)); % 交集大小
    unionSize = numel(union(vector1, vector2)); % 并集大小
    if intersection == [] 
        similarity = 0;
    elseif intersection == 0
        similarity = 0;
    else
        similarity = intersection / unionSize; % 概念相似度
    end
end