function path = pathgeneration(node, lenth, pos)
% node为节点
% lenth 为路径长度
% pos 为邻居集合节点的
path = []; 
path = [path,node];
for i = 1:lenth-1
    postemp = cell2mat(pos(path(1,end),:));
    candidate = remove_vectors(postemp, path);
    if isempty(candidate)
        path = [path,node];
    else
        path = [path, candidate(randi(length(candidate)))];
    end
end

end