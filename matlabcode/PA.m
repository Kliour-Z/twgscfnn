% PA_LinkPrediction.m
function time = PA(finame)


load(finame + ".mat",'net');
A = full(net);
clear net
if isequal(A, A')
    disp('矩阵是对称矩阵');
else
    disp('矩阵不是对称矩阵');
    return
end
if max(max(A)) ~= 1
    disp('矩阵有问题');
    return
else
    disp('继续');
end
tic
S = sparse(A);
[~, ~, train_edges, test_edges] = split_graph_undirected(S, 0.8);
clear S
% 构建训练图
train_pos = train_edges(train_edges(:,3) == 1, :);
G = graph(train_pos(:,1), train_pos(:,2), [], N);

% 初始化结果
testp = zeros(size(test_edges, 1), 1);
testy = zeros(size(test_edges, 1), 1);

% 计算测试集每条边的PA值
for i = 1:size(test_edges, 1)
    node_i = test_edges(i, 1);
    node_j = test_edges(i, 2);
    
    % 计算节点度
    deg_i = degree(G, node_i);
    deg_j = degree(G, node_j);
    
    % PA值：deg_i * deg_j
    pa_value = deg_i * deg_j;
    
    testp(i) = pa_value;
    testy(i) = A(node_i, node_j); % 真实标签
end
if isequal(testy,test_edges(:,3))
    disp("没有问题")
end
% 归一化
min_score = min(testp);
max_score = max(testp);
if max_score > min_score
    testp = (testp - min_score) / (max_score - min_score);
else
    testp = zeros(size(testp));
end
time = toc;
% 保存结果
targetFolder = 'output_PA';
mkdir(targetFolder);
prob_rounded = round(testp, 4);
writematrix(prob_rounded, fullfile(targetFolder, finame + "_testp.txt"));
writematrix(testy, fullfile(targetFolder, finame + "_testy.txt"));
end