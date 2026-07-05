% function  AllGranularconcept = GranularConceptualSpace(UAI,N)
% %% 生成UAI对应的粒等势概念空间
% % N = size(UAI,1) UAI为修正邻接矩阵
% AllGranularconcept = GenerativeGrainconcept(UAI, 1);
% for i = 2: N
%     AllGranularconcept = [AllGranularconcept;GenerativeGrainconcept(UAI, i)];
% end
% %% 去除重复的概念
% % % 确保数据不为空
% % if isempty(AllGranularconcept)
% %     error('Input data is empty!');
% % end
% % % 去重
% % numRows = N;
% % uniqueRows = true(numRows, 1);
% % for i = 1:numRows
% %     if ~uniqueRows(i)
% %         continue;
% %     end
% %     for j = i+1:numRows
% %         if isequal(AllGranularconcept(i, :), AllGranularconcept(j, :))
% %             uniqueRows(j) = false;
% %         end
% %     end
% % end
% % 
% % % 返回去重后的结果
% % AllGranularconcept = AllGranularconcept(uniqueRows, :);
% end

function [AllGranularconcept,time] = GranularConceptualSpace(UAI, N, jincheng)
% 生成UAI对应的粒概念空间
% N = size(UAI,1) UAI为修正邻接矩阵
delete(gcp('nocreate'));  % 关闭现有池
parpool('local', jincheng); % 限制为2个进程
% 并行计算
tic
parfor i = 1:N
    AllGranularconcept{i} = GenerativeGrainconcept(UAI, i);
end
time = toc;
% 将结果拼接成矩阵
AllGranularconcept = vertcat(AllGranularconcept{:});