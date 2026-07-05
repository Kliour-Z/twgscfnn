function TriadicConcept =  GenerativeGrainconcept(UAI, i)
%% 生成对象i对应的三支粒概念
[POS, NEG] = computePNConcepts(UAI, i);
X = X_TriadicConcept(UAI, POS, NEG);
TriadicConcept = {X, POS, NEG};
end
%% 生成限制负概念的大小
% function TriadicConcept =  GenerativeGrainconcept(UAI, i)
%% 生成对象i对应的三支粒等势概念
% [POS, NEG] = computePNConcepts(UAI, i);
% [~,X]= verifyTriadicConcept(UAI, i, POS, NEG);
% maxNEGSize  = 25;% 生成限制负概念的大小
%     if numel(NEG) > maxNEGSize       
%         % 如果需要随机选择maxNEGSize个元素（不需要按顺序）
%         NEG = NEG(randperm(numel(NEG), maxNEGSize)); % 随机选择maxNEGSize个元素
%     end
% TriadicConcept = {X, POS, NEG};
% end