clc,clear,close all
%% 循环运行数据集
% 读取数据集文件内容
fid = fopen('F:\LINK_PREDICTION\pythonProject\dataresult\dataname.txt', 'r');  % 打开文件
% fid = fopen('F:\LINK_PREDICTION\pythonProject\dataresult\dataname_ML.txt', 'r');  % 打开文件
if fid == -1
    error('文件不存在，请检查路径是否正确');
end
var = {};  % 初始化单元格数组
i = 1;
% 循环读取每一行
while ~feof(fid)
    line = fgetl(fid);  % 读取一行内容
    if ischar(line) && ~isempty(strtrim(line))  % 排除空行和非字符内容
        var{i} = strtrim(line);  % 去除首尾空格后存入var(i)
        i = i + 1;  % 索引自增
    end
end
fclose(fid);  % 关闭文件
% 调用函数执行
% for p=0.05:0.05:0.25
for i= 1:length(var)
    disp(var{i})
    method8(var{i})
end
% end




