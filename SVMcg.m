function [bestacc, bestc, bestg] = SVMcg(label, data, cmin, cmax, gmin, gmax, v, cstep, gstep)

% function [bestacc, bestc, bestg] = SVMcg(label, data, cmin, cmax, gmin, gmax, v, cstep, gstep)
% calculate the best cost and gamma for libsvm
% Inputs:
%   label    - train set label
%   data     - train set data
%   cmin     - minimum cost
%   cmax     - maximum cost
%   gmin     - minimum gamma
%   gmax     - maximum gamma
%   v        - validation folds
%   cstep    - cost range step
%   gstep    - gamma range step
%   
% Outputs:
%   bestacc  - best accuracy
%   bestc    - best cost
%   bestg    - best gamma

% generate cg matrix
[c, g] = meshgrid(cmin:cstep:cmax, gmin:gstep:gmax);
[m, n] = size(c);
cg = zeros(m, n);

% calculate accuracy with different c and g, and find the best c, best g
% for best accuracy
bestc = 0;
bestg = 0;
bestacc = 0;
for i = 1:m
    for j = 1:n
        % exponential transformation to make sure cost and gamma larger
        % than zero
        cmd = ['-v ',num2str(v), ' -c ',num2str(2^c(i, j)), ' -g ',num2str(2^g(i, j))];
        % v-fold cross-validation
        cg(i, j) = svmtrain(label, data, cmd);
        
        if cg(i, j) > bestacc
            bestacc = cg(i, j);
            bestc = 2^c(i, j);
            bestg = 2^g(i, j);
        end
    end
end

