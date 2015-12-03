clc;clear;
% load iris data
iris_data = load('iris.txt');
% find the best cost and best gamma for best accuracy
% cost varies from -4 to 4 with step 0.1
% gamma varies from -4 to 4 with step 0.1
% 3-fold cross-validation
[bestacc, bestc, bestg] = SVMcg(iris_data(:, 5), iris_data(:, 1:4), -4, 4, -4, 4, 3, 0.1, 0.1);
% verify the best parameters
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -v 3'];
accuracy = svmtrain(iris_data(:, 5), iris_data(:, 1:4), cmd);