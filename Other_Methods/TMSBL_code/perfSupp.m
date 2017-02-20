function [F, P, R, ind] = perfSupp(estSource, trueSourceIndex, varargin)
% F-Measure of support recovery
%
% *** INPUT Parameters ***
%   estSource               : estimated sources
%   trueSourceIndex         : true support location index
%   varargin{1} == 'firstLargest' & varargin{2} == D 
%       : select the support of D largest sources (in terms of 2-norm)
%   varargin{1} == 'largerThan' & varargin{2} == D
%       : select the support of elements larger than D
%
% *** OUTPUT Parameters ***
%   F  : F-Measure of support recovery (=1: perfect): 
%        F = (2*P*R)/(R+P)
%   P  : Precision of support recovery: 
%        P = (common set of estimated support and true support)/(set of estimated support);
%   R  : Recall of support recovery: 
%        R = (common set of estimated support and true support)/(set of true support);
% ind  : the support index used to measure performance
%
% Use:
%     F = perfSupp(estSource, trueSourceIndex, 'firstLargest', 10);
%     [F,P,R] = perfSupp(estSource, trueSourceIndex, 'largerThan', 1e-5);
% 
%   Author: Zhilin Zhang
%   Date  : July, 2010
% Version : 1.2



if length(varargin) == 2
    switch lower(varargin{1})
        case 'firstlargest'
            numThreshold = varargin{2};
            estSource = sum(estSource.^2,2);
            [sortedSource,sortInd] = sort(estSource, 1, 'descend');
            ind = sortInd(1:numThreshold);

        case 'largerthan'
            valThreshold = varargin{2};
            estSource = sum(estSource.^2,2);
            ind = find(estSource >= valThreshold);

    end
else
    error('Optional parameters are wrong!\n');
end

commonSupp = intersect(ind,trueSourceIndex);

if isempty(commonSupp), 
    F = 0; P = 0; R = 0;
else
    P = length(commonSupp)/length(ind);
    R = length(commonSupp)/length(trueSourceIndex);
    F = 2 * P * R/(P + R);
end



