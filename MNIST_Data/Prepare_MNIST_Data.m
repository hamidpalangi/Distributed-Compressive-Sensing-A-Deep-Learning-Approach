% This Mfile prepares the MNIST data set to feed into LSTM-CS algorithm.
% It creates 10 channels each one including samples of one of the numbers
% from the set {0,1,2,..,9}.
% Contact: Hamid Palangi, hamidp@ece.ubc.ca

clear all;close all;clc;
% You need to download following files from:
% "http://yann.lecun.com/exdb/mnist/" and edit following paths accordingly.
DataPathTr = '..\train-images.idx3-ubyte';
LabelPathTr = '..\train-labels.idx1-ubyte';
DataPathTest = '..\t10k-images.idx3-ubyte';
LabelPathTest = '..\t10k-labels.idx1-ubyte';
Ims1 = loadMNISTImages(DataPathTr);
Labels = loadMNISTLabels(LabelPathTr);
% Reference for above two functions:
% "http://ufldl.stanford.edu/wiki/index.php/Main_Page".
ImsTr = cell(10,1);
for i=0:9
    idx = find( Labels == i);
    Ims = zeros(28,28,length(idx));
    temp = Ims1(:,idx);
    for j=1:length(idx)
        Ims(:,:,j) = col2im( temp(:,j), [28 28], [28 28], 'distinct' );
    end
    ImsTr{i+1,1} = Ims;
end
clear temp;
clear Ims;
clear idx;
ImsVal = cell(10,1);
for i=1:10
    temp = ImsTr{i,1};
    ImsVal{i,1} = temp(:,:,1:100);
    temp(:,:,1:100) = [];
    ImsTr{i,1} = temp;
end
clear temp;
Ims1 = loadMNISTImages(DataPathTest);
Labels = loadMNISTLabels(LabelPathTest);
ImsTest = cell(10,1);
for i=0:9
    idx = find( Labels == i);
    Ims = zeros(28,28,length(idx));
    temp = Ims1(:,idx);
    for j=1:length(idx)
        Ims(:,:,j) = col2im( temp(:,j), [28 28], [28 28], 'distinct' );
    end
    ImsTest{i+1,1} = Ims;
end
% Add the path that you want to save the prepared files:
save('..\ImsTr.mat','ImsTr');
save('..\ImsVal.mat','ImsVal');
save('..\ImsTest.mat','ImsTest');

