%% ��]���ފ�w�K�E���؃v���O����
%% �T�v
% * �摜��{ 'DIHS','DRESS','EM_Major','EM_minor','MPE','MP_DR','SJS','SJS_TEN','TEN'}�ɕ���
% * �T�u�T�u�t�H���_���Ńe�X�g�f�[�^�ƂɌP���p�E���ؗp�f�[�^�𕪊�
% * ���O�w�K�ς݃l�b�g���[�N(GoogLeNet�Ȃǁj�̓]�ڊw�K�ŕ��ފ��݌v
% * �O�����ł͒����������`�ŃN���b�s���O�C���͑w�ɍ��킹�ă��T�C�Y
% 
% �i�]�ڊw�K�j <https://jp.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html?lang=en 
% https://jp.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html?lang=en>
% 
% 2019/1/25  �V����w�@���R�Ȋw�n�i�H�w���j�@��������
%% �l�b�g���[�N����
% GoogLeNet�Ȃǂ̓ǂݍ��݁i���O�w�K�ς݁C�I���j
% 
% * <https://jp.mathworks.com/help/deeplearning/ref/googlenet.html https://jp.mathworks.com/help/deeplearning/ref/googlenet.html>

close all
deepnet = googlenet;
analyzeNetwork(deepnet)
layers = deepnet.Layers;
%% 
% ���͑w�̑���

inlayer = layers(1);
insz = inlayer.InputSize;
%% 
% �o�͑w�̑���

%outlayer = layers(end);
%categorynames = outlayer.ClassNames;
%disp(categorynames)
if isa(deepnet,'SeriesNetwork')
    lgraph = layerGraph(deepnet.Layers);
else
    lgraph = layerGraph(deepnet);
end
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]
%% �~�j�o�b�`�T�C�Y�ƍő�G�|�b�N��
%%
%�o�b�`�T�C�Y
miniBatchSize = 128;
%�ő�G�|�b�N��
maxEpochs = 100;
%% �t�H���_�i�P�[�X�j���̔䗦
% * �P���p�F���ؗp�F�e�X�g�p

p = [8 1 1];
%% �f�[�^����
% �ȉ��̃T�u�t�H���_�[�i���x�����j���܂ރf�[�^�t�H���_�[��z��
% 
% * 'EM_Major'
% * 'EM_minor'
% * 'MPE'
% * 'SJS'
% * 'SJS_TEN'
% * 'TEN'
% * 'DIHS'
% * 'DRESS'
% * 'MP_DR'

folder = './Img_For_SJS_TEN_Classifier_20190307/';
% �T�u�t�H���_���X�g
labelset =  { 'DIHS','DRESS','EM_Major','EM_minor','MPE','MP_DR','SJS','SJS_TEN','TEN' };
readfcn = @(x) skinread(x);
clear imds
fullfiles = {};
fulllabels = categorical();
for iLabel = 1:length(labelset)
    label = labelset{iLabel};
    subfolder = [ folder label ];
    subimds = imageDatastore(subfolder,'ReadFcn',readfcn,...
        'IncludeSubfolders',true);
    nFiles = length(subimds.Files);
    labels = cell(nFiles,1);
    for idx = 1:nFiles
        labels{idx} = label;
    end
    subimds.Labels = categorical(labels);
    disp(label)
    size(subimds.Labels)
    fullfiles = cat(1,fullfiles,subimds.Files);
    fulllabels = cat(1,fulllabels,subimds.Labels);
end
imds = imageDatastore(fullfiles,'Labels',fulllabels);
% �f�[�^����
disp('�f�[�^����')
size(imds.Files)
%% �N���X�Ē�`
% ���ݒ�i�I���j
% 
% 1.  {�d�ǖ�] (TEN�ASJS/TEN�ASJS)} �A{�ʏ��]�iEM_Major�AEM_minor�AMPE)} ��2�N���X����
% 
% 2.  {�d�ǖ�] (DIHS�ADRESS�AMP/DR)} �A{�ʏ��]�iEM_Major�AEM_minor�AMPE)}��2�N���X����
% 
% 3.  {SJS/TEN�Q (TEN�ASJS/TEN�ASJS)�AEM�Q (EM_Major�AEM_minor)�AMPE} ��3�N���X����
% 
% 4.  {DIHS/DRESS�Q (DIHS�ADRESS�AMP/DR)�AEM�Q (EM_Major�AEM_minor)�AMPE} ��3�N���X����
% 
% 5.  {TEN�ASJS/TEN�ASJS�AEM_Major�AEM_minor�AMPE} ��6�N���X����
% 
% 6.  {DIHS�ADRESS�AMP/DR�AEM_Major�AEM_minor�AMPE} ��6�N���X����

% ���̑I��
problemindex = 1;
% �I���W�i���̃��x����
labelnames = imds.Labels;
categories(labelnames)
switch problemindex
    case 1 % 2�N���X
        classnames = mergecats(labelnames,{'SJS','SJS_TEN','TEN'},'SJSTENCls');
        classnames = mergecats(classnames,{'EM_Major','EM_minor','MPE'}, 'EMMPECls');
    case 2 % 2�N���X
        classnames = mergecats(labelnames,{'DIHS','DRESS','MP_DR'},'DIHSDRESSCls');
        classnames = mergecats(classnames,{'EM_Major','EM_minor','MPE'}, 'EMMPECls');
    case 3 % 3�N���X
        classnames = mergecats(labelnames,{'SJS','SJS_TEN','TEN'},'SJSTENCls');
        classnames = mergecats(classnames,{'EM_Major','EM_minor'}, 'EMCls');
        classnames = renamecats(classnames,'MPE','MPECls');
    case 4 % 3�N���X
        classnames = mergecats(labelnames,{'DIHS','DRESS','MP_DR'},'DIHSDRESSCls');
        classnames = mergecats(classnames,{'EM_Major','EM_minor'}, 'EMCls');
        classnames = renamecats(classnames,'MPE','MPECls');
    case 5 % 6�N���X
        classnames = renamecats(labelnames,'TEN','TENCls');
        classnames = renamecats(classnames,'SJS_TEN','SJS_TENCls');
        classnames = renamecats(classnames,'SJS','SJSCls');
        classnames = renamecats(classnames,'EM_Major','EM_MajorCls');
        classnames = renamecats(classnames,'EM_minor','EM_minorCls');
        classnames = renamecats(classnames,'MPE','MPECls');
    case 6 % 6�N���X
        classnames = renamecats(labelnames,'DIHS','DIHSCls');
        classnames = renamecats(classnames,'DRESS','DRESSCls');
        classnames = renamecats(classnames,'MP_DR','MPDRCls');
        classnames = renamecats(classnames,'EM_Major','EM_MajorCls');
        classnames = renamecats(classnames,'EM_minor','EM_minorCls');
        classnames = renamecates(classnames,'MPE','MPECls');
    otherwise
        error('Invalid problemindex')
end
imds.Labels = classnames;
% �Ώۃ��x�����o
imdsflag = logical(contains(string(imds.Labels),'Cls'));
imdsfiles = imds.Files;
imdslabels = imds.Labels;
imds.Files = imdsfiles(imdsflag);
imds.Labels = removecats(imdslabels(imdsflag));
% ���x����
catlabels = categories(imds.Labels)
nlabels = numel(catlabels)
%% 
% 
%% �f�[�^�̕���
% * �N���X���ɁA�w�肵���䗦�ɉ����ăt�H���_�i�P�[�X�j���P���E���؁E�e�X�g�p�ɕ���
% * imds�̃t�@�C�����X�g�A���x�����X�g�̕����t���O�𒊏o

% �t�H���_�䗦�i�P���F���؁F�e�X�g�j�̐��K��
p = p/sum(p);

% �����_���Ƀt�H���_�P�ʂŕ����i�t�H���_���̔䗦�j
% �N���X�̒��o
catlist = categories(imds.Labels);
ncats = numel(catlist);

% �N���X���̃t�H���_�I��
datafolder = fullfile(pwd,folder);
infoSplit = cell(ncats,1);
tf4train = cell(ncats,1);
tf4validate = cell(ncats,1);
tf4test = cell(ncats,1);
for icat = 1:ncats
    catname = catlist{icat}
    
    % �e�N���X�̃f�[�^�Z�b�g���o
    catflag = (imds.Labels == catname);
    
    % �T�u�T�u�t�H���_�̃��X�g���o
    filelist = imds.Files(catflag);
    subsubfolderfilelist = cellfun(@(x) extractAfter(x,datafolder),filelist,'UniformOutput',false);
    [subfolderlist,remain] = strtok(subsubfolderfilelist,'\');
    subsubfolderlist = cellfun(@(x,y) strjoin({x,y},'\'),subfolderlist,strtok(remain,'\'),'UniformOutput',false);
    
    % �T�u�T�u�t�H���_�̃V���b�t��
    caselist = categories(categorical(subsubfolderlist));
    ncases = length(caselist);
    shuffledidx = randperm(ncases); 
    
    % �T�u�T�u�t�H���_�̐U�蕪��
    ncases4train = round(p(1)*ncases);
    ncases4validate = round(p(2)*ncases);
    cases4train = caselist(shuffledidx(1:ncases4train));
    cases4validate = caselist(shuffledidx(ncases4train+1:ncases4train+ncases4validate));
    cases4test = caselist(shuffledidx(ncases4train+ncases4validate+1:end));   

    % �t���O���o
    tf4train{icat} = contains(imds.Files,cases4train);
    tf4validate{icat} = contains(imds.Files,cases4validate);
    tf4test{icat} = contains(imds.Files,cases4test);
    
    % �U�蕪�����̋L�^
    infoSplit{icat}.cases4train = cases4train;
    infoSplit{icat}.cases4validate = cases4validate;
    infoSplit{icat}.cases4test = cases4test;
end
%% 
% �t���O�̊m�F
% 
% * �d���͂Ȃ����H
% * �S�f�[�^���J�o�[���Ă��邩�H

fulltf4train = zeros(length(imds.Files),1);
fulltf4validate = zeros(length(imds.Files),1);
fulltf4test = zeros(length(imds.Files),1);
for icat=1:ncats
    fulltf4train = fulltf4train + tf4train{icat};
    fulltf4validate = fulltf4validate + tf4validate{icat};
    fulltf4test = fulltf4test + tf4test{icat};
end
tfcheck = fulltf4train + fulltf4validate + fulltf4test;
disp(['#Total: ' num2str(length(imds.Files))])
disp(['#Files for Train : ' num2str(nnz(fulltf4train))])
disp(['#Files for Validate: ' num2str(nnz(fulltf4validate))])
disp(['#Files for Test: ' num2str(nnz(fulltf4test))])
assert(~any(tfcheck > 1),'Intersection')
assert(~any(tfcheck == 0),'Uncover')
%% �f�[�^�X�g�A�̕���
%%
subdsTrain = cell(ncats,1);
subdsValidate = cell(ncats,1);
subdsTest = cell(ncats,1);
for icat=1:ncats
    ntrain = nnz(tf4train{icat});
    nvalidate = nnz(tf4validate{icat});
    [subdsTrain{icat},subdsValidate{icat},subdsTest{icat}] = ...
        splitEachLabel(imds,ntrain,nvalidate,'Include',catlist{icat});
end
%% �f�[�^�X�g�A�̍���
%%
filesTrain = {};
filesValidate = {};
filesTest = {};
labelsTrain = categorical();
labelsValidate = categorical();
labelsTest = categorical();
for icat = 1:ncats
    filesTrain = cat(1,filesTrain,subdsTrain{icat}.Files);
    labelsTrain = cat(1,labelsTrain,subdsTrain{icat}.Labels);
    filesValidate = cat(1,filesValidate,subdsValidate{icat}.Files);
    labelsValidate = cat(1,labelsValidate,subdsValidate{icat}.Labels);
    filesTest = cat(1,filesTest,subdsTest{icat}.Files);
    labelsTest = cat(1,labelsTest,subdsTest{icat}.Labels);
end
dsTrain = imageDatastore(filesTrain,'Labels',labelsTrain);
dsValidate = imageDatastore(filesValidate,'Labels',labelsValidate);
dsTest = imageDatastore(filesTest,'Labels',labelsTest);

dsTrain = shuffle(dsTrain);
dsValidate = shuffle(dsValidate);
dsTest = shuffle(dsTest);

% �����_���ɕ����i�]���@�j
%[dsTrain,dsValidate,dsTest] = splitEachLabel(imds,...
% length(dsTrain.Files),length(dsValidate.Files),length(dsTest.Files),...
% 'randomized');

disp('dsTrain')
countEachLabel(dsTrain)
disp('dsValidate')
countEachLabel(dsValidate)
disp('dsTest')
countEachLabel(dsTest)
%% �l�b�g���[�N�ݒ�
% �S�����w�̐����i�N���X�����C���j

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(nlabels, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',1,...10, ...
        'BiasLearnRateFactor',1);...10);
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
newLearnableLayer = convolution2dLayer(1,nlabels, ...
    'Name','new_conv', ...
    'WeightLearnRateFactor',1,...0, ...
    'BiasLearnRateFactor',1);%...0);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
%% �������C���̓���
%%
%layers = lgraph.Layers;
%connections = lgraph.Connections;

%layers(1:10) = freezeWeights(layers(1:10));
%lgraph = createLgraphUsingConnections(layers,connections);
%% ���ރl�b�g���[�N�̓]�ڊw�K
% �I�v�V�����ݒ�

numIterationsPerEpoch = floor(numel(dsTrain.Labels)/miniBatchSize);
augimdsValidation = augmentedImageDatastore(insz(1:2),dsValidate); % �T�C�Y�C���̂�
opts = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs', maxEpochs, ...
    'InitialLearnRate',1e-4,...3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',numIterationsPerEpoch,...
    'ValidationFrequency',100, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% 
% �f�[�^�g��

%pixelRange = [ -30 30 ];
%scaleRange = [ 0.9 1.1 ];
augmenter = imageDataAugmenter(...
    'RandRotation',[0 360]...
    ...'RandXReflection',true, ...
    ...'RandYReflection',true...,...
    ...'RandXTranslation',pixelRange, ...
    ...'RandYTranslation',pixelRange, ...
    ...'RandXScale',scaleRange, ...
    ...'RandYScale',scaleRange...
    );
datasource = augmentedImageDatastore(insz(1:2),dsTrain,...
    'DataAugmentation',augmenter);
%% 
% �g���[�j���O

[skinnet,info] = trainNetwork(datasource,lgraph,opts);
%% 
% �l�b�g���[�N�̕ۑ�

dt = char(datetime('now','TimeZone','local','Format','d-MM-y-HH-mm-ssZ'));
targetdir = ['./skinnet' dt];
if exist(targetdir,'dir') ~= 7
    mkdir(targetdir)
end
save([targetdir '/skinnet.mat'],...
    'skinnet','info','dsTrain','dsTest','dsValidate','infoSplit','catlist')
%save skinnet skinnet info -mat
%% ���ރe�X�g
% �e�X�g�摜�̃N���X�\��

datatest = augmentedImageDatastore(insz(1:2),dsTest); % �T�C�Y�C���̂�
[skinPreds,probs] = classify(skinnet,datatest,...
    'MiniBatchSize',miniBatchSize);
%% 
% �N���X�\���e�X�g�̐��x�]��

accuracy = mean(skinPreds == dsTest.Labels);
disp(['Macro Accuracy: ' num2str(accuracy)]);
%% 
% �����s��ɂ������@ (�����i%�j)

figure(2)
[skinconf,skinnames] = confusionmat(dsTest.Labels,skinPreds);
skinconfnorm = skinconf*diag(1./sum(skinconf))*100;
skinconfnorm_ff = flipud(fliplr (skinconfnorm))
skinconf_ff = flipud(fliplr (skinconf))
skinnames_ff = flipud(fliplr(skinnames))
hh = heatmap(skinnames_ff,skinnames_ff,skinconfnorm_ff);
%hh = heatmap(skinnames,skinnames,skinconfnorm);
title('Performance of the DCNN classification')
xlabel('Diagnosis')
ylabel('DCNN output')
%xlswrite('skinconfnorm_ff',skinconfnorm_ff)�F�G���[���o�����߃R�����g�A�E�g
%xlswrite('skinconfnorm',skinconfnorm)
%% 
% �����s��ɂ������A (��ΐ�)

figure(3)
hh = heatmap(skinnames_ff,skinnames_ff,skinconf_ff);
%hh = heatmap(skinnames,skinnames,skinconf);
title('�e�X�g���ʂ̉���')
xlabel('���͉摜')
ylabel('���ތ���')
%xlswrite('skinconf_ff',skinconf_ff)�F�G���[���ł����߃R�����g�A�E�g
%xlswrite('skinconf',skinconf)
%% 
% �e�X�g���ʂ̗�

figure(4)
idx = randperm(numel(dsTest.Files),100);
for i = 1:6
    subplot(3,2,i)
    I = readimage(dsTest,idx(i));
    imshow(I)
    predlabel = skinPreds(idx(i));
    truelabel = dsTest.Labels(idx(i));
    
    title('Pred: ' + string(predlabel) + ' (' + string(truelabel) +'), ' + num2str(100*max(probs(idx(i),:)),3) + '%');
end
%% 
% F�l

truepositive = diag(skinconf);
conditionpos = sum(skinconf,1);
predcondpos = sum(skinconf,2);
%% 
% �}�C�N���]��

%disp(skinnames.')
% recall
murecall = truepositive(:).'./conditionpos(:).';
%disp(['�}�C�N���Č���(�� Recall): ' num2str(murecall)]);

% precision
muprecision = truepositive(:).'./predcondpos(:).';
%disp(['�}�C�N���K����(�� Precision): ' num2str(muprecision)]);

% F-score
mufmeasure = 2*murecall.*muprecision./(muprecision+murecall);
%disp(['�}�C�N��F�l(�� F-measure): ' num2str(mufmeasure)]);

%
T = [murecall; muprecision; mufmeasure];
v = cell(3,ncats);
for icat = 1:ncats
    for idx = 1:3
        v{idx,icat} = T(idx,icat);
    end
end
T = cell2table(v);
T.Properties.RowNames = {'Recall','Precision','F-measure'};
T.Properties.VariableNames = string(catlist);
T
%% 
% �}�N���]��

% recall
emrecall = mean(truepositive(:)./conditionpos(:));
disp(['Macro Recall: ' num2str(emrecall)]);

% precision
emprecision = mean(truepositive(:)./predcondpos(:));
disp(['Macro Precision: ' num2str(emprecision)]);

% F-score
emfmeasure = 2*emrecall*emprecision/(emprecision+emrecall);
disp(['Macro F-measure: ' num2str(emfmeasure)]);
%% �T���v���̊���
%%
figure(5)
histogram(imds.Labels)
title('�T���v���̊���')
%% �����ʕ���
% �B��w�̏o��

layerFeature = skinnet.Layers(end-3).Name;
disp(layerFeature)
features = activations(skinnet,datatest,layerFeature,'OutputAs','rows');
size(features)
%% 
% ������Ԃ̉��� (2�����j

figure(6)
Y = tsne(features);
gscatter(Y(:,1),Y(:,2),dsTest.Labels)
%% 
% ������Ԃ̉��� (3�����j

figure(7)
Y = tsne(features,'NumDimensions',3);
scatter3(Y(:,1),Y(:,2),Y(:,3),5,dsTest.Labels,'filled')
%% 
% ���ފ��ROC�Ȑ� (���x�E���ٓx)_�C����

figure(9)
plot(1-X,Y,'b')
hold on
plot(1-OPTROCPT(1),OPTROCPT(2),'bo')
xlabel('Sensitivity') 
ylabel('Specificity')
title('Classification performance of the DCNN')
%% *Sensitiyity and Specificity of TRN, BCD and DCNN*
%% �f�[�^�ǂݍ���
%%
T_ALL = readtable('AI_TEST_FOR_MATLAB_20190603_01.xlsx');
T_ALL = sortrows(T_ALL,'CL3');
%% BCD, TRN, DCNN�ɕ���
%%
T_TRN = T_ALL(T_ALL.CL3 == 0,:);
T_BCD = T_ALL(T_ALL.CL3 == 1,:);
T_DER = T_ALL(T_ALL.CL3 == 0 | 1,:);
T_DCNN = T_ALL(T_ALL.CL3 == 2,:);
%% �p�����[�^�v�Z
% Accuracy (%)

Accuracy_ALL = 100*(T_ALL.TP + T_ALL.TN)./(T_ALL.TP + T_ALL.FP + T_ALL.TN + T_ALL.FN);
Accuracy_BCD = 100*(T_BCD.TP + T_BCD.TN)./(T_BCD.TP + T_BCD.FP + T_BCD.TN + T_BCD.FN);
Accuracy_TRN = 100*(T_TRN.TP + T_TRN.TN)./(T_TRN.TP + T_TRN.FP + T_TRN.TN + T_TRN.FN);
Accuracy_DER = 100*(T_DER.TP + T_DER.TN)./(T_DER.TP + T_DER.FP + T_DER.TN + T_DER.FN);
Accuracy_DCNN = 100*(T_DCNN.TP + T_DCNN.TN)./(T_DCNN.TP + T_DCNN.FP + T_DCNN.TN + T_DCNN.FN);
%% 
% Precision (%)

Precision_ALL = 100*T_ALL.TP ./(T_ALL.TP + T_ALL.FP);
Precision_BCD = 100*T_BCD.TP./(T_BCD.TP + T_BCD.FP);
Precision_TRN = 100*T_TRN.TP./(T_TRN.TP + T_TRN.FP);
Precision_DER = 100*T_DER.TP./(T_DER.TP + T_DER.FP);
Precision_DCNN = 100*T_DCNN.TP./(T_DCNN.TP + T_DCNN.FP);
%% 
% Recall (Sensitivity) (%)

Recall_ALL = 100*T_ALL.TP ./(T_ALL.TP + T_ALL.FN);
Recall_BCD = 100*T_BCD.TP./(T_BCD.TP + T_BCD.FN);
Recall_TRN = 100*T_TRN.TP./(T_TRN.TP + T_TRN.FN);
Recall_DER = 100*T_DER.TP./(T_DER.TP + T_DER.FN);
Recall_DCNN = 100*T_DCNN.TP./(T_DCNN.TP + T_DCNN.FN);
%% 
% Specificity (%)

Specificity_ALL = 100*T_ALL.TN ./(T_ALL.FP + T_ALL.TN);
Specificity_BCD = 100*T_BCD.TN ./(T_BCD.FP + T_BCD.TN);
Specificity_TRN = 100*T_TRN.TN ./(T_TRN.FP + T_TRN.TN);
Specificity_DER = 100*T_DER.TN ./(T_DER.FP + T_DER.TN);
Specificity_DCNN = 100*T_DCNN.TN ./(T_DCNN.FP + T_DCNN.TN);
%% 
% F-measure (%)

F_measure_ALL = 2 * Recall_ALL .* Precision_ALL ./ (Recall_ALL + Precision_ALL);
F_measure_BCD = 2 * Recall_BCD .* Precision_BCD ./ (Recall_BCD + Precision_BCD);
F_measure_TRN = 2 * Recall_TRN .* Precision_TRN ./ (Recall_TRN + Precision_TRN);
F_measure_DER = 2 * Recall_DER .* Precision_DER ./ (Recall_DER + Precision_DER);
F_measure_DCNN = 2 * Recall_DCNN .* Precision_DCNN ./ (Recall_DCNN + Precision_DCNN);
%% 
% ���ފ��ROC�Ȑ� (���x�E���ٓx) TRN, BCD�v���b�g����

figure(10)
plot(1-X,Y,'b')
hold on
plot(1-OPTROCPT(1),OPTROCPT(2),'bo')
plot(Recall_TRN/100,Specificity_TRN/100,'r.')
plot(mean(Recall_TRN/100),mean(Specificity_TRN/100),'g.')
plot(mean(Recall_BCD/100),mean(Specificity_BCD/100),'gd')
xlabel('Sensitivity') 
ylabel('Specificity')
title('Classification performance of the DCNN and dermatologists')
%% 
% �S���[�N�X�y�[�X�ϐ��̕ۑ�

save(dt)
%% �O�����֐��̒�`
%%
function img = skinread(file)
% �摜�̓ǂݍ���
img = imread(file);
% �����������`�̈�ŃN���b�s���O
imsz = size(img);
mnsz = min(imsz(1:2));
xmin = floor(imsz(2)-mnsz)/2+1;
ymin = floor(imsz(1)-mnsz)/2+1;
img = imcrop(img,[ xmin ymin mnsz mnsz ]);
% ���͑w�ɍ��킹�ă��T�C�Y
%img = imresize(img,insz(1:2));
end