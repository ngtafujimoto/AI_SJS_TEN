%% 薬疹分類器学習・検証プログラム
%% 概要
% * 画像を{ 'DIHS','DRESS','EM_Major','EM_minor','MPE','MP_DR','SJS','SJS_TEN','TEN'}に分類
% * サブサブフォルダ毎でテストデータとに訓練用・検証用データを分割
% * 事前学習済みネットワーク(GoogLeNetなど）の転移学習で分類器を設計
% * 前処理では中央部正方形でクリッピング，入力層に合わせてリサイズ
% 
% （転移学習） <https://jp.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html?lang=en 
% https://jp.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html?lang=en>
% 
% 2019/1/25  新潟大学　自然科学系（工学部）　村松正吾
%% ネットワーク準備
% GoogLeNetなどの読み込み（事前学習済み，選択可）
% 
% * <https://jp.mathworks.com/help/deeplearning/ref/googlenet.html https://jp.mathworks.com/help/deeplearning/ref/googlenet.html>

close all
deepnet = googlenet;
analyzeNetwork(deepnet)
layers = deepnet.Layers;
%% 
% 入力層の属性

inlayer = layers(1);
insz = inlayer.InputSize;
%% 
% 出力層の属性

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
%% ミニバッチサイズと最大エポック数
%%
%バッチサイズ
miniBatchSize = 128;
%最大エポック数
maxEpochs = 100;
%% フォルダ（ケース）数の比率
% * 訓練用：検証用：テスト用

p = [8 1 1];
%% データ準備
% 以下のサブフォルダー（ラベル名）を含むデータフォルダーを想定
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
% サブフォルダリスト
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
% データ総数
disp('データ総数')
size(imds.Files)
%% クラス再定義
% 問題設定（選択可）
% 
% 1.  {重症薬疹 (TEN、SJS/TEN、SJS)} 、{通常薬疹（EM_Major、EM_minor、MPE)} の2クラス分類
% 
% 2.  {重症薬疹 (DIHS、DRESS、MP/DR)} 、{通常薬疹（EM_Major、EM_minor、MPE)}の2クラス分類
% 
% 3.  {SJS/TEN群 (TEN、SJS/TEN、SJS)、EM群 (EM_Major、EM_minor)、MPE} の3クラス分類
% 
% 4.  {DIHS/DRESS群 (DIHS、DRESS、MP/DR)、EM群 (EM_Major、EM_minor)、MPE} の3クラス分類
% 
% 5.  {TEN、SJS/TEN、SJS、EM_Major、EM_minor、MPE} の6クラス分類
% 
% 6.  {DIHS、DRESS、MP/DR、EM_Major、EM_minor、MPE} の6クラス分類

% 問題の選択
problemindex = 1;
% オリジナルのラベル名
labelnames = imds.Labels;
categories(labelnames)
switch problemindex
    case 1 % 2クラス
        classnames = mergecats(labelnames,{'SJS','SJS_TEN','TEN'},'SJSTENCls');
        classnames = mergecats(classnames,{'EM_Major','EM_minor','MPE'}, 'EMMPECls');
    case 2 % 2クラス
        classnames = mergecats(labelnames,{'DIHS','DRESS','MP_DR'},'DIHSDRESSCls');
        classnames = mergecats(classnames,{'EM_Major','EM_minor','MPE'}, 'EMMPECls');
    case 3 % 3クラス
        classnames = mergecats(labelnames,{'SJS','SJS_TEN','TEN'},'SJSTENCls');
        classnames = mergecats(classnames,{'EM_Major','EM_minor'}, 'EMCls');
        classnames = renamecats(classnames,'MPE','MPECls');
    case 4 % 3クラス
        classnames = mergecats(labelnames,{'DIHS','DRESS','MP_DR'},'DIHSDRESSCls');
        classnames = mergecats(classnames,{'EM_Major','EM_minor'}, 'EMCls');
        classnames = renamecats(classnames,'MPE','MPECls');
    case 5 % 6クラス
        classnames = renamecats(labelnames,'TEN','TENCls');
        classnames = renamecats(classnames,'SJS_TEN','SJS_TENCls');
        classnames = renamecats(classnames,'SJS','SJSCls');
        classnames = renamecats(classnames,'EM_Major','EM_MajorCls');
        classnames = renamecats(classnames,'EM_minor','EM_minorCls');
        classnames = renamecats(classnames,'MPE','MPECls');
    case 6 % 6クラス
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
% 対象ラベル抽出
imdsflag = logical(contains(string(imds.Labels),'Cls'));
imdsfiles = imds.Files;
imdslabels = imds.Labels;
imds.Files = imdsfiles(imdsflag);
imds.Labels = removecats(imdslabels(imdsflag));
% ラベル数
catlabels = categories(imds.Labels)
nlabels = numel(catlabels)
%% 
% 
%% データの分割
% * クラス毎に、指定した比率に応じてフォルダ（ケース）を訓練・検証・テスト用に分割
% * imdsのファイルリスト、ラベルリストの分割フラグを抽出

% フォルダ比率（訓練：検証：テスト）の正規化
p = p/sum(p);

% ランダムにフォルダ単位で分離（フォルダ数の比率）
% クラスの抽出
catlist = categories(imds.Labels);
ncats = numel(catlist);

% クラス毎のフォルダ選択
datafolder = fullfile(pwd,folder);
infoSplit = cell(ncats,1);
tf4train = cell(ncats,1);
tf4validate = cell(ncats,1);
tf4test = cell(ncats,1);
for icat = 1:ncats
    catname = catlist{icat}
    
    % 各クラスのデータセット抽出
    catflag = (imds.Labels == catname);
    
    % サブサブフォルダのリスト抽出
    filelist = imds.Files(catflag);
    subsubfolderfilelist = cellfun(@(x) extractAfter(x,datafolder),filelist,'UniformOutput',false);
    [subfolderlist,remain] = strtok(subsubfolderfilelist,'\');
    subsubfolderlist = cellfun(@(x,y) strjoin({x,y},'\'),subfolderlist,strtok(remain,'\'),'UniformOutput',false);
    
    % サブサブフォルダのシャッフル
    caselist = categories(categorical(subsubfolderlist));
    ncases = length(caselist);
    shuffledidx = randperm(ncases); 
    
    % サブサブフォルダの振り分け
    ncases4train = round(p(1)*ncases);
    ncases4validate = round(p(2)*ncases);
    cases4train = caselist(shuffledidx(1:ncases4train));
    cases4validate = caselist(shuffledidx(ncases4train+1:ncases4train+ncases4validate));
    cases4test = caselist(shuffledidx(ncases4train+ncases4validate+1:end));   

    % フラグ抽出
    tf4train{icat} = contains(imds.Files,cases4train);
    tf4validate{icat} = contains(imds.Files,cases4validate);
    tf4test{icat} = contains(imds.Files,cases4test);
    
    % 振り分け情報の記録
    infoSplit{icat}.cases4train = cases4train;
    infoSplit{icat}.cases4validate = cases4validate;
    infoSplit{icat}.cases4test = cases4test;
end
%% 
% フラグの確認
% 
% * 重複はないか？
% * 全データをカバーしているか？

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
%% データストアの分割
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
%% データストアの合成
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

% ランダムに分割（従来法）
%[dsTrain,dsValidate,dsTest] = splitEachLabel(imds,...
% length(dsTrain.Files),length(dsValidate.Files),length(dsTest.Files),...
% 'randomized');

disp('dsTrain')
countEachLabel(dsTrain)
disp('dsValidate')
countEachLabel(dsValidate)
disp('dsTest')
countEachLabel(dsTest)
%% ネットワーク設定
% 全結合層の生成（クラス数を修正）

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
%% 初期レイヤの凍結
%%
%layers = lgraph.Layers;
%connections = lgraph.Connections;

%layers(1:10) = freezeWeights(layers(1:10));
%lgraph = createLgraphUsingConnections(layers,connections);
%% 分類ネットワークの転移学習
% オプション設定

numIterationsPerEpoch = floor(numel(dsTrain.Labels)/miniBatchSize);
augimdsValidation = augmentedImageDatastore(insz(1:2),dsValidate); % サイズ修整のみ
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
% データ拡張

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
% トレーニング

[skinnet,info] = trainNetwork(datasource,lgraph,opts);
%% 
% ネットワークの保存

dt = char(datetime('now','TimeZone','local','Format','d-MM-y-HH-mm-ssZ'));
targetdir = ['./skinnet' dt];
if exist(targetdir,'dir') ~= 7
    mkdir(targetdir)
end
save([targetdir '/skinnet.mat'],...
    'skinnet','info','dsTrain','dsTest','dsValidate','infoSplit','catlist')
%save skinnet skinnet info -mat
%% 分類テスト
% テスト画像のクラス予測

datatest = augmentedImageDatastore(insz(1:2),dsTest); % サイズ修整のみ
[skinPreds,probs] = classify(skinnet,datatest,...
    'MiniBatchSize',miniBatchSize);
%% 
% クラス予測テストの精度評価

accuracy = mean(skinPreds == dsTest.Labels);
disp(['Macro Accuracy: ' num2str(accuracy)]);
%% 
% 混合行列による可視化① (割合（%）)

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
%xlswrite('skinconfnorm_ff',skinconfnorm_ff)：エラーが出たためコメントアウト
%xlswrite('skinconfnorm',skinconfnorm)
%% 
% 混合行列による可視化② (絶対数)

figure(3)
hh = heatmap(skinnames_ff,skinnames_ff,skinconf_ff);
%hh = heatmap(skinnames,skinnames,skinconf);
title('テスト結果の可視化')
xlabel('入力画像')
ylabel('分類結果')
%xlswrite('skinconf_ff',skinconf_ff)：エラーがでたためコメントアウト
%xlswrite('skinconf',skinconf)
%% 
% テスト結果の例

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
% F値

truepositive = diag(skinconf);
conditionpos = sum(skinconf,1);
predcondpos = sum(skinconf,2);
%% 
% マイクロ評価

%disp(skinnames.')
% recall
murecall = truepositive(:).'./conditionpos(:).';
%disp(['マイクロ再現率(μ Recall): ' num2str(murecall)]);

% precision
muprecision = truepositive(:).'./predcondpos(:).';
%disp(['マイクロ適合率(μ Precision): ' num2str(muprecision)]);

% F-score
mufmeasure = 2*murecall.*muprecision./(muprecision+murecall);
%disp(['マイクロF値(μ F-measure): ' num2str(mufmeasure)]);

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
% マクロ評価

% recall
emrecall = mean(truepositive(:)./conditionpos(:));
disp(['Macro Recall: ' num2str(emrecall)]);

% precision
emprecision = mean(truepositive(:)./predcondpos(:));
disp(['Macro Precision: ' num2str(emprecision)]);

% F-score
emfmeasure = 2*emrecall*emprecision/(emprecision+emrecall);
disp(['Macro F-measure: ' num2str(emfmeasure)]);
%% サンプルの割合
%%
figure(5)
histogram(imds.Labels)
title('サンプルの割合')
%% 特徴量分析
% 隠れ層の出力

layerFeature = skinnet.Layers(end-3).Name;
disp(layerFeature)
features = activations(skinnet,datatest,layerFeature,'OutputAs','rows');
size(features)
%% 
% 特徴空間の可視化 (2次元）

figure(6)
Y = tsne(features);
gscatter(Y(:,1),Y(:,2),dsTest.Labels)
%% 
% 特徴空間の可視化 (3次元）

figure(7)
Y = tsne(features,'NumDimensions',3);
scatter3(Y(:,1),Y(:,2),Y(:,3),5,dsTest.Labels,'filled')
%% 
% 分類器のROC曲線 (感度・特異度)_修正版

figure(9)
plot(1-X,Y,'b')
hold on
plot(1-OPTROCPT(1),OPTROCPT(2),'bo')
xlabel('Sensitivity') 
ylabel('Specificity')
title('Classification performance of the DCNN')
%% *Sensitiyity and Specificity of TRN, BCD and DCNN*
%% データ読み込み
%%
T_ALL = readtable('AI_TEST_FOR_MATLAB_20190603_01.xlsx');
T_ALL = sortrows(T_ALL,'CL3');
%% BCD, TRN, DCNNに分割
%%
T_TRN = T_ALL(T_ALL.CL3 == 0,:);
T_BCD = T_ALL(T_ALL.CL3 == 1,:);
T_DER = T_ALL(T_ALL.CL3 == 0 | 1,:);
T_DCNN = T_ALL(T_ALL.CL3 == 2,:);
%% パラメータ計算
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
% 分類器のROC曲線 (感度・特異度) TRN, BCDプロットあり

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
% 全ワークスペース変数の保存

save(dt)
%% 前処理関数の定義
%%
function img = skinread(file)
% 画像の読み込み
img = imread(file);
% 中央部正方形領域でクリッピング
imsz = size(img);
mnsz = min(imsz(1:2));
xmin = floor(imsz(2)-mnsz)/2+1;
ymin = floor(imsz(1)-mnsz)/2+1;
img = imcrop(img,[ xmin ymin mnsz mnsz ]);
% 入力層に合わせてリサイズ
%img = imresize(img,insz(1:2));
end