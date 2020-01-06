%% *Welch t-test TRN v.s. BCD v.s. DCNN*
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
% Precision (Positive predictive value: PPV) (%)

Precision_ALL = 100*T_ALL.TP ./(T_ALL.TP + T_ALL.FP);
Precision_BCD = 100*T_BCD.TP./(T_BCD.TP + T_BCD.FP);
Precision_TRN = 100*T_TRN.TP./(T_TRN.TP + T_TRN.FP);
Precision_DER = 100*T_DER.TP./(T_DER.TP + T_DER.FP);
Precision_DCNN = 100*T_DCNN.TP./(T_DCNN.TP + T_DCNN.FP);
%% 
% Negative predictive value: NPV (%)

NPV_ALL = 100*T_ALL.TN ./(T_ALL.TN + T_ALL.FN);
NPV_BCD = 100*T_BCD.TN./(T_BCD.TN + T_BCD.FN);
NPV_TRN = 100*T_TRN.TN./(T_TRN.TN + T_TRN.FN);
NPV_DER = 100*T_DER.TN./(T_DER.TN + T_DER.FN);
NPV_DCNN = 100*T_DCNN.TN./(T_DCNN.TN + T_DCNN.FN);
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
%% 各パラメーターの平均値と95%CI
% Accuracy_BCD

mean(Accuracy_BCD)
pd = fitdist(Accuracy_BCD,'Normal')
ci = paramci(pd)
%% 
% Accuracy_TRN

mean(Accuracy_TRN)
pd = fitdist(Accuracy_TRN,'Normal')
ci = paramci(pd)
%% 
% Accuracy_DCNN

mean(Accuracy_DCNN)
pd = fitdist(Accuracy_DCNN,'Normal')
ci = paramci(pd)
%% 
% Precision_BCD

mean(Precision_BCD)
pd = fitdist(Precision_BCD,'Normal')
ci = paramci(pd)
%% 
% Precision_TRN

mean(Precision_TRN)
pd = fitdist(Precision_TRN,'Normal')
ci = paramci(pd)
%% 
% Precision_DCNN

mean(Precision_DCNN)
pd = fitdist(Precision_DCNN,'Normal')
ci = paramci(pd)
%% 
% NPV_BCD

mean(NPV_BCD)
pd = fitdist(NPV_BCD,'Normal')
ci = paramci(pd)
%% 
% NPV_TRN

mean(NPV_TRN)
pd = fitdist(NPV_TRN,'Normal')
ci = paramci(pd)
%% 
% NPV_DCNN

mean(NPV_DCNN)
pd = fitdist(NPV_DCNN,'Normal')
ci = paramci(pd)
%% 
% Recall_BCD

mean(Recall_BCD)
pd = fitdist(Recall_BCD,'Normal')
ci = paramci(pd)
%% 
% Recall_TRN

mean(Recall_TRN)
pd = fitdist(Recall_TRN,'Normal')
ci = paramci(pd)
%% 
% Recall_DCNN

mean(Recall_DCNN)
pd = fitdist(Recall_DCNN,'Normal')
ci = paramci(pd)
%% 
% Specificity_BCD

mean(Specificity_BCD)
pd = fitdist(Specificity_BCD,'Normal')
ci = paramci(pd)
%% 
% Specificity_TRN

mean(Specificity_TRN)
pd = fitdist(Specificity_TRN,'Normal')
ci = paramci(pd)
%% 
% Specificity_DCNN

mean(Specificity_DCNN)
pd = fitdist(Specificity_DCNN,'Normal')
ci = paramci(pd)
%% Welch t-test: Accuracy
% TRNs v.s. BCDs

[h,p,ci,stats] = ttest2(Accuracy_TRN,Accuracy_BCD)
%% 
% BCDs v.s. DCNN

[h,p,ci,stats] = ttest2(Accuracy_BCD,Accuracy_DCNN)
%% 
% TRNs v.s. DCNN

[h,p,ci,stats] = ttest2(Accuracy_TRN,Accuracy_DCNN)
%% 
% Dermatologists v.s. DCNN

[h,p,ci,stats] = ttest2(Accuracy_DER,Accuracy_DCNN)
%% Welch t-test: Precision (PPV)
% TRNs v.s. BCDs

[h,p,ci,stats] = ttest2(Precision_TRN,Precision_BCD)
%% 
% BCDs v.s. DCNN

[h,p,ci,stats] = ttest2(Precision_BCD,Precision_DCNN)
%% 
% TRNs v.s. DCNN

[h,p,ci,stats] = ttest2(Precision_TRN,Precision_DCNN)
%% 
% Dermatologists v.s. DCNN

[h,p,ci,stats] = ttest2(Precision_DER,Precision_DCNN)
%% Welch t-test: NPV
% TRNs v.s. BCDs

[h,p,ci,stats] = ttest2(NPV_TRN,NPV_BCD)
%% 
% BCDs v.s. DCNN

[h,p,ci,stats] = ttest2(NPV_BCD,NPV_DCNN)
%% 
% TRNs v.s. DCNN

[h,p,ci,stats] = ttest2(NPV_TRN,NPV_DCNN)
%% 
% Dermatologists v.s. DCNN

[h,p,ci,stats] = ttest2(NPV_DER,NPV_DCNN)
%% Welch t-test: Recall (Sensitivity)
% TRNs v.s. BCDs

[h,p,ci,stats] = ttest2(Recall_TRN,Recall_BCD)
%% 
% BCDs v.s. DCNN

[h,p,ci,stats] = ttest2(Recall_BCD,Recall_DCNN)
%% 
% TRNs v.s. DCNN

[h,p,ci,stats] = ttest2(Recall_TRN,Recall_DCNN)
%% 
% Dermatologists v.s. DCNN

[h,p,ci,stats] = ttest2(Recall_DER,Recall_DCNN)
%% Welch t-test: Specificity
% TRNs v.s. BCDs

[h,p,ci,stats] = ttest2(Specificity_TRN,Specificity_BCD)
%% 
% BCDs v.s. DCNN

[h,p,ci,stats] = ttest2(Specificity_BCD,Specificity_DCNN)
%% 
% TRNs v.s. DCNN

[h,p,ci,stats] = ttest2(Specificity_TRN,Specificity_DCNN)
%% 
% Dermatologists v.s. DCNN

[h,p,ci,stats] = ttest2(Specificity_DER,Specificity_DCNN)
%% Welch t-test: F_measure
% TRNs v.s. BCDs

[h,p,ci,stats] = ttest2(F_measure_TRN,F_measure_BCD)
%% 
% BCDs v.s. DCNN

[h,p,ci,stats] = ttest2(F_measure_BCD,F_measure_DCNN)
%% 
% TRNs v.s. DCNN

[h,p,ci,stats] = ttest2(F_measure_TRN,F_measure_DCNN)
%% 
% Dermatologists v.s. DCNN

[h,p,ci,stats] = ttest2(F_measure_DER,F_measure_DCNN)
%% 可視化 (TRNs v.s. BCDs v.s. DCNN)
% Accuracy

data = {Accuracy_TRN,Accuracy_BCD,Accuracy_DCNN};
catIdx = [T_ALL.CL3];
figure
plotSpread(data,'categoryIdx',catIdx,... 
    'categoryMarkers',{'.','d','x'},'categoryColors',{'k','k','k'})
hold on
boxplot(Accuracy_ALL,T_ALL.CL3,'Labels',{'TRN','BCD','DCNN'},'Colors','k','OutlierSize',2)
ylabel('Accuracy (%)')
title('Accuracy of image classification by TRN, BCD and DCNN')
xlim([0.50 3.50])
ylim([0 100])
hold off
%% 
% Precision (Positive predictive value: PPV)

data = {Precision_TRN,Precision_BCD,Precision_DCNN};
catIdx = [T_ALL.CL3];
figure
plotSpread(data,'categoryIdx',catIdx,... 
    'categoryMarkers',{'.','d','x'},'categoryColors',{'k','k','k'})
hold on
boxplot(Precision_ALL,T_ALL.CL3,'Labels',{'TRN','BCD','DCNN'},'Colors','k','OutlierSize',2)
ylabel('Precision (Positive predictive value: PPV) (%)')
title('Precision of image classification by TRN, BCD and DCNN')
xlim([0.50 3.50])
ylim([0 100])
hold off
%% 
% Negative predictive value: NPV

data = {NPV_TRN,NPV_BCD,NPV_DCNN};
catIdx = [T_ALL.CL3];
figure
plotSpread(data,'categoryIdx',catIdx,... 
    'categoryMarkers',{'.','d','x'},'categoryColors',{'k','k','k'})
hold on
boxplot(NPV_ALL,T_ALL.CL3,'Labels',{'TRN','BCD','DCNN'},'Colors','k','OutlierSize',2)
ylabel('Negative predictive value: NPV (%)')
title('NPV of image classification by TRN, BCD and DCNN')
xlim([0.50 3.50])
ylim([0 100])
hold off
%% 
% Recall

data = {Recall_TRN,Recall_BCD,Recall_DCNN};
catIdx = [T_ALL.CL3];
figure
plotSpread(data,'categoryIdx',catIdx,... 
    'categoryMarkers',{'.','d','x'},'categoryColors',{'k','k','k'})
hold on
boxplot(Recall_ALL,T_ALL.CL3,'Labels',{'TRN','BCD','DCNN'},'Colors','k','OutlierSize',2)
ylabel('Recall (%)')
title('Recall of image classification by TRN, BCD and DCNN')
xlim([0.50 3.50])
ylim([0 100])
hold off
%% 
% Specificity

data = {Specificity_TRN,Specificity_BCD,Specificity_DCNN};
catIdx = [T_ALL.CL3];
figure
plotSpread(data,'categoryIdx',catIdx,... 
    'categoryMarkers',{'.','d','x'},'categoryColors',{'k','k','k'})
hold on
boxplot(Specificity_ALL,T_ALL.CL3,'Labels',{'TRN','BCD','DCNN'},'Colors','k','OutlierSize',2)
ylabel('Specificity (%)')
title('Specificity of image classification by TRN, BCD and DCNN')
xlim([0.50 3.50])
ylim([0 100])
hold off
%% 
% F_measure

data = {F_measure_TRN,F_measure_BCD,F_measure_DCNN};
catIdx = [T_ALL.CL3];
figure
plotSpread(data,'categoryIdx',catIdx,... 
    'categoryMarkers',{'.','d','x'},'categoryColors',{'k','k','k'})
hold on
boxplot(F_measure_ALL,T_ALL.CL3,'Labels',{'TRN','BCD','DCNN'},'Colors','k','OutlierSize',2)
ylabel('F-measure (%)')
title('F-measure of image classification by TRN, BCD and DCNN')
xlim([0.50 3.50])
ylim([0 100])
hold off
%% ROC曲線上にTRN,BCDの感度・特異度をプロット
%%
figure
hold on
plot(Recall_TRN/100,Specificity_TRN/100,'.')

plot(Recall_BCD/100,Specificity_BCD/100,'d')