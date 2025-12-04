
#################### 导入需要的python库 ####################
## run start ##
import warnings
warnings.filterwarnings("ignore")
import os
import pickle
from scipy.stats import norm
import numpy as np
import pandas as pd
from tableone import TableOne
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from missforest import MissForest
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import shap
## run end ##

####################### 数据导入 #######################
os.getcwd() # 当前工作路径
# 导入数据import os
data = pd.read_csv("data/raw/diabetes_raw.csv", encoding="GBK")
# 查看数据情况
print(data.head())  # 查看前五行数据
print(data.info())  # 查看数据结构

#####################################################################################
################################# 数据理解和探索 #####################################
#####################################################################################
#################### 查看变量分布情况 ####################
## 统计分类变量diabetes的频数和频率 ##
data['diabetes'].value_counts() # 频数
data['diabetes'].value_counts(normalize=True) # 频率

## 查看连续变量glucose的直方图和箱线图 ##
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # 创建1行2列的画布
axes[0].hist(data['glucose'], bins=20, edgecolor='black', alpha=0.7) # 绘制 glucose（空腹血糖）的直方图
axes[0].set_title("Histogram of Glucose") # 直方图标题
axes[0].set_xlabel("Glucose") # x轴标签
axes[0].set_ylabel("Frequency") # y轴标签

axes[1].boxplot(data['glucose'], vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue")) # 绘制glucose（空腹血糖）的箱线图
axes[1].set_title("Boxplot of Glucose") # 箱线图标题
axes[1].set_ylabel("Glucose") # y轴标签

plt.tight_layout() # 调整画布布局，避免边界内容显示不全
##  plt.savefig("glucose的直方图和箱线图.png", dpi=300)
plt.show() # 显示图片

#####################################################################################
#################################### 数据清洗 ########################################
#####################################################################################
######################## 1. 处理变量glucose中的异常值 #####################
# 计算glucose变量的IQR（四分位间距）
Q1 = data['glucose'].quantile(0.25)  # 第一四分位数（25%分位数）
Q3 = data['glucose'].quantile(0.75)  # 第三四分位数（75%分位数）
IQR = Q3 - Q1  # 计算IQR
# 定义异常值的阈值（1.5倍IQR规则，可以自行调整） 
lower_bound = Q1 - 1.5 * IQR # 判断的下界值
upper_bound = Q3 + 1.5 * IQR # 判断的上界值
# 找出异常值的索引，分析可能原因
outliers_index = data[(data['glucose'] < lower_bound) | (data['glucose'] > upper_bound)].index # 异常值所在的行
outliers_index
data.loc[outliers_index,] # 查看异常值所在的行
# 分析原因，修改异常值：去掉最后一位数字
data.loc[outliers_index, 'glucose'] = [118,99]
data.loc[outliers_index, ]

######################### 2. 处理变量insulin中的缺失值 ##########################
# 检查数据集中缺失值的情况
missing_values = data.isnull().sum() # 各变量的缺失数
missing_values

# 用MissForest填补变量insulin中的缺失值
imputer = MissForest() # 创建 MissForest 插补器对象
data_imputed = imputer.fit_transform(data) # # 对包含缺失值的数据 data 进行拟合并填补，返回填补后的新数据 data_imputed

# 注意整数变量的填补值也要保留整数
data_imputed['insulin'] = data_imputed['insulin'].round()

# 查看填补后数据的缺失情况
data_imputed.isnull().sum()

# 保存填补后的数据
data_imputed.to_csv("data/processed/diabetes_imputed.csv", index=False)

#####################################################################################
#################################### 数据拆分 ########################################
#####################################################################################
##################### 1. 数据集拆分：训练集和测试集 ######################
data = pd.read_csv("data/processed/diabetes_imputed.csv", encoding="GBK") # 读取填补后的数据
train_data, test_data = train_test_split(data, test_size=0.3, # 拆分数据
                                         stratify=data["diabetes"], random_state=2025) # https://scikit-learn.org.cn/view/649.html

# 保存这两个数据集
train_data.to_csv("data/processed/train_data_notscaled.csv") # 保存训练集
test_data.to_csv("data/processed/test_data_notscaled.csv") # 保存测试集

######################## 2. 查看训练集vs测试集的变量均衡性 ########################
# 为两个数据集都生成一个group变量，便于后续变量均衡性检查
train_data["group"] = "train_set"
test_data["group"] = "test_set"
# 合并这两个数据集
total = pd.concat([train_data, test_data]) # 默认按行合并
# 创建描述性统计表
categorical_vars = ["diabetes", "gender", "exercise", "race", "his", "hyperlip", "pregnant"] # 分类变量的变量名
all_vars = total.columns.values[0:len(total.columns)-1].tolist() # 除了'group'变量外的所有变量的变量名
varbalance_table = TableOne(data=total, columns=all_vars, 
                            categorical=categorical_vars, groupby="group", pval=True) # 以group为分组，创建描述性统计表
# 查看变量均衡情况
varbalance_table
# 保存为csv文件，方便查看和粘贴
varbalance_table.to_csv("results/tables/varbalance_table.csv", encoding="utf-8-sig")


#####################################################################################
#################################### 特征工程 ########################################
#####################################################################################
################ 1. 连续型变量标准化，后续加快机器学习模型收敛 #################
## 首先训练集连续变量的标准化
train_data = train_data.drop(columns='group')
train_data
continuous_vars = ['age', 'glucose', 'pressure', 'triceps', 'bmi', 'pedigree', 'insulin'] # 连续变量的变量名
scaler = StandardScaler()  # 创建 scaler 并在训练集 fit，测试集标准化也用此 scaler
train_data[continuous_vars] = scaler.fit_transform(train_data[continuous_vars]) # 对训练集中的连续变量进行标准化
train_data
# 保存标准化后的训练集
train_data.to_csv("data/processed/train_data_scaled.csv")
## 接下来测试集连续变量的标准化，用同一个 scaler 标准化测试集（只能 transform，不要再次fit）
test_data = test_data.drop(columns='group')
test_data[continuous_vars] = scaler.transform(test_data[continuous_vars])
# 保存标准化后的测试集
test_data.to_csv("data/processed/test_data_scaled.csv")

##################### 2. 基于训练集进行变量筛选 ######################
#### 第一种方式：先单因素Logistic筛选，再多因素Logistic筛选 ####
## 逐个自变量进行单因素 Logistic 回归 ##
train_data = pd.read_csv("data/processed/train_data_notscaled.csv",index_col=0) # 后续Logistic预测模型可能需要绘制列线图等，用未标准化的数据
all_vars = train_data.columns.values.tolist() # 所有变量名
all_vars
independent_vars = all_vars[1:len(all_vars)] # 自变量名
independent_vars

results_univariable = [] # 空的列表，用于存储单因素Logistic回归结果
y_train = train_data['diabetes'] # 结局变量
for var in independent_vars: # 逐个自变量进行单因素 Logistic 回归分析，并保存单因素结果
    print("####### "+var+" ########") # 用于显示进度
    x = sm.add_constant(train_data[var])  # 添加常数项（截距），满足Logistic建模需要
    model = sm.Logit(y_train, x).fit(disp=0)  # 进行单因素 Logistic 回归分析，disp=0 关闭迭代信息
    coef = model.params[var]  # 变量的回归系数值
    p_value = model.pvalues[var]  # 变量的P值
    results_univariable.append({'Variable': var, 'Coefficient': coef, 
                                'P-value': p_value}) # 单因素结果信息保存到列表results_univariable中

results_univariable # 查看单因素分析结果
# 转换为 DataFrame，更好查看
results_univariable_df = pd.DataFrame(results_univariable)
results_univariable_df
# 提取单因素P<0.05的变量名
significant_vars_univ = results_univariable_df[results_univariable_df['P-value'] < 0.05]['Variable'].tolist()
significant_vars_univ

## 单因素分析结果显著的变量纳入多因素 Logistic 回归分析 ##
X_sig = train_data[significant_vars_univ] # 单因素显著的变量的数据
model_multilog = sm.Logit(y_train, X_sig).fit(disp=0) # 拟合多因素 Logistic 回归模型
# 多因素分析结果的表格
results_mulvariable_df = pd.DataFrame({
        "Variable": X_sig.columns,
        "Coefficient": model_multilog.params,
        "Odd Ratio": np.exp(model_multilog.params),
        "P-value": model_multilog.pvalues
    })
results_mulvariable_df # 查看结果
results_mulvariable_df.to_csv("results/tables/results_mulvariable_df.csv",index=False) # 结果保存为csv
significant_vars_multi = results_mulvariable_df[results_mulvariable_df['P-value'] \
                                                < 0.05]['Variable'].tolist() # 多因素分析显著的变量名
significant_vars_multi # ['exercise', 'hyperlip', 'pregnant', 'glucose', 'pressure', 'insulin']
with open('models/significant_vars.pkl', 'wb') as f: # 将列名保存为文件，方便之后读取
    pickle.dump(significant_vars_multi, f)


##### 第二种方式：LASSO 回归直接筛选变量（根据变量系数是否为0） #####
# 训练 LASSO 模型进行变量选择
lasso = Lasso(alpha=0.001)  # 创建 Lasso 模型，设置 alpha=0.001，表示正则化强度较小
lasso = Lasso(alpha=0.1) # alpha越大表示惩罚力度越大，自变量系数越有可能被压缩至0，可按需要设置
X_train = train_data[independent_vars] # 训练数据集
lasso.fit(X_train, y_train) # 在训练集上拟合 Lasso 模型
# 提取非零系数对应的变量
lasso.coef_
selected_variables = np.array(independent_vars)[lasso.coef_ != 0].tolist()
selected_variables


###################################################################################################################
########### 基于训练集构建预测模型并基于验证集调优模型超参数（测试集数据应当只参与外部验证，其他地方不能用到） #############
###################################################################################################################
############################# 1. Logistic模型 ##############################
########### 线性模型不涉及超参数调优，我们用训练集训练模型后内部验证 ############
# 我们这里用多因素Logistic回归分析P值显著的变量训练模型 #
with open('models/significant_vars.pkl', 'rb') as f:
    significant_vars_multi = pickle.load(f)
significant_vars_multi # 多因素分析显著的变量名
train_data = pd.read_csv("data/processed/train_data_notscaled.csv", encoding="GBK",index_col=0) # 训练集数据
# 构建预测模型 
X_train = train_data[significant_vars_multi] # 提取训练集中显著的自变量
X_train_const = sm.add_constant(X_train) # 训练集添加常数项（截距）
y_train = train_data['diabetes'] # 结局变量
# 训练模型 
logist_model = sm.Logit(y_train, X_train_const).fit(disp=0)  # 关闭迭代信息
logist_model.summary() # 查看模型信息
# 计算训练集AUC，浅看模型效果
y_train_pred_prob_logist = logist_model.predict(X_train_const) # 在训练集预测所有个体的结局发生概率
auc_logist = roc_auc_score(y_train, y_train_pred_prob_logist) # 计算AUC
auc_logist
# 保存训练好的Logistic模型
with open("models/logistic_model.pkl", 'wb') as f:
    pickle.dump(logist_model, f)

############################### 2. 决策树模型 ###############################
# 导入标准化的训数据集
train_data_scaled = pd.read_csv("data/processed/train_data_scaled.csv", encoding="GBK",index_col=0)
# 提取自变量和结局变量
X = train_data_scaled.loc[:, train_data_scaled.columns != 'diabetes']
y = train_data_scaled['diabetes'] 
# 2.1. 将原始训练数据集再次划分训练集和验证集（可以理解为内部测试集）（7:3）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
# 2.2. 使用默认参数训练CART决策树
tree_default = DecisionTreeClassifier(random_state=123) # 创建决策树分类模型，使用默认参数
tree_default.fit(X_train, y_train) # 在训练集上拟合决策树模型
# 查看模型的默认参数
print("模型默认参数:", pd.DataFrame.from_dict(tree_default.get_params(),orient='index'))
# 2.3. 计算默认参数模型的验证集AUC
y_val_pred_prob_treed = tree_default.predict_proba(X_val)[:, 1] # 验证集预测结局概率
auc_treed = roc_auc_score(y_val, y_val_pred_prob_treed) # 计算验证集AUC
print("默认参数模型的验证集 AUC:", auc_treed)
## 进行模型超参数的网格搜索调优模型超参数（基于验证集 AUC 最高） ##
# 2.4. 定义超参数搜索范围
param_grid = {
    'max_depth': [3, 5, 10, None], # 树的最大深度
    'min_samples_split': [5, 10, 20], # 节点至少需要多少个样本，才会继续分裂
    'max_features': ['sqrt', None], # 在每次分裂时，决策树可以考虑的最大特征数
    'ccp_alpha': [0.0, 0.01, 0.1] # 剪枝时的复杂度惩罚系数
}
# 2.5. 使用网格搜索同时优化这些超参数
best_auc_tree = 0  # 用于存储，记录最高 AUC
tree_model_best = None # 用于记录最佳决策树模型
best_max_depth = None  # 用于记录最佳 max_depth
best_min_samples_split = None  # 用于记录最佳 min_samples_split
best_max_features = None  # 用于记录最佳 max_features
best_ccp_alpha = None  # 用于记录最佳 ccp_alpha
for max_depth in param_grid['max_depth']:
    for min_samples_split in param_grid['min_samples_split']:
         for max_features in param_grid['max_features']:
              for ccp_alpha in param_grid['ccp_alpha']:
                # 设定决策树模型超参数
                tree_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                                    max_features=max_features, ccp_alpha=ccp_alpha,
                                                    random_state=123)
                # 训练模型
                tree_model.fit(X_train, y_train)
                # 在验证集上计算 AUC
                y_val_pred_prob = tree_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_val_pred_prob)
                # 记录最优参数
                if auc > best_auc_tree:
                    tree_model_best = tree_model
                    best_auc_tree = auc
                    best_max_depth = max_depth
                    best_min_samples_split = min_samples_split
                    best_max_features = max_features
                    best_ccp_alpha = ccp_alpha
# 2.6. 输出最优超参数
print("最佳参数组合: max_depth =", best_max_depth, ", min_samples_split, =", best_min_samples_split, 
        ", max_features, =", best_max_features, ", ccp_alpha, =", best_ccp_alpha)
# 2.7. 验证集AUC对比：参数调优决策树 vs 默认参数决策树
print("默认参数决策树模型的验证集 AUC:", auc_treed)
print("参数调优决策树模型的验证集 AUC:", best_auc_tree)
print("调优模型参数:", pd.DataFrame.from_dict(tree_model_best.get_params(),orient='index'))
# 2.8. 绘制决策树
plt.figure(figsize=(10, 5))
plot_tree(tree_model_best, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.savefig("results/performance/tree_structure.jpg", dpi=500)
plt.show()
# 保存训练好的模型
with open("models/tree_model.pkl", 'wb') as f:
    pickle.dump(tree_model_best, f)

################### 3. 随机森林(RF)模型 ##########################
# 3.1. 使用默认参数训练随机森林模型
rf_model_default = RandomForestClassifier(random_state=123, oob_score=True) # 默认使用100棵数，最大特征数为变量数的开方（这里为4）
rf_model_default.fit(X_train, y_train)
# 查看模型默认参数
print("模型默认参数:", pd.DataFrame.from_dict(rf_model_default.get_params(),orient='index'))
# 3.2. 计算默认参数模型的验证集AUC
y_val_pred_prob_rfd = rf_model_default.predict_proba(X_val)[:, 1]
auc_rfd = roc_auc_score(y_val, y_val_pred_prob_rfd)
print("默认参数模型的验证集 AUC:", auc_rfd)
# 进行超参数网格搜索调优（基于验证集 AUC 最高） #
# 3.3. 定义超参数搜索范围
param_grid = {
    'n_estimators': np.arange(50, 500, 50),  # 树的数量：50 到 450，每次增加 50
    'max_features': list(range(2, round(np.sqrt(X.shape[1])) + 1))  # 每棵树的最大特征使用数：2 到 自变量数
}
# 3.4. 使用网格搜索同时优化这些超参数
best_auc_rf = 0
rf_model_best = None
best_params_rf = {}
# 遍历所有超参数组合
for n_estimators in param_grid['n_estimators']:
    for max_features in param_grid['max_features']:
        # 定义随机森林模型
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_features=max_features,
            random_state=123,
            n_jobs=-1
        )
        # 训练模型
        rf_model.fit(X_train, y_train)
        # 计算验证集AUC
        y_val_pred_prob = rf_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred_prob)
        # 记录最佳参数组合
        if auc > best_auc_rf:
            best_auc_rf = auc
            rf_model_best = rf_model
            best_params_rf = {
                'n_estimators': n_estimators,
                'max_features': max_features
            }
# 3.4. 输出最优超参数
best_ntree = best_params_rf['n_estimators']
best_mtry = best_params_rf['max_features']
print("最佳参数组合: n_estimators =", best_ntree, ", max_features =", best_mtry)
# 3.5. 输出最优超参数组合
print("最佳RF参数组合:", best_params_rf)
print("默认参数RF模型的验证集 AUC:", auc_rfd)
print("参数调优RF模型的验证集 AUC:", best_auc_rf)
print("调优RF模型参数:", pd.DataFrame.from_dict(rf_model_best.get_params(), orient='index'))
# 3.6. 保存训练好的模型
with open("models/rf_model.pkl", 'wb') as f:
    pickle.dump(rf_model_best, f)

###################### 4. Xgboost模型 ##########################
# 4.1. 使用默认参数训练XGBoost
xgb_default = XGBClassifier(random_state=123, use_label_encoder=False, eval_metric='logloss')
xgb_default.fit(X_train, y_train)
# 4.2. 计算默认参数模型的验证集AUC
y_val_pred_prob_xgbd = xgb_default.predict_proba(X_val)[:, 1]
auc_xgbd = roc_auc_score(y_val, y_val_pred_prob_xgbd)
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)
# 4.3. 定义超参数搜索范围
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'max_depth': [3, 5, 10],  # 最大深度
    'n_estimators': [50, 100, 200],  # 弱分类器数量
    'subsample': [0.6, 0.8, 1.0]  # 采样比例
}
# 4.4. 进行网格搜索调优
best_auc_xgb = 0
xgb_model_best = None
best_params_xgb = {}
for learning_rate in param_grid['learning_rate']:
    for max_depth in param_grid['max_depth']:
        for n_estimators in param_grid['n_estimators']:
            for subsample in param_grid['subsample']:
                # 定义XGBoost模型
                xgb_model = XGBClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    random_state=123,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                # 训练模型
                xgb_model.fit(X_train, y_train)
                # 计算验证集AUC
                y_val_pred_prob = xgb_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_val_pred_prob)
                # 记录最佳参数组合
                if auc > best_auc_xgb:
                    best_auc_xgb = auc
                    xgb_model_best = xgb_model
                    best_params_xgb = {
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'n_estimators': n_estimators,
                        'subsample': subsample
                    }
# 4.5. 输出最优超参数组合
print("最佳XGBoost参数组合:", best_params_xgb)
print("默认参数XGBoost模型的验证集 AUC:", auc_xgbd)
print("参数调优XGBoost模型的验证集 AUC:", best_auc_xgb)
print("调优XGBoost模型参数:", pd.DataFrame.from_dict(xgb_model_best.get_params(), orient='index'))
# 4.6. 保存训练好的模型
with open("models/xgb_model.pkl", 'wb') as f:
    pickle.dump(xgb_model_best, f)

###################### 5. LightGBM模型 ##########################
# 5.1. 使用默认参数训练LightGBM
lgb_default = lgb.LGBMClassifier(random_state=123)
lgb_default.fit(X_train, y_train)
# 5.2. 计算默认参数模型的验证集AUC
y_val_pred_prob_lgbd = lgb_default.predict_proba(X_val)[:, 1]
auc_lgbd = roc_auc_score(y_val, y_val_pred_prob_lgbd)
print("默认参数LightGBM模型的验证集 AUC:", auc_lgbd)
# 5.3. 定义超参数搜索范围
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'num_leaves': [31, 50, 100],  # 叶子节点数
    'n_estimators': [50, 100, 200],  # 弱分类器数量
    'subsample': [0.6, 0.8, 1.0],  # 采样比例
    'colsample_bytree': [0.6, 0.8, 1.0]  # 特征子集采样比例
}
# 5.4. 进行网格搜索调优
best_auc_lgb = 0
lgb_model_best = None
best_params_lgb = {}
for learning_rate in param_grid['learning_rate']:
    for num_leaves in param_grid['num_leaves']:
        for n_estimators in param_grid['n_estimators']:
            for subsample in param_grid['subsample']:
                for colsample_bytree in param_grid['colsample_bytree']:
                    # 定义LightGBM模型
                    lgb_model = lgb.LGBMClassifier(
                        learning_rate=learning_rate,
                        num_leaves=num_leaves,
                        n_estimators=n_estimators,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=123
                    )
                    
                    # 训练模型
                    lgb_model.fit(X_train, y_train)
                    
                    # 计算验证集AUC
                    y_val_pred_prob = lgb_model.predict_proba(X_val)[:, 1]
                    auc = roc_auc_score(y_val, y_val_pred_prob)
                    
                    # 记录最佳参数组合
                    if auc > best_auc_lgb:
                        best_auc_lgb = auc
                        lgb_model_best = lgb_model
                        best_params_lgb = {
                            'learning_rate': learning_rate,
                            'num_leaves': num_leaves,
                            'n_estimators': n_estimators,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree
                        }
# 5.5. 输出最优超参数组合
print("最佳LightGBM参数组合:", best_params_lgb)
print("默认参数LightGBM模型的验证集 AUC:", auc_lgbd)
print("参数调优LightGBM模型的验证集 AUC:", best_auc_lgb)
print("调优LightGBM模型参数:", pd.DataFrame.from_dict(lgb_model_best.get_params(), orient='index'))
# 5.6. 保存训练好的模型
with open("models/lgb_model.pkl", 'wb') as f:
    pickle.dump(lgb_model_best, f)

###################### 6. 支持向量机（SVM）模型 ##########################
# 6.1. 使用默认参数训练SVM
svm_default = SVC(probability=True, random_state=123)
svm_default.fit(X_train, y_train)
# 6.2. 计算默认参数模型的验证集AUC
y_val_pred_prob_svmd = svm_default.predict_proba(X_val)[:, 1]
auc_svmd = roc_auc_score(y_val, y_val_pred_prob_svmd)
print("默认参数SVM模型的验证集 AUC:", auc_svmd)
# 6.3. 定义超参数搜索范围
param_grid = {
    'C': [0.1, 2, 10],  # 正则化参数
    'kernel': ['linear', 'rbf', 'poly'],  # 核函数类型
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # 核函数系数（适用于 'rbf' 和 'poly'）
    'degree': [2, 3, 4]  # 多项式核的阶数（仅适用于 'poly' 核）
}
# 6.4. 进行网格搜索调优
best_auc_svm = 0
svm_model_best = None
best_params_svm = {}
for C in param_grid['C']:
    for kernel in param_grid['kernel']:
        for gamma in param_grid['gamma']:
            for degree in param_grid['degree'] if kernel == 'poly' else [None]:
                # 定义SVM模型
                svm_model = SVC(
                    C=C,
                    kernel=kernel,
                    gamma=gamma if kernel in ['rbf', 'poly'] else 'scale',
                    degree=degree if kernel == 'poly' else 3,
                    probability=True,
                    random_state=123
                )
                # 训练模型
                svm_model.fit(X_train, y_train)
                # 计算验证集AUC
                y_val_pred_prob = svm_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_val_pred_prob)
                # 记录最佳参数组合
                if auc > best_auc_svm:
                    best_auc_svm = auc
                    svm_model_best = svm_model
                    best_params_svm = {
                        'C': C,
                        'kernel': kernel,
                        'gamma': gamma,
                        'degree': degree if kernel == 'poly' else None
                    }
# 6.5. 输出最优超参数组合
print("最佳SVM参数组合:", best_params_svm)
print("默认参数SVM模型的验证集 AUC:", auc_svmd)
print("参数调优SVM模型的验证集 AUC:", best_auc_svm)
print("默认SVM 和 调优SVM 的参数:", 
      pd.concat(
          [pd.DataFrame.from_dict(svm_default.get_params(), orient='index'),
           pd.DataFrame.from_dict(svm_model_best.get_params(), orient='index')],
          axis=1))
# 6.6. 保存训练好的模型
with open("models/svm_model.pkl", 'wb') as f:
    pickle.dump(svm_model_best, f)

###################### 7. 人工神经网络（ANN）模型 ##########################
# 7.1. 使用默认参数训练ANN
ann_default = MLPClassifier(random_state=123, max_iter=500)
ann_default.fit(X_train, y_train)
# 7.2. 计算默认参数模型的验证集AUC
y_val_pred_prob_annd = ann_default.predict_proba(X_val)[:, 1]
auc_annd = roc_auc_score(y_val, y_val_pred_prob_annd)
print("默认参数ANN模型的验证集 AUC:", auc_annd)
# 7.3. 定义超参数搜索范围
param_grid = {
    'hidden_layer_sizes': [(25,), (50,), (100,), (10, 10), (50, 50), (100,100), (50, 50, 50)],  # 隐藏层神经元数
    'activation': ['relu', 'tanh', 'logistic']  # 激活函数
}
# 7.4. 进行网格搜索调优
best_auc_ann = 0
ann_model_best = None
best_params_ann = {}
for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
    for activation in param_grid['activation']:
        # 定义ANN模型
        ann_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            random_state=123,
            max_iter=500
        )
        # 训练模型
        ann_model.fit(X_train, y_train)
        # 计算验证集AUC
        y_val_pred_prob = ann_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred_prob)
        # 记录最佳参数组合
        if auc > best_auc_ann:
            best_auc_ann = auc
            ann_model_best = ann_model
            best_params_ann = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation
            }
# 7.5. 输出最优超参数组合
print("最佳ANN参数组合:", best_params_ann)
print("默认参数ANN模型的验证集 AUC:", auc_annd)
print("参数调优ANN模型的验证集 AUC:", best_auc_ann)
print("默认ANN 和 调优ANN 的参数:", 
      pd.concat(
          [pd.DataFrame.from_dict(ann_default.get_params(), orient='index'),
           pd.DataFrame.from_dict(ann_model_best.get_params(), orient='index')],
          axis=1))
# 7.6. 保存训练好的模型
with open("models/ann_model.pkl", 'wb') as f:
    pickle.dump(ann_model_best, f)

################################################################################################
################################## 验证数据集评价模型预测效果 ####################################
################################################################################################
## run start ##
# 得到训练集数据（对于Logistic模型）
with open('models/significant_vars.pkl', 'rb') as f:
    significant_vars_multi = pickle.load(f)
significant_vars_multi
train_data = pd.read_csv("data/processed/train_data_notscaled.csv", encoding="GBK",index_col=0)
X_train_logist = train_data[significant_vars_multi]
X_train_logist_const = sm.add_constant(X_train_logist)
y_train_logist = train_data['diabetes']
# 得到验证数据集（对于机器学习模型）
train_data_scaled = pd.read_csv("data/processed/train_data_scaled.csv", encoding="GBK",index_col=0)
X = train_data_scaled.loc[:, train_data_scaled.columns != 'diabetes']
y = train_data_scaled['diabetes'] 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, 
                                                  random_state=123, stratify=y) # 数据拆分时种子数必须和之前训练时的相同
###################### 1. 加载训练好的模型 ##########################
# 1.1. Logistic模型
with open("models/logistic_model.pkl", 'rb') as f:
    logist_model = pickle.load(f)
# 1.2. 决策树模型
with open("models/tree_model.pkl", 'rb') as f:
    tree_model = pickle.load(f)
# 1.3. 随机森林模型
with open("models/rf_model.pkl", 'rb') as f:
    rf_model = pickle.load(f)
# 1.4. XGBoost模型
with open("models/xgb_model.pkl", 'rb') as f:
    xgb_model = pickle.load(f)
# 1.5. LightGBM模型
with open("models/lgb_model.pkl", 'rb') as f:
    lgb_model = pickle.load(f)
# 1.6. SVM模型
with open("models/svm_model.pkl", 'rb') as f:
    svm_model = pickle.load(f)
# 1.7. ANN模型
with open("models/ann_model.pkl", 'rb') as f:
    ann_model = pickle.load(f)
## run end ##

###################### 2. 得到验证数据集预测结果，包括预测概率和预测分类 ##########################
# 2.1. Logistic模型（该模型为训练集评价，因为不涉及调参，也就没有验证集）
y_train_pred_prob_logist = logist_model.predict(X_train_logist_const) # 预测概率
y_train_pred_logist = (y_train_pred_prob_logist >= 0.5).astype(int) # 预测分类值（阈值0.5）
# 1.2. 决策树模型
y_val_pred_prob_tree = tree_model.predict_proba(X_val)[:, 1]
y_val_pred_tree = (y_val_pred_prob_tree >= 0.5).astype(int)
# 2.3. 随机森林模型
y_val_pred_prob_rf = rf_model.predict_proba(X_val)[:, 1]
y_val_pred_rf = (y_val_pred_prob_rf >= 0.5).astype(int)
# 2.4. XGBoost模型
y_val_pred_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred_xgb = (y_val_pred_prob_xgb >= 0.5).astype(int)
# 2.5. LightGBM模型
y_val_pred_prob_lgb = lgb_model.predict_proba(X_val)[:, 1]
y_val_pred_lgb = (y_val_pred_prob_lgb >= 0.5).astype(int)
# 2.6. SVM模型
y_val_pred_prob_svm = svm_model.predict_proba(X_val)[:, 1]
y_val_pred_svm = (y_val_pred_prob_svm >= 0.5).astype(int)
# 2.7. ANN模型
y_val_pred_prob_ann = ann_model.predict_proba(X_val)[:, 1]
y_val_pred_ann = (y_val_pred_prob_ann >= 0.5).astype(int)

###################### 3. 计算混淆矩阵并可视化 ##########################
## 编写混淆矩阵可视化函数，方便调用 ##
def CM_plot(cm):
    plt.figure(figsize=(5, 4)) # 可视化
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
# 3.1. Logistic模型
cm_logist = confusion_matrix(y_train_logist, y_train_pred_logist) # 混淆矩阵
cm_logist
CM_plot(cm_logist)
# 3.2. 决策树模型
cm_tree = confusion_matrix(y_val, y_val_pred_tree)
CM_plot(cm_tree)
# 3.3. 随机森林模型
cm_rf = confusion_matrix(y_val, y_val_pred_rf)
CM_plot(cm_rf)
# 3.4. XGBoost模型
cm_xgb = confusion_matrix(y_val, y_val_pred_xgb)
CM_plot(cm_xgb)
# 3.5. LightGBM模型
cm_lgb = confusion_matrix(y_val, y_val_pred_lgb)
CM_plot(cm_lgb)
# 3.6. SVM模型
cm_svm = confusion_matrix(y_val, y_val_pred_svm)
CM_plot(cm_svm)
# 3.7. ANN模型
cm_ann = confusion_matrix(y_val, y_val_pred_ann)
CM_plot(cm_ann)

###################### 4. 计算准确率、精确率、灵敏度、f1分数、特异度 ##########################
def calculate_acc_pre_sen_f1_spc(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    accuracy = (tp + tn) / (tp + fn + tn + fp) # 计算准确率（Accuracy rate）
    precision = tp / (tp + fp) # 计算精确率（Precision）
    sensitivity = tp / (tp + fn) # 计算灵敏度（Sensitivity, 召回率）
    f1score = 2 * (precision * sensitivity) / (precision + sensitivity)
    specificity = tn / (tn + fp) # 计算特异度（Specificity）
    return accuracy, precision, sensitivity, f1score, specificity
# 4.1. Logistic模型
accuracy_logist, precision_logist, sensitivity_logist, f1_logist, specificity_logist = calculate_acc_pre_sen_f1_spc(cm_logist)
print(f"Logistic Model → Accuracy: {accuracy_logist:.3f}, Precision: {precision_logist:.3f}, Sensitivity: {sensitivity_logist:.3f}, F1 Score: {f1_logist:.3f}, Specificity: {specificity_logist:.3f}")
# 4.2. 决策树模型
accuracy_tree, precision_tree, sensitivity_tree, f1_tree, specificity_tree = calculate_acc_pre_sen_f1_spc(cm_tree)
print(f"Decision Tree Model → Accuracy: {accuracy_tree:.3f}, Precision: {precision_tree:.3f}, Sensitivity: {sensitivity_tree:.3f}, F1 Score: {f1_tree:.3f}, Specificity: {specificity_tree:.3f}")
# 4.3. 随机森林模型
accuracy_rf, precision_rf, sensitivity_rf, f1_rf, specificity_rf = calculate_acc_pre_sen_f1_spc(cm_rf)
print(f"Random Forest Model → Accuracy: {accuracy_rf:.3f}, Precision: {precision_rf:.3f}, Sensitivity: {sensitivity_rf:.3f}, F1 Score: {f1_rf:.3f}, Specificity: {specificity_rf:.3f}")
# 4.4. XGBoost模型
accuracy_xgb, precision_xgb, sensitivity_xgb, f1_xgb, specificity_xgb = calculate_acc_pre_sen_f1_spc(cm_xgb)
print(f"XGBoost Model → Accuracy: {accuracy_xgb:.3f}, Precision: {precision_xgb:.3f}, Sensitivity: {sensitivity_xgb:.3f}, F1 Score: {f1_xgb:.3f}, Specificity: {specificity_xgb:.3f}")
# 4.5. LightGBM模型
accuracy_lgb, precision_lgb, sensitivity_lgb, f1_lgb, specificity_lgb = calculate_acc_pre_sen_f1_spc(cm_lgb)
print(f"LightGBM Model → Accuracy: {accuracy_lgb:.3f}, Precision: {precision_lgb:.3f}, Sensitivity: {sensitivity_lgb:.3f}, F1 Score: {f1_lgb:.3f}, Specificity: {specificity_lgb:.3f}")
# 4.6. SVM模型
accuracy_svm, precision_svm, sensitivity_svm, f1_svm, specificity_svm = calculate_acc_pre_sen_f1_spc(cm_svm)
print(f"SVM Model → Accuracy: {accuracy_svm:.3f}, Precision: {precision_svm:.3f}, Sensitivity: {sensitivity_svm:.3f}, F1 Score: {f1_svm:.3f}, Specificity: {specificity_svm:.3f}")
# 4.7. ANN模型
accuracy_ann, precision_ann, sensitivity_ann, f1_ann, specificity_ann = calculate_acc_pre_sen_f1_spc(cm_ann)
print(f"ANN Model → Accuracy: {accuracy_ann:.3f}, Precision: {precision_ann:.3f}, Sensitivity: {sensitivity_ann:.3f}, F1 Score: {f1_ann:.3f}, Specificity: {specificity_ann:.3f}")


###################### 5. 计算AUC及其95%置信区间 ##########################
## 编写AUC及其95%置信区间计算的函数 ##
def calculate_auc(y_label, y_pred_prob):
    auc_value = roc_auc_score(y_label, y_pred_prob)
    se_auc = np.sqrt((auc_value * (1 - auc_value)) / len(y_label))
    z = norm.ppf(0.975)  # 95% CI 的z值
    auc_ci_lower = auc_value - z * se_auc
    auc_ci_upper = auc_value + z * se_auc
    return auc_value, auc_ci_lower, auc_ci_upper
# 5.1. Logistic模型
auc_value_logist, auc_ci_lower_logist, auc_ci_upper_logist = calculate_auc(y_train_logist, y_train_pred_prob_logist)
print(f"Logistic Model AUC: {auc_value_logist:.3f} (95% CI: {auc_ci_lower_logist:.3f} - {auc_ci_upper_logist:.3f})")
# 5.2. 决策树模型
auc_value_tree, auc_ci_lower_tree, auc_ci_upper_tree = calculate_auc(y_val, y_val_pred_prob_tree)
print(f"Decision Tree Model AUC: {auc_value_tree:.3f} (95% CI: {auc_ci_lower_tree:.3f} - {auc_ci_upper_tree:.3f})")
# 5.3. 随机森林模型
auc_value_rf, auc_ci_lower_rf, auc_ci_upper_rf = calculate_auc(y_val, y_val_pred_prob_rf)
print(f"Random Forest Model AUC: {auc_value_rf:.3f} (95% CI: {auc_ci_lower_rf:.3f} - {auc_ci_upper_rf:.3f})")
# 5.4. XGBoost模型
auc_value_xgb, auc_ci_lower_xgb, auc_ci_upper_xgb = calculate_auc(y_val, y_val_pred_prob_xgb)
print(f"XGBoost Model AUC: {auc_value_xgb:.3f} (95% CI: {auc_ci_lower_xgb:.3f} - {auc_ci_upper_xgb:.3f})")
# 5.5. LightGBM模型
auc_value_lgb, auc_ci_lower_lgb, auc_ci_upper_lgb = calculate_auc(y_val, y_val_pred_prob_lgb)
print(f"LightGBM Model AUC: {auc_value_lgb:.3f} (95% CI: {auc_ci_lower_lgb:.3f} - {auc_ci_upper_lgb:.3f})")
# 5.6. SVM模型
auc_value_svm, auc_ci_lower_svm, auc_ci_upper_svm = calculate_auc(y_val, y_val_pred_prob_svm)
print(f"SVM Model AUC: {auc_value_svm:.3f} (95% CI: {auc_ci_lower_svm:.3f} - {auc_ci_upper_svm:.3f})")
# 5.7. ANN模型
auc_value_ann, auc_ci_lower_ann, auc_ci_upper_ann = calculate_auc(y_val, y_val_pred_prob_ann)
print(f"ANN Model AUC: {auc_value_ann:.3f} (95% CI: {auc_ci_lower_ann:.3f} - {auc_ci_upper_ann:.3f})")

###################### 6. 绘制ROC曲线 ##########################
## 编写绘制ROC曲线的函数 ##
def ROC_plot(y_label, y_pred_prob,auc_value):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.show()
# 6.1. Logistic模型
ROC_plot(y_train_logist, y_train_pred_prob_logist, auc_value_logist)
# 6.2. 决策树模型
ROC_plot(y_val, y_val_pred_prob_tree, auc_value_tree)
# 6.3. 随机森林模型
ROC_plot(y_val, y_val_pred_prob_rf, auc_value_rf)
# 6.4. XGBoost模型
ROC_plot(y_val, y_val_pred_prob_xgb, auc_value_xgb)
# 6.5. LightGBM模型
ROC_plot(y_val, y_val_pred_prob_lgb, auc_value_lgb)
# 6.6. SVM模型
ROC_plot(y_val, y_val_pred_prob_svm, auc_value_svm)
# 6.7. ANN模型
ROC_plot(y_val, y_val_pred_prob_ann, auc_value_ann)

###################### 7. 绘制校准曲线 ##########################
## 编写绘制校准曲线的函数 ##
def CaliC_plot(y_label, y_pred_prob,n_bins=10):
    prob_true, prob_pred = calibration_curve(y_label, y_pred_prob, n_bins=n_bins)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()
# 7.1. Logistic模型
CaliC_plot(y_train_logist, y_train_pred_prob_logist)
CaliC_plot(y_train_logist, y_train_pred_prob_logist, n_bins=8)
# 7.2. 决策树模型
CaliC_plot(y_val, y_val_pred_prob_tree) # 出现原因是数据异质性不高
# 7.3. 随机森林模型
CaliC_plot(y_val, y_val_pred_prob_rf) # 预测概率高的人少+数据异质性不高，人数多就好了
# 7.4. XGBoost模型
CaliC_plot(y_val, y_val_pred_prob_xgb)
# 7.5. LightGBM模型
CaliC_plot(y_val, y_val_pred_prob_lgb)
# 7.6. SVM模型
CaliC_plot(y_val, y_val_pred_prob_svm)
# 7.7. ANN模型
CaliC_plot(y_val, y_val_pred_prob_ann)

###################### 8. 绘制决策分析曲线 (DCA) ##########################
## 编写计算净收益的函数，方便调用 ##
def calculate_net_benefi(y_label, y_pred_prob,
                                thresholds = np.linspace(0.01, 1, 100)):
    net_benefit_model = [] # 用于保存在不同阳性阈值下基于模型的净收益
    net_benefit_alltrt = [] # 用于保存在不同阳性阈值下假定所有人都接受治疗时的净收益
    net_benefits_notrt = [0] * len(thresholds) # 假定所有人都不接受治疗时的净收益，始终为0
    total_obs = len(y_label)
    for thresh in thresholds:
        # 对于基于模型的净收益
        y_pred_label = y_pred_prob > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        net_benefit = (tp / total_obs) - (fp / total_obs) * (thresh / (1 - thresh))
        net_benefit_model.append(net_benefit)
        # 对于假定所有人都接受治疗时的净收益
        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total_right = tp + tn
        net_benefit = (tp / total_right) - (tn / total_right) * (thresh / (1 - thresh))
        net_benefit_alltrt.append(net_benefit)
    return net_benefit_model, net_benefit_alltrt, net_benefits_notrt
## 编写绘制DCA曲线的函数，方便调用 ##
def DCA_plot(net_benefit_model, net_benefit_alltrt, net_benefits_notrt,
             thresholds = np.linspace(0.01,0.99,100)):
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, net_benefit_model, label="Model Net Benefit", color='blue', linewidth=2)
    plt.plot(thresholds, net_benefit_alltrt, label="Treat All", color="red", linewidth=2)
    plt.plot(thresholds, net_benefits_notrt, linestyle='--', color='green', label="Treat None", linewidth=2)
    plt.xlabel("Threshold Probability")
    plt.ylim(-0.15,np.nanmax(np.array(net_benefit_model))+0.1)
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()
# 8.1. Logistic模型
net_benefit_logist, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_train_logist, y_train_pred_prob_logist)
DCA_plot(net_benefit_logist, net_benefit_alltrt, net_benefits_notrt)
# 8.2. 决策树模型
net_benefit_tree, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_tree)
DCA_plot(net_benefit_tree, net_benefit_alltrt, net_benefits_notrt)
# 8.3. 随机森林模型
net_benefit_rf, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_rf)
DCA_plot(net_benefit_rf, net_benefit_alltrt, net_benefits_notrt)
# 8.4. XGBoost模型
net_benefit_xgb, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_xgb)
DCA_plot(net_benefit_xgb, net_benefit_alltrt, net_benefits_notrt)
# 8.5. LightGBM模型
net_benefit_lgb, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_lgb)
DCA_plot(net_benefit_lgb, net_benefit_alltrt, net_benefits_notrt)
# 8.6. SVM模型
net_benefit_svm, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_svm)
DCA_plot(net_benefit_svm, net_benefit_alltrt, net_benefits_notrt)
# 8.7. ANN模型
net_benefit_ann, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_val, y_val_pred_prob_ann)
DCA_plot(net_benefit_ann, net_benefit_alltrt, net_benefits_notrt)

###################### 9. 所有模型的训练集/验证集预测效果汇总，方便对比 ##########################
# 9.1. 汇总模型评估指标并保存 
model_results_validation = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "SVM", "ANN"],
    "AUC": [auc_value_logist, auc_value_tree, auc_value_rf, auc_value_xgb, auc_value_lgb, auc_value_svm, auc_value_ann],
    "AUC 95% CI Lower": [auc_ci_lower_logist, auc_ci_lower_tree, auc_ci_lower_rf, auc_ci_lower_xgb, auc_ci_lower_lgb, auc_ci_lower_svm, auc_ci_lower_ann],
    "AUC 95% CI Upper": [auc_ci_upper_logist, auc_ci_upper_tree, auc_ci_upper_rf, auc_ci_upper_xgb, auc_ci_upper_lgb, auc_ci_upper_svm, auc_ci_upper_ann],
    "Accuracy": [accuracy_logist, accuracy_tree, accuracy_rf, accuracy_xgb, accuracy_lgb, accuracy_svm, accuracy_ann],
    "Precision": [precision_logist, precision_tree, precision_rf, precision_xgb, precision_lgb, precision_svm, precision_ann],
    "Sensitivity": [sensitivity_logist, sensitivity_tree, sensitivity_rf, sensitivity_xgb, sensitivity_lgb, sensitivity_svm, sensitivity_ann],
    "Specificity": [specificity_logist, specificity_tree, specificity_rf, specificity_xgb, specificity_lgb, specificity_svm, specificity_ann],
    "F1 Score": [f1_logist, f1_tree, f1_rf, f1_xgb, f1_lgb, f1_svm, f1_ann]
}) # 创建DataFrame存储结果
model_results_validation
model_results_validation.to_csv("results/tables/model_performance_validation.csv", index=False) # 保存为CSV文件
# 9.2. 在一张图上绘制所有模型的ROC曲线 #
plt.figure(figsize=(8, 6))
models = {
    "Logistic": (y_train_logist, y_train_pred_prob_logist, auc_value_logist),
    "Decision Tree": (y_val, y_val_pred_prob_tree, auc_value_tree),
    "Random Forest": (y_val, y_val_pred_prob_rf, auc_value_rf),
    "XGBoost": (y_val, y_val_pred_prob_xgb, auc_value_xgb),
    "LightGBM": (y_val, y_val_pred_prob_lgb, auc_value_lgb),
    "SVM": (y_val, y_val_pred_prob_svm, auc_value_svm),
    "ANN": (y_val, y_val_pred_prob_ann, auc_value_ann)
}
for model_name, (y_true, y_pred_prob, auc_value) in models.items():
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f})") # 逐个绘制每个模型的ROC曲线
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend(loc='lower right')
plt.grid()
plt.savefig("results/performance/ROC_curves_allmodel_validation.png", dpi=300)
plt.show()
# 9.3. 在一张图上绘制所有模型的校准曲线 
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=model_name) # 逐个绘制每个模型的校准曲线
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves for All Models")
plt.legend(loc='upper left')
plt.grid()
plt.savefig("results/performance/Calibration_curves_allmodel_validation.png", dpi=300)
plt.show()
# 9.4. 在一张图上绘制所有模型的DCA曲线 
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models.items():
    net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_true, y_pred_prob)
    plt.plot(np.linspace(0.01, 0.99, 100), net_benefit, label=model_name) # 逐个绘制DCA曲线
plt.plot(np.linspace(0.01, 0.99, 100), net_benefit_alltrt, linestyle="--", color="red", label="Treat All")
plt.plot(np.linspace(0.01, 0.99, 100), net_benefits_notrt, linestyle="--", color="green", label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylim(-0.3,np.nanmax(np.array(net_benefit))+0.1)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis for All Models")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("results/performance/DCA_curves_allmodel_validation.png", dpi=300)
plt.show()


################################################################################################
################################## 测试数据集评价模型预测效果 ####################################
################################################################################################
## run start ##
# 得到测试集数据（Logistic模型）
with open('models/significant_vars.pkl', 'rb') as f:
    significant_vars_multi = pickle.load(f)
significant_vars_multi
test_data = pd.read_csv("data/processed/test_data_notscaled.csv", encoding="GBK",index_col=0)
X_test_logist = test_data[significant_vars_multi]
X_test_logist_const = sm.add_constant(X_test_logist)
y_test_logist = test_data['diabetes']
# 得到测试数据集（机器学习模型）
test_data_scaled = pd.read_csv("data/processed/test_data_scaled.csv", encoding="GBK",index_col=0)
X_test = test_data_scaled.loc[:, test_data_scaled.columns != 'diabetes']
y_test = test_data_scaled['diabetes'] 
## run end ##

###################### 1. 计算测试数据集预测结果 ##########################
# 1.1. Logistic模型
y_test_pred_prob_logist = logist_model.predict(X_test_logist_const) # 预测概率
y_test_pred_logist = (y_test_pred_prob_logist >= 0.5).astype(int) # 预测分类值（阈值0.5）
# 1.2. 决策树模型
y_test_pred_prob_tree = tree_model.predict_proba(X_test)[:, 1]
y_test_pred_tree = (y_test_pred_prob_tree >= 0.5).astype(int)
# 1.3. 随机森林模型
y_test_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_test_pred_rf = (y_test_pred_prob_rf >= 0.5).astype(int)
# 1.4. XGBoost模型
y_test_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_test_pred_xgb = (y_test_pred_prob_xgb >= 0.5).astype(int)
# 1.5. LightGBM模型
y_test_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_test_pred_lgb = (y_test_pred_prob_lgb >= 0.5).astype(int)
# 1.6. SVM模型
y_test_pred_prob_svm = svm_model.predict_proba(X_test)[:, 1]
y_test_pred_svm = (y_test_pred_prob_svm >= 0.5).astype(int)
# 1.7. ANN模型
y_test_pred_prob_ann = ann_model.predict_proba(X_test)[:, 1]
y_test_pred_ann = (y_test_pred_prob_ann >= 0.5).astype(int)

###################### 2. 计算混淆矩阵并可视化 ##########################
# 2.1. Logistic模型
cm_logist_test = confusion_matrix(y_test_logist, y_test_pred_logist)
CM_plot(cm_logist_test)
# 2.2. 决策树模型
cm_tree_test = confusion_matrix(y_test, y_test_pred_tree)
CM_plot(cm_tree_test)
# 2.3. 随机森林模型
cm_rf_test = confusion_matrix(y_test, y_test_pred_rf)
CM_plot(cm_rf_test)
# 2.4. XGBoost模型
cm_xgb_test = confusion_matrix(y_test, y_test_pred_xgb)
CM_plot(cm_xgb_test)
# 2.5. LightGBM模型
cm_lgb_test = confusion_matrix(y_test, y_test_pred_lgb)
CM_plot(cm_lgb_test)
# 2.6. SVM模型
cm_svm_test = confusion_matrix(y_test, y_test_pred_svm)
CM_plot(cm_svm_test)
# 单独保存如下
plt.figure(figsize=(5, 4))
sns.heatmap(cm_svm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
os.makedirs("results/performance", exist_ok=True)
plt.savefig("results/performance/cm_svm_test.png", dpi=500, bbox_inches="tight")
plt.show()
# 2.7. ANN模型
cm_ann_test = confusion_matrix(y_test, y_test_pred_ann)
CM_plot(cm_ann_test)
# 单独保存如下
plt.figure(figsize=(5, 4))
sns.heatmap(cm_svm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
os.makedirs("results/performance", exist_ok=True)
plt.savefig("results/performance/cm_ann_test.png", dpi=500, bbox_inches="tight")
plt.show()

###################### 3. 计算准确率、精确率、灵敏度、f1分数、特异度 ##########################
# 3.1. Logistic模型
accuracy_logist_test, precision_logist_test, sensitivity_logist_test, f1_logist_test, specificity_logist_test = calculate_acc_pre_sen_f1_spc(cm_logist_test)
# 3.2. 决策树模型
accuracy_tree_test, precision_tree_test, sensitivity_tree_test, f1_tree_test, specificity_tree_test = calculate_acc_pre_sen_f1_spc(cm_tree_test)
# 3.3. 随机森林模型
accuracy_rf_test, precision_rf_test, sensitivity_rf_test, f1_rf_test, specificity_rf_test = calculate_acc_pre_sen_f1_spc(cm_rf_test)
# 3.4. XGBoost模型
accuracy_xgb_test, precision_xgb_test, sensitivity_xgb_test, f1_xgb_test, specificity_xgb_test = calculate_acc_pre_sen_f1_spc(cm_xgb_test)
# 3.5. LightGBM模型
accuracy_lgb_test, precision_lgb_test, sensitivity_lgb_test, f1_lgb_test, specificity_lgb_test = calculate_acc_pre_sen_f1_spc(cm_lgb_test)
# 3.6. SVM模型
accuracy_svm_test, precision_svm_test, sensitivity_svm_test, f1_svm_test, specificity_svm_test = calculate_acc_pre_sen_f1_spc(cm_svm_test)
# 3.7. ANN模型
accuracy_ann_test, precision_ann_test, sensitivity_ann_test, f1_ann_test, specificity_ann_test = calculate_acc_pre_sen_f1_spc(cm_ann_test)

###################### 4. 计算AUC及其95%置信区间 ##########################
# 4.1. Logistic模型
auc_value_logist_test, auc_ci_lower_logist_test, auc_ci_upper_logist_test = calculate_auc(y_test_logist, y_test_pred_prob_logist)
# 4.2. 决策树模型
auc_value_tree_test, auc_ci_lower_tree_test, auc_ci_upper_tree_test = calculate_auc(y_test, y_test_pred_prob_tree)
# 4.3. 随机森林模型
auc_value_rf_test, auc_ci_lower_rf_test, auc_ci_upper_rf_test = calculate_auc(y_test, y_test_pred_prob_rf)
# 4.4. XGBoost模型
auc_value_xgb_test, auc_ci_lower_xgb_test, auc_ci_upper_xgb_test = calculate_auc(y_test, y_test_pred_prob_xgb)
# 4.5. LightGBM模型
auc_value_lgb_test, auc_ci_lower_lgb_test, auc_ci_upper_lgb_test = calculate_auc(y_test, y_test_pred_prob_lgb)
# 4.6. SVM模型
auc_value_svm_test, auc_ci_lower_svm_test, auc_ci_upper_svm_test = calculate_auc(y_test, y_test_pred_prob_svm)
# 4.7. ANN模型
auc_value_ann_test, auc_ci_lower_ann_test, auc_ci_upper_ann_test = calculate_auc(y_test, y_test_pred_prob_ann)

###################### 5. 所有模型的测试集预测效果汇总，方便对比 ##########################
model_results_test = pd.DataFrame({
    "Model": ["Logistic", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "SVM", "ANN"],
    "AUC": [auc_value_logist_test, auc_value_tree_test, auc_value_rf_test, auc_value_xgb_test, auc_value_lgb_test, auc_value_svm_test, auc_value_ann_test],
    "AUC 95% CI Lower": [auc_ci_lower_logist_test, auc_ci_lower_tree_test, auc_ci_lower_rf_test, auc_ci_lower_xgb_test, auc_ci_lower_lgb_test, auc_ci_lower_svm_test, auc_ci_lower_ann_test],
    "AUC 95% CI Upper": [auc_ci_upper_logist_test, auc_ci_upper_tree_test, auc_ci_upper_rf_test, auc_ci_upper_xgb_test, auc_ci_upper_lgb_test, auc_ci_upper_svm_test, auc_ci_upper_ann_test],
    "Accuracy": [accuracy_logist_test, accuracy_tree_test, accuracy_rf_test, accuracy_xgb_test, accuracy_lgb_test, accuracy_svm_test, accuracy_ann_test],
    "Precision": [precision_logist_test, precision_tree_test, precision_rf_test, precision_xgb_test, precision_lgb_test, precision_svm_test, precision_ann_test],
    "Sensitivity": [sensitivity_logist_test, sensitivity_tree_test, sensitivity_rf_test, sensitivity_xgb_test, sensitivity_lgb_test, sensitivity_svm_test, sensitivity_ann_test],
    "Specificity": [specificity_logist_test, specificity_tree_test, specificity_rf_test, specificity_xgb_test, specificity_lgb_test, specificity_svm_test, specificity_ann_test],
    "F1 Score": [f1_logist_test, f1_tree_test, f1_rf_test, f1_xgb_test, f1_lgb_test, f1_svm_test, f1_ann_test]
})  # 创建DataFrame存储结果
model_results_test
model_results_test.to_csv("results/tables/model_performance_test.csv", index=False)  # 保存为CSV文件

###################### 6. 绘制ROC曲线 ##########################
plt.figure(figsize=(8, 6))
models_test = {
    "Logistic": (y_test_logist, y_test_pred_prob_logist, auc_value_logist_test),
    "Decision Tree": (y_test, y_test_pred_prob_tree, auc_value_tree_test),
    "Random Forest": (y_test, y_test_pred_prob_rf, auc_value_rf_test),
    "XGBoost": (y_test, y_test_pred_prob_xgb, auc_value_xgb_test),
    "LightGBM": (y_test, y_test_pred_prob_lgb, auc_value_lgb_test),
    "SVM": (y_test, y_test_pred_prob_svm, auc_value_svm_test),
    "ANN": (y_test, y_test_pred_prob_ann, auc_value_ann_test)
}
for model_name, (y_true, y_pred_prob, auc_value) in models_test.items():
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models (Test Set)")
plt.legend(loc='lower right')
plt.grid()
plt.savefig("results/performance/ROC_curves_allmodel_test.png", dpi=300)
plt.show()

###################### 7. 绘制校准曲线 ##########################
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models_test.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curves for All Models (Test Set)")
plt.legend(loc='upper left')
plt.grid()
plt.savefig("results/performance/Calibration_curves_allmodel_test.png", dpi=300)
plt.show()

###################### 8. 绘制DCA曲线 ##########################
plt.figure(figsize=(10, 8))
for model_name, (y_true, y_pred_prob, _) in models_test.items():
    net_benefit, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(y_true, y_pred_prob)
    plt.plot(np.linspace(0.01, 0.99, 100), net_benefit, label=model_name)
plt.plot(np.linspace(0.01, 0.99, 100), net_benefit_alltrt, linestyle="--", color="red", label="Treat All")
plt.plot(np.linspace(0.01, 0.99, 100), net_benefits_notrt, linestyle="--", color="green", label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylim(-0.2,np.nanmax(np.array(net_benefit))+0.1)
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis for All Models (Test Set)")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("results/performance/DCA_curves_allmodel_test.png", dpi=300)
plt.show()


################################################################################################
################################## 模型的SHAP解释 ###############################################
################################################################################################
save_path = "results/shap/" # 图片保存路径
## 编写绘制可解释性图片的函数，方便调用 ##
def shap_explain_model(model, X_train, X_test, model_name="logist", n_interpret=50, obs_index=5):
    """
    使用 SHAP 进行模型的可解释性分析，并保存相关图像。
    参数:
    model: 训练好的机器学习模型
    X_train: 训练数据
    X_test: 测试数据（包含常数项）
    model_name: 模型名称（用于保存文件名）
    n_interpret: 用于 SHAP 解释的数据样本量（按照自己需求更改,不能超过测试集的样本量）
    obs_index: 选择哪一条观测用于瀑布图和单个观测的力图 （按照自己需求更改,不能超过n_interpret)
    """
    if model_name == "logist":
        # 1. 设定模型预测函数
        def model_predict(X):
            return model.predict(X)
        # 2. 建立 SHAP 核解释器（KernelExplainer 适用于任何模型）
        explainer = shap.KernelExplainer(model_predict, X_train)
        shap_values = explainer(X_test.iloc[:n_interpret, :])
        # 3. 生成 SHAP 解释性图像并保存 #
        os.makedirs(save_path, exist_ok=True)
        # 3.1 特征重要性条形图（全局 SHAP 重要性）
        plt.figure(figsize=(15, 12))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.savefig(f"{save_path}bar_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 3.2 特征重要性蜂群图
        plt.figure(figsize=(15, 12))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.savefig(f"{save_path}beeswarm_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 3.3 变量相关性散点图
        plt.figure(figsize=(15, 12))
        shap.plots.scatter(shap_values[:, "pressure"], color=shap_values[:, "glucose"], show=False) # 这里的pressure和glucose变量可以换成自己感兴趣的
        plt.savefig(f"{save_path}scatter_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 3.4 瀑布图（展示选定观测的数据解释）
        plt.figure(figsize=(15, 12))
        shap.plots.waterfall(shap_values[obs_index], max_display=15, show=False)
        plt.savefig(f"{save_path}waterfall_{obs_index}_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 3.5 力图（选定观测）
        force_plot_sig = shap.plots.force(shap_values[obs_index, :])
        shap.save_html(f"{save_path}force_plot_sig_{obs_index}_{model_name}.html", force_plot_sig)
        # 3.6 力图（所有观测）
        force_plot_all = shap.plots.force(shap_values)
        shap.save_html(f"{save_path}force_plot_all_{model_name}.html", force_plot_all)

    if model_name != "logist":
        # 1. 建立 SHAP 核解释器（KernelExplainer 适用于任何模型）
        explainer = shap.KernelExplainer(lambda X: model.predict_proba(X), X_train)
        shap_values = explainer(X_test.iloc[:n_interpret, :])
        # 2. 生成 SHAP 解释性图像并保存
        os.makedirs(save_path, exist_ok=True)
        # 2.1 特征重要性条形图（全局 SHAP 重要性）
        plt.figure(figsize=(15, 12))
        shap.plots.bar(shap_values[:,:,1], max_display=15, show=False)
        plt.savefig(f"{save_path}bar_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 2.2 特征重要性蜂群图
        plt.figure(figsize=(15, 12))
        shap.plots.beeswarm(shap_values[:,:,1], max_display=15, show=False)
        plt.savefig(f"{save_path}beeswarm_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 2.3 依赖性图
        plt.figure(figsize=(15, 12))
        shap.plots.scatter(shap_values[:, "age",1], color=shap_values[:, "gender",1], show=False) # 这里的age和gender变量可以换成自己感兴趣的
        plt.savefig(f"{save_path}scatter_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 2.4 瀑布图（展示选定观测的数据解释）
        plt.figure(figsize=(15, 12))
        shap.plots.waterfall(shap_values[obs_index][:,1], max_display=15, show=False)
        plt.savefig(f"{save_path}waterfall_{obs_index}_{model_name}.png", dpi=200, bbox_inches='tight')
        plt.close()
        # 2.5 力图（选定观测）
        force_plot_sig = shap.plots.force(shap_values[obs_index, :, 1])
        shap.save_html(f"{save_path}force_plot_sig_{obs_index}_{model_name}.html", force_plot_sig)
        # 2.6 力图（所有观测）
        force_plot_all = shap.plots.force(shap_values[:, :, 1])
        shap.save_html(f"{save_path}force_plot_all_{model_name}.html", force_plot_all)
    
    print(f"所有 SHAP 解释性图像已保存在 {save_path}")

###################### 1. Logistic模型的可解释性 ##########################    
shap_explain_model(model=logist_model, X_train=X_train_logist_const, X_test=X_test_logist_const, 
                   model_name="logist", n_interpret=50, obs_index=5)    
###################### 2. 决策树模型的可解释性 ##########################
shap_explain_model(model=tree_model, X_train=X_train, X_test=X_test, 
                   model_name="tree", n_interpret=50, obs_index=5)   
###################### 3. 随机森林模型的可解释性 ##########################
shap_explain_model(model=rf_model, X_train=X_train, X_test=X_test, 
                   model_name="rf", n_interpret=50, obs_index=5)  
###################### 4. XGBoost模型的可解释性 ##########################
shap_explain_model(model=xgb_model, X_train=X_train, X_test=X_test, 
                   model_name="xgb", n_interpret=50, obs_index=5)
###################### 5. LightGBM模型的可解释性 ##########################
shap_explain_model(model=lgb_model, X_train=X_train, X_test=X_test, 
                   model_name="lgb", n_interpret=50, obs_index=5)
###################### 6. SVM模型的可解释性 ##########################
shap_explain_model(model=svm_model, X_train=X_train, X_test=X_test, 
                   model_name="svm", n_interpret=50, obs_index=5)
###################### 7. ANN模型的可解释性 ##########################
shap_explain_model(model=ann_model, X_train=X_train, X_test=X_test, 
                   model_name="ann", n_interpret=50, obs_index=5)



################################################################################################
################################## 跨模型的 SHAP 一致性分析 ####################################
################################################################################################

# 本节：在已有模型基础上，做“跨模型”的 SHAP 方向一致性分析 + 雷达图
# 说明：
#   1）统一使用标准化后的全部样本（训练集 + 测试集）的特征；
#   2）纳入的模型：决策树、随机森林、XGBoost、LightGBM、SVM、ANN；
#      Logistic 模型使用的特征集合与其他模型不同，为避免特征不对齐，此处不纳入跨模型一致性分析。

from scipy.stats import spearmanr

###################### 1. 准备用于 SHAP 的数据和模型列表 ##########################

# 读取标准化后的训练集和测试集，并合并（保证和建模时的特征一致）
train_scaled_all = pd.read_csv("data/processed/train_data_scaled.csv", encoding="GBK", index_col=0)
test_scaled_all  = pd.read_csv("data/processed/test_data_scaled.csv",  encoding="GBK", index_col=0)
data_scaled_all  = pd.concat([train_scaled_all, test_scaled_all], axis=0, ignore_index=True)

# 所有自变量（已标准化），用于 SHAP 计算
X_all_shap = data_scaled_all.drop(columns="diabetes")
feature_names = X_all_shap.columns.tolist()

# 需要做“跨模型一致性”的模型（均基于同一套标准化特征 X）
models_for_consistency = {
    "Decision Tree": tree_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "SVM": svm_model,
    "ANN": ann_model
}

# 为了控制 Kernel SHAP 的计算量，从全部样本中抽取一部分用于背景和评估
n_bg   = 200   # 作为背景分布的样本量
n_eval = 300   # 用于计算 SHAP 的样本量（用于 Spearman 相关）

bg_data = X_all_shap.sample(min(n_bg,  X_all_shap.shape[0]), random_state=2025)
eval_data = X_all_shap.sample(min(n_eval, X_all_shap.shape[0]), random_state=2025)

SIGN_TOL = 0.05   # |Spearman rho| < SIGN_TOL 视为方向不确定（记为 0）

rows_sign = []        # 保存每个模型、每个特征的 rho 与 sign
rows_importance = []  # 保存每个模型、每个特征的 mean(|SHAP|)

###################### 2. 逐模型计算 SHAP 值 + 方向信息 ##########################
## 跨模型一致性最好用“同一批样本 + 同一背景”来比较
## 模型本身完全复用前面训练好的 tree / rf / xgb / lgb / svm / ann
## 前面SHAP 只用来画图，并没有保存原始矩阵，且每个模型的背景分布不一
## 做“统一背景 + 统一样本”的跨模型比较，SHAP 这里再算一遍

for m_name, model in models_for_consistency.items():
    print(f"==== 计算 {m_name} 的 SHAP 值并提取方向 ====")

    # 使用 KernelExplainer 计算该模型在 eval_data 上的 SHAP（针对结局 = 1）
    # 注意：lambda 返回 predict_proba 第 2 列，即“患糖尿病”的概率
    explainer = shap.KernelExplainer(lambda X: model.predict_proba(X)[:, 1],
                                     bg_data)
    shap_values = explainer(eval_data)

    # 兼容不同版本的 shap：优先用 .values，没有则直接用返回值
    shap_array = getattr(shap_values, "values", shap_values)
    shap_df = pd.DataFrame(shap_array, columns=feature_names, index=eval_data.index)

    # 2.1. 计算每个变量的 Spearman 相关系数及方向 sign（-1 / 0 / +1）
    for feat in feature_names:
        r, _ = spearmanr(eval_data[feat].values,
                         shap_df[feat].values,
                         nan_policy="omit")
        if np.isnan(r):
            s = 0
        else:
            if abs(r) < SIGN_TOL:
                s = 0
            else:
                s = 1 if r > 0 else -1
        rows_sign.append({
            "model": m_name,
            "feature": feat,
            "spearman_r": r,
            "sign": int(s)
        })

    # 2.2. 该模型下各变量的 mean(|SHAP|)，后续用于“跨模型重要性”
    mean_abs = shap_df.abs().mean(axis=0)
    for feat in feature_names:
        rows_importance.append({
            "model": m_name,
            "feature": feat,
            "mean_abs_shap": mean_abs.loc[feat]
        })

df_shap_signs = pd.DataFrame(rows_sign)
df_shap_importance = pd.DataFrame(rows_importance)

# 保存原始结果，便于后续查看或复现
shap_consistency_dir = "results/shap_consistency/"
os.makedirs(shap_consistency_dir, exist_ok=True)

df_shap_signs.to_csv(
    os.path.join(shap_consistency_dir, "shap_direction_signs_cross_models.csv"),
    index=False, encoding="utf-8-sig"
)
df_shap_importance.to_csv(
    os.path.join(shap_consistency_dir, "shap_meanabs_cross_models.csv"),
    index=False, encoding="utf-8-sig"
)
print("【提示】跨模型 SHAP 方向原始结果已保存。")

###################### 3. 汇总“跨模型方向一致性” + “跨模型重要性” ##########################

# 3.1. 对每个模型内部的 mean(|SHAP|) 做 0–1 归一化 → I_norm（模型内相对重要度）
rows_norm = []
for m_name, g in df_shap_importance.groupby("model"):
    v = g.set_index("feature")["mean_abs_shap"]
    vmax = v.max()
    if (vmax is None) or (vmax <= 0) or np.isnan(vmax):
        v_norm = v * 0.0
    else:
        v_norm = v / vmax
    for feat, val in v_norm.items():
        rows_norm.append({
            "model": m_name,
            "feature": feat,
            "I_norm": float(val)
        })

df_importance_norm = pd.DataFrame(rows_norm)

# 3.2. 在模型之间对 I_norm 取平均 → “跨模型归一化重要性” I_norm（0~1）
df_I_overall = (
    df_importance_norm
    .groupby("feature", as_index=False)["I_norm"]
    .mean()   # 直接命名为 I_norm，便于后续调用
)

'''
# 3.3. 计算每个变量在不同模型之间的“方向一致性” （旧版本） 
    
rows_dir = []
for feat, g in df_shap_signs.groupby("feature"):
    s = g["sign"].values.astype(float)
    if s.size == 0:
        continue
    # 模型间平均 sign，范围 [-1, 1]
    S_overall = float(np.mean(s))
    # 主方向：+1 / -1 / 0
    if S_overall > 0:
        majority = 1
    elif S_overall < 0:
        majority = -1
    else:
        majority = 0
    # Q_agree：有多少比例的模型“支持主方向”
    if majority == 0:
        Q_agree = 0.0
    else:
        Q_agree = float(np.mean(np.sign(s) == majority))
    rows_dir.append({
        "feature": feat,
        "S_overall": S_overall,
        "S_abs": abs(S_overall),
        "Q_agree": Q_agree,
        "n_models": int(s.size)
    })

df_dir_overall = pd.DataFrame(rows_dir)
'''

# 3.3. 计算每个变量在不同模型之间的“方向一致性” 

#旧版本思路（见https://github.com/MinglongCheng/binge-eating-bariatric-analysis）：
#   1）每个模型内部：Spearman(特征, SHAP) → r → 只看 sign(r)，得到 -1/0/+1；
#   2）跨模型：对这些 sign 取平均 S_overall，再算有多少模型支持主方向 Q_agree。
#   这种“纯投票”方法适用于模型/折数比较多的情况。
#   但在本项目里，很多特征在 6 个模型里几乎都是“稳稳的同向”，
#   导致 sign 几乎全是 +1 → S_overall = 1, Q_agree = 1，再乘在一起就接近 1，看起来“全满格”。
#
# 新方法（当前实现）：
#   直觉：每个模型内部的 r 是用几百个样本算出来的，可以利用 r 的大小来拉开差距，
#   而不是只看 sign(r) 的投票结果。
#   仍然保留“每个模型里先算 Spearman r”这一步，然后做跨模型汇总：
#     mean_r : 该特征在所有模型中的 Spearman ρ 的均值（带符号）；
#     S_abs  = |mean_r|，作为“方向 + 单调性强度”，取值范围 [0, 1]；
#     Q_agree：还是用 sign(r) 做投票，计算“有多少比例的模型与主方向同号”
#              （主方向由 sign(r) 的多数票给出）；
#     下面的雷达图默认使用 S_abs（|mean ρ|），如需更严格可改用 S_abs × Q_agree。
#     在 use_metric="S_abs"（方向一致性指标：'S_abs' 或 'S_abs_times_Q'）处修改。

rows_dir = []
for feat, g in df_shap_signs.groupby("feature"):
    # 该特征在不同模型中的 Spearman ρ
    r = g["spearman_r"].values.astype(float)
    r = r[~np.isnan(r)]   # 去掉 NaN
    if r.size == 0:
        continue

    # 3.3.1 基于 Spearman ρ 的“方向 + 强度”
    mean_r = float(np.mean(r))    # 可以为正也可以为负
    S_abs  = abs(mean_r)          # |mean ρ|，范围 [0,1]，越大表示越单调、越稳定

    # 3.3.2 基于符号的“一致性比例”Q_agree（和之前类似，但只把 |r| 很小的当成 0）
    s = np.sign(r)
    s[np.abs(r) < SIGN_TOL] = 0   # |ρ| < SIGN_TOL 视为方向不确定

    if s.size == 0:
        continue

    S_overall_sign = float(np.mean(s))  # 平均 sign（-1~1）
    # 主方向：+1 / -1 / 0
    if S_overall_sign > 0:
        majority = 1
    elif S_overall_sign < 0:
        majority = -1
    else:
        majority = 0

    if majority == 0:
        Q_agree = 0.0
    else:
        Q_agree = float(np.mean(s == majority))

    rows_dir.append({
        "feature": feat,
        "mean_r": mean_r,            # 平均 Spearman ρ
        "S_abs": S_abs,              # |mean ρ|
        "S_overall": S_overall_sign, # 平均 sign（主要用于 direction_txt）
        "Q_agree": Q_agree,
        "n_models": int(r.size)
    })

df_dir_overall = pd.DataFrame(rows_dir)

# 3.4. 合并“方向” + “重要性”
df_shap_direction = (
    df_dir_overall
    .merge(df_I_overall, on="feature", how="left")
)

# 方向一致性综合指标：
#   S_abs = |mean Spearman ρ|
#   S_abs_times_Q = |mean ρ| × Q_agree
df_shap_direction["S_abs_times_Q"] = df_shap_direction["S_abs"] * df_shap_direction["Q_agree"]

# 把方向变成文本（+ / − / 0），方便在图上标注
def _sign_to_txt(x):
    if x > 0:
        return "+"
    elif x < 0:
        return "−"
    else:
        return "0"

df_shap_direction["direction_txt"] = df_shap_direction["S_overall"].apply(_sign_to_txt)

# 按跨模型重要性 I_norm 从高到低排序
df_shap_direction = df_shap_direction.sort_values("I_norm", ascending=False)

# 保存最终汇总表
df_shap_direction.to_csv(
    os.path.join(shap_consistency_dir, "shap_direction_overall_cross_models.csv"),
    index=False, encoding="utf-8-sig"
)
print("【提示】跨模型 SHAP 方向一致性汇总表已保存：",
      os.path.join(shap_consistency_dir, "shap_direction_overall_cross_models.csv"))

###################### 4. 绘制“跨模型的 SHAP 一致性”雷达图 ##########################

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize

# 与前面的图风格保持一致
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

def plot_shap_consistency_radar(
    df_direction,
    top_k=8,
    use_metric="S_abs", # 方向一致性指标：'S_abs' 或 'S_abs_times_Q'
    save_dir="results/shap_consistency/",
    filename="shap_consistency_radar_top8.png"
):
    """
    绘制“跨模型的 SHAP 一致性分析”雷达图。

    参数说明：
      df_direction : 包含 I_norm、S_overall、S_abs、Q_agree、S_abs_times_Q、direction_txt 的汇总表
      top_k        : 取 I_norm 排名前 top_k 的变量
      use_metric   : 雷达图蓝线所用的方向一致性指标（'S_abs' 或 'S_abs_times_Q'）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 选取 I_norm 排名前 top_k 的变量
    df_plot = df_direction.sort_values("I_norm", ascending=False).head(top_k).copy()

    feats = df_plot["feature"].tolist()
    I_norm = df_plot["I_norm"].values

    # 2. 根据 use_metric 选择方向一致性指标 + 图例标签
    if use_metric == "S_abs":
        r_line = df_plot["S_abs"].values
        line_label = "Direction consistency\n(|mean ρ|)" # \n 换行
    elif use_metric == "S_abs_times_Q":
        r_line = df_plot["S_abs_times_Q"].values
        line_label = "Direction consistency\n(|mean ρ| × agreement)"
    else:
        raise ValueError("use_metric 只能是 'S_abs' 或 'S_abs_times_Q'")
    r_line = np.clip(r_line, 0, 1)
    

    # 标签：变量名 + 主方向 (+ / − / 0)
    labels = [f"{f} ({d})" for f, d in zip(df_plot["feature"], df_plot["direction_txt"])]

    # 3. 极坐标基本设置
    n = len(feats)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.4  # 扇形宽度

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)   # 0° 放在正上方
    ax.set_theta_direction(-1)       # 顺时针
    ax.spines['polar'].set_visible(False)  # 隐藏最外圈边框

    #图的总标题 → 针对整个 figure 居中 不想要标题可以删掉
    #fig.suptitle("Cross-model SHAP consistency analysis",
                 #fontsize=12, y=0.98)
    #fig.subplots_adjust(top=0.982)    # 给标题留一点空间

    # 参考圈 & 辐射虚线
    tt = np.linspace(0, 2 * np.pi, 512)
    for rr in [0.25, 0.50, 0.75]:
        ax.plot(tt, np.full_like(tt, rr), linestyle="-", linewidth=0.8, color="0.85", zorder=1)
    ax.plot(tt, np.full_like(tt, 1.00), linestyle="-", linewidth=0.9, color="black", zorder=2)
    for ang in theta:
        ax.plot([ang, ang], [0.0, 1.0], ls="--", lw=0.7, color="0.80", zorder=1)

    # 4. 径向扇形条：表示跨模型归一化重要性 I_norm
    cmap = LinearSegmentedColormap.from_list("bpr", ["#2c7bb6", "#7b3294"])
    norm = Normalize(vmin=0.0, vmax=1.0)
    bar_colors = cmap(norm(I_norm))
    heights = np.clip(I_norm, 0, 1.0)
    ax.bar(theta, heights, width=width, bottom=0.0, color=bar_colors,
           alpha=0.7, edgecolor="none", zorder=3)

    # 5. 蓝色折线：方向一致性指标
    th = np.concatenate([theta, theta[:1]])
    rr = np.concatenate([r_line, r_line[:1]])
    ax.plot(th, rr, color="tab:blue", linewidth=2.5,
            label=line_label, zorder=5) 
        
    # 5. 径向标签和特征名（与半径垂直）
    ax.set_xticks([])
    label_r = 1.06
    for ang, lab in zip(theta, labels):
        deg = np.degrees(ang)
        rot = 180.0 - deg
        rmod = rot % 360.0
        if 90.0 < rmod < 270.0:
            rot -= 180.0
        ax.text(ang, label_r, lab, rotation=rot, rotation_mode="anchor",
                ha="center", va="center", fontsize=10, color="black")

    # 6. 径向刻度
    ax.set_ylim(0, label_r + 0.01) ##与径向标签和特征名 label_r = 1.06 一起配合着改
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_rlabel_position(65)
    for t in ax.get_yticklabels():
        t.set_color("0.45")
        t.set_fontsize(10)
    ax.tick_params(axis="y", pad=6)

    # 7. 颜色条：表示 I_norm 高低
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # 关键：显式指定 ax=ax，避免 "Unable to determine Axes to steal space for Colorbar" 错误
    # fraction 控制色条长度，shrink 控制整体缩放
    cbar = plt.colorbar(sm, ax=ax, pad=0.12, fraction=0.025, shrink=0.8)
    cbar.set_label("Normalized cross-model importance\n(mean |SHAP value|)", fontsize=9) #\n换行
    cbar.ax.tick_params(labelsize=8)

    # 8. 图例 & 标题
    leg = ax.legend(
    loc="upper right",             # 以右下角为参考，右下角"lower right"
    bbox_to_anchor=(1.30, 0.96),  # (x, y) 是相对轴的坐标，可按需要微调
    borderaxespad=0.0)
    # 让图例文字居中（含多行文字）
    for txt in leg.get_texts():
        txt.set_ha("center")
    plt.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.show()
    print("【提示】跨模型 SHAP 一致性雷达图已保存为：", out_path)

# 默认：取 I_norm 排名前 8 的变量，用 "S_abs" 作为方向一致性指标
plot_shap_consistency_radar(
    df_shap_direction,
    top_k=8,
    use_metric="S_abs",
    save_dir=shap_consistency_dir,
    filename="shap_consistency_radar_top8.png"
)
