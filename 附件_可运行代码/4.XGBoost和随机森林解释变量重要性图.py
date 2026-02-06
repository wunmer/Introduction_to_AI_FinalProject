import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 改变当前工作目录
os.chdir(script_dir)

print(f"原工作目录: {os.getcwd()}")  # 改变前
os.chdir(script_dir)
print(f"新工作目录: {os.getcwd()}")  # 改变后


# ==================== 导入必要的库 ====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载与预处理 ====================
print("=== 数据加载与预处理 ===")

# 读取数据
data = pd.read_csv("WineQT.csv")  # 请确保文件路径正确
print(f"数据集形状: {data.shape}")
print(f"数据列名: {data.columns.tolist()}")

# 删除ID列
if 'Id' in data.columns:
    data = data.drop('Id', axis=1)
    print("已删除Id列")
else:
    print("未找到Id列")

# 检查缺失值
print(f"缺失值统计:\n{data.isnull().sum()}")

# 分离特征和目标变量
X = data.drop('quality', axis=1)  # 特征
y = data['quality']               # 目标变量

print(f"\n特征数据形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 检查目标变量分布
print(f"\n目标变量分布:\n{y.value_counts().sort_index()}")

# ==================== 2. 划分训练集和测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # 使用分层抽样处理不平衡数据
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"训练集目标变量分布:\n{y_train.value_counts().sort_index()}")
print(f"测试集目标变量分布:\n{y_test.value_counts().sort_index()}")

# ==================== 3. 随机森林模型 ====================
print("\n" + "="*50)
print("随机森林模型训练")
print("="*50)

# 创建随机森林模型
rf_model = RandomForestRegressor(
    n_estimators=100,      # 树的数量
    max_depth=10,          # 最大深度
    min_samples_split=5,   # 内部节点再划分所需最小样本数
    min_samples_leaf=2,    # 叶节点最小样本数
    max_features='sqrt',   # 寻找最佳分割时考虑的特征数
    random_state=42,
    n_jobs=-1              # 使用所有CPU核心
)

# 训练模型
rf_model.fit(X_train, y_train)

# 预测
y_pred_rf = rf_model.predict(X_test)

# 评估指标
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"随机森林模型性能:")
print(f"MSE:  {mse_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"MAE:  {mae_rf:.4f}")
print(f"R²:   {r2_rf:.4f}")

# 特征重要性
rf_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\n随机森林特征重要性 (Top 5):")
print(rf_importance.head(5))

# ==================== 4. XGBoost模型 ====================
print("\n" + "="*50)
print("XGBoost模型训练")
print("="*50)

# 创建XGBoost模型
xgb_model = XGBRegressor(
    n_estimators=100,       # 树的数量
    max_depth=6,            # 最大深度
    learning_rate=0.1,      # 学习率
    subsample=0.8,          # 样本采样比例
    colsample_bytree=0.8,   # 特征采样比例
    reg_alpha=0.1,          # L1正则化
    reg_lambda=1,           # L2正则化
    random_state=42,
    n_jobs=-1               # 使用所有CPU核心
)

# 训练模型
xgb_model.fit(X_train, y_train)

# 预测
y_pred_xgb = xgb_model.predict(X_test)

# 评估指标
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost模型性能:")
print(f"MSE:  {mse_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.4f}")
print(f"MAE:  {mae_xgb:.4f}")
print(f"R²:   {r2_xgb:.4f}")

# 特征重要性
xgb_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': xgb_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\nXGBoost特征重要性 (Top 5):")
print(xgb_importance.head(5))

# ==================== 5. 结果比较 ====================
print("\n" + "="*50)
print("模型性能比较")
print("="*50)

results = pd.DataFrame({
    '模型': ['随机森林', 'XGBoost'],
    'MSE': [mse_rf, mse_xgb],
    'RMSE': [rmse_rf, rmse_xgb],
    'MAE': [mae_rf, mae_xgb],
    'R²': [r2_rf, r2_xgb]
})

print(results)

# ==================== 6. 使用Plotly绘制特征重要性图 ====================
print("\n" + "="*50)
print("绘制特征重要性图")
print("="*50)

# 海洋清风配色方案
ocean_colors = [
    '#BFDFD2', '#51999F', '#4198AC', '#7BC0CD', '#DBCB92',
    '#ECB66C', '#EA9E58', '#ED8D5A', '#E67E51', '#D35451'
]

# 创建子图
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('随机森林特征重要性', 'XGBoost特征重要性'),
    horizontal_spacing=0.2
)

# 随机森林特征重要性图
fig.add_trace(
    go.Bar(
        x=rf_importance['重要性'][:10],
        y=rf_importance['特征'][:10],
        orientation='h',
        marker=dict(
            color=ocean_colors[:10],
            line=dict(color='#2C3E50', width=1)
        ),
        text=rf_importance['重要性'][:10].round(4),
        textposition='outside',
        name='随机森林'
    ),
    row=1, col=1
)

# XGBoost特征重要性图
fig.add_trace(
    go.Bar(
        x=xgb_importance['重要性'][:10],
        y=xgb_importance['特征'][:10],
        orientation='h',
        marker=dict(
            color=ocean_colors[:10],
            line=dict(color='#2C3E50', width=1)
        ),
        text=xgb_importance['重要性'][:10].round(4),
        textposition='outside',
        name='XGBoost'
    ),
    row=1, col=2
)

# 更新布局
fig.update_layout(
    title=dict(
        text='模型特征重要性比较',
        font=dict(size=24, color='#2C3E50'),
        x=0.5
    ),
    showlegend=False,
    height=600,
    width=1200,
    plot_bgcolor='white',
    paper_bgcolor='#F9F9F9'
)

# 更新x轴和y轴
fig.update_xaxes(
    title_text='特征重要性',
    showgrid=True,
    gridwidth=0.5,
    gridcolor='#E0E0E0',
    zerolinecolor='#E0E0E0'
)

fig.update_yaxes(
    title_text='特征',
    showgrid=False,
    autorange='reversed'  # 使最重要的特征显示在最上面
)

# 为每个子图单独设置y轴反转
fig.update_yaxes(autorange='reversed', row=1, col=1)
fig.update_yaxes(autorange='reversed', row=1, col=2)

# 保存为HTML文件
fig.write_html('特征重要性图.html')
print("特征重要性图已保存为 '特征重要性图.html'")

# 显示图表
fig.show()

# ==================== 7. 预测结果分析 ====================
print("\n" + "="*50)
print("预测结果分析")
print("="*50)

# 创建预测结果对比表
predictions_comparison = pd.DataFrame({
    '真实值': y_test.values,
    '随机森林预测': y_pred_rf,
    'XGBoost预测': y_pred_xgb,
    '随机森林残差': y_test.values - y_pred_rf,
    'XGBoost残差': y_test.values - y_pred_xgb
})

print("前10个样本的预测结果:")
print(predictions_comparison.head(10))

# 计算残差统计
print(f"\n随机森林残差统计:")
print(f"平均残差: {predictions_comparison['随机森林残差'].mean():.4f}")
print(f"残差标准差: {predictions_comparison['随机森林残差'].std():.4f}")
print(f"残差绝对值≤1的比例: {(abs(predictions_comparison['随机森林残差']) <= 1).mean()*100:.2f}%")

print(f"\nXGBoost残差统计:")
print(f"平均残差: {predictions_comparison['XGBoost残差'].mean():.4f}")
print(f"残差标准差: {predictions_comparison['XGBoost残差'].std():.4f}")
print(f"残差绝对值≤1的比例: {(abs(predictions_comparison['XGBoost残差']) <= 1).mean()*100:.2f}%")

# ==================== 8. 绘制残差分布图 ====================
print("\n" + "="*50)
print("绘制残差分布图")
print("="*50)

# 创建残差分布子图
fig_residuals = make_subplots(
    rows=1, cols=2,
    subplot_titles=('随机森林残差分布', 'XGBoost残差分布'),
    horizontal_spacing=0.15
)

# 随机森林残差分布
fig_residuals.add_trace(
    go.Histogram(
        x=predictions_comparison['随机森林残差'],
        nbinsx=30,
        marker=dict(
            color=ocean_colors[2],
            line=dict(color='#2C3E50', width=1)
        ),
        name='随机森林残差',
        opacity=0.8
    ),
    row=1, col=1
)

# XGBoost残差分布
fig_residuals.add_trace(
    go.Histogram(
        x=predictions_comparison['XGBoost残差'],
        nbinsx=30,
        marker=dict(
            color=ocean_colors[6],
            line=dict(color='#2C3E50', width=1)
        ),
        name='XGBoost残差',
        opacity=0.8
    ),
    row=1, col=2
)

# 更新布局
fig_residuals.update_layout(
    title=dict(
        text='模型残差分布',
        font=dict(size=24, color='#2C3E50'),
        x=0.5
    ),
    showlegend=False,
    height=500,
    width=1000,
    plot_bgcolor='white',
    paper_bgcolor='#F9F9F9'
)

# 更新x轴和y轴
fig_residuals.update_xaxes(
    title_text='残差值',
    showgrid=True,
    gridwidth=0.5,
    gridcolor='#E0E0E0'
)

fig_residuals.update_yaxes(
    title_text='频数',
    showgrid=True,
    gridwidth=0.5,
    gridcolor='#E0E0E0'
)

# 保存残差分布图
fig_residuals.write_html('残差分布图.html')
print("残差分布图已保存为 '残差分布图.html'")

# 显示残差分布图
fig_residuals.show()

# ==================== 9. 保存结果 ====================
print("\n" + "="*50)
print("保存结果")
print("="*50)

# 保存模型性能结果
results.to_csv('模型性能比较.csv', index=False, encoding='utf-8-sig')

# 保存预测结果
predictions_comparison.to_csv('预测结果对比.csv', index=False, encoding='utf-8-sig')

# 保存特征重要性
rf_importance.to_csv('随机森林特征重要性.csv', index=False, encoding='utf-8-sig')
xgb_importance.to_csv('XGBoost特征重要性.csv', index=False, encoding='utf-8-sig')

print("已保存以下文件:")
print("1. 模型性能比较.csv")
print("2. 预测结果对比.csv")
print("3. 随机森林特征重要性.csv")
print("4. XGBoost特征重要性.csv")
print("5. 特征重要性图.html")
print("6. 残差分布图.html")

# ==================== 10. 可选：模型调参（如果需要） ====================
print("\n" + "="*50)
print("可选：模型调参示例")
print("="*50)

# 随机森林调参示例（使用网格搜索）
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# 创建网格搜索对象
grid_search_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid_rf,
    cv=5,  # 5折交叉验证
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# 执行网格搜索（可选，可能需要较长时间）
# print("开始随机森林网格搜索...")
# grid_search_rf.fit(X_train, y_train)
# print(f"最佳参数: {grid_search_rf.best_params_}")
# print(f"最佳交叉验证分数: {-grid_search_rf.best_score_:.4f}")

print("模型训练完成！")