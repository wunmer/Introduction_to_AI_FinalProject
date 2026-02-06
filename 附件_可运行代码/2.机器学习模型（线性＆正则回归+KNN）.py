import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 改变当前工作目录
os.chdir(script_dir)

print(f"原工作目录: {os.getcwd()}")  # 改变前
os.chdir(script_dir)
print(f"新工作目录: {os.getcwd()}")  # 改变后


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据准备 ====================
# 读取数据
data = pd.read_csv(r'WineQT.csv')

# 定义响应变量和解释变量
y = data['quality']
X = data.drop(['quality', 'Id'], axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# 定义配色方案
ocean_colors = ['#BFDFD2', '#51999F', '#4198AC', '#7BC0CD', 
                '#DBCB92', '#ECB66C', '#EA9E58', '#ED8D5A']

# ==================== 2. 多元线性回归 ====================
lm_model = LinearRegression()
lm_model.fit(X_train, y_train)
y_pred_lm = lm_model.predict(X_test)
mse_lm = mean_squared_error(y_test, y_pred_lm)
r2_lm = r2_score(y_test, y_pred_lm)

print("\n=== 多元线性回归 ===")
print(f"测试集MSE: {mse_lm:.4f}")
print(f"R²: {r2_lm:.4f}")

# ==================== 3. 岭回归 ====================
ridge_cv = GridSearchCV(Ridge(), 
                       param_grid={'alpha': np.logspace(-4, 4, 100)},
                       cv=10, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)
best_alpha_ridge = ridge_cv.best_params_['alpha']

ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\n=== 岭回归 ===")
print(f"最优alpha: {best_alpha_ridge:.4f}")
print(f"测试集MSE: {mse_ridge:.4f}")
print(f"R²: {r2_ridge:.4f}")

# 计算CV曲线数据
alphas = np.logspace(-4, 4, 100)
cv_scores = []
cv_std = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y_train, 
                            cv=10, scoring='neg_mean_squared_error')
    cv_scores.append(-np.mean(scores))
    cv_std.append(np.std(scores))

# ==================== 4. Lasso回归 ====================
lasso_cv = GridSearchCV(Lasso(max_iter=10000), 
                       param_grid={'alpha': np.logspace(-4, 2, 100)},
                       cv=10, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train_scaled, y_train)
best_alpha_lasso = lasso_cv.best_params_['alpha']

lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

selected_vars = X.columns[lasso_model.coef_ != 0].tolist()

print("\n=== Lasso回归 ===")
print(f"测试集MSE: {mse_lasso:.4f}")
print(f"R²: {r2_lasso:.4f}")
print(f"选中的关键变量: {selected_vars}")

# ==================== 5. 弹性网回归 ====================
alphas_enet = [0.2, 0.5, 0.8]
results_enet = []

for alpha_val in alphas_enet:
    enet_cv = GridSearchCV(ElasticNet(max_iter=10000), 
                          param_grid={'alpha': np.logspace(-4, 2, 50),
                                     'l1_ratio': [alpha_val]},
                          cv=10, scoring='neg_mean_squared_error')
    enet_cv.fit(X_train_scaled, y_train)
    
    enet_model = ElasticNet(alpha=enet_cv.best_params_['alpha'], 
                           l1_ratio=alpha_val, max_iter=10000)
    enet_model.fit(X_train_scaled, y_train)
    y_pred = enet_model.predict(X_test_scaled)
    mse_val = mean_squared_error(y_test, y_pred)
    
    results_enet.append({
        'alpha': alpha_val,
        'lambda': enet_cv.best_params_['alpha'],
        'mse': mse_val,
        'model': enet_model
    })

# 选择最佳模型
best_enet = min(results_enet, key=lambda x: x['mse'])
enet_final = best_enet['model']
y_pred_enet = enet_final.predict(X_test_scaled)
mse_enet = best_enet['mse']
r2_enet = r2_score(y_test, y_pred_enet)

important_vars = X.columns[enet_final.coef_ != 0].tolist()

print("\n=== 弹性网回归 ===")
print(f"最优alpha: {best_enet['alpha']:.2f}")
print(f"最优lambda: {best_enet['lambda']:.4f}")
print(f"测试集MSE: {mse_enet:.4f}")
print(f"R²: {r2_enet:.4f}")
print(f"主要非零系数变量: {important_vars}")

# ==================== 6. K近邻回归 ====================
# 尝试不同的K值
k_values = range(1, 21)
k_results = []

for k_val in k_values:
    knn = KNeighborsRegressor(n_neighbors=k_val)
    mse_cv = -cross_val_score(knn, X_train_scaled, y_train, 
                             cv=5, scoring='neg_mean_squared_error').mean()
    k_results.append({'k': k_val, 'mse': mse_cv})

# 转换为DataFrame
k_results_df = pd.DataFrame(k_results)
best_k = k_results_df.loc[k_results_df['mse'].idxmin(), 'k']

# 在测试集上评估
knn_model = KNeighborsRegressor(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print("\n=== K近邻回归 ===")
print(f"最优K值: {best_k}")
print(f"测试集MSE: {mse_knn:.4f}")
print(f"R²: {r2_knn:.4f}")

# ==================== 7. LDA线性判别分析 ====================
# 将响应变量转换为分类变量（保持原始评分3-8）
y_train_class = y_train.astype(int)
y_test_class = y_test.astype(int)

# 训练LDA模型
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_scaled, y_train_class)

# 预测
y_pred_class = lda_model.predict(X_test_scaled)

# 计算分类准确率
accuracy = accuracy_score(y_test_class, y_pred_class)
conf_matrix = confusion_matrix(y_test_class, y_pred_class)

# 计算LDA的"类MSE"（将类别视为数值）
y_test_numeric = y_test_class.values
y_pred_numeric = y_pred_class
mse_lda = mean_squared_error(y_test_numeric, y_pred_numeric)
r2_lda = r2_score(y_test_numeric, y_pred_numeric)
mae_lda = mean_absolute_error(y_test_numeric, y_pred_numeric)

print("\n=== LDA线性判别分析 ===")
print(f"分类准确率: {accuracy:.4f}")
print(f"测试集MSE（数值化）: {mse_lda:.4f}")
print(f"测试集MAE（数值化）: {mae_lda:.4f}")
print(f"测试集R²（数值化）: {r2_lda:.4f}")

# ==================== 8. 模型汇总与比较 ====================
results_summary = pd.DataFrame({
    '模型': ['线性回归', '岭回归', 'Lasso回归', '弹性网回归', 'KNN回归', 'LDA分类'],
    '测试集MSE': [round(mse_lm, 4), round(mse_ridge, 4), 
                  round(mse_lasso, 4), round(mse_enet, 4), 
                  round(mse_knn, 4), round(mse_lda, 4)],
    'R²': [round(r2_lm, 4), round(r2_ridge, 4), round(r2_lasso, 4),
           round(r2_enet, 4), round(r2_knn, 4), round(r2_lda, 4)]
})

print("\n=== 模型比较汇总 ===")
print(results_summary)

# ==================== 9. Plotly图表绘制 ====================

# 图表1: 不同回归模型的测试集MSE对比柱状图
fig1 = go.Figure(data=[
    go.Bar(
        x=results_summary['模型'],
        y=results_summary['测试集MSE'],
        marker_color=ocean_colors[:len(results_summary)],
        text=results_summary['测试集MSE'],
        textposition='auto',
        texttemplate='%{text:.4f}',
        hovertemplate='模型: %{x}<br>MSE: %{y:.4f}<br>R²: %{customdata:.4f}<extra></extra>',
        customdata=results_summary['R²']
    )
])

fig1.update_layout(
    title='不同回归模型的测试集MSE对比',
    xaxis_title='模型',
    yaxis_title='测试集MSE',
    template='plotly_white',
    showlegend=False,
    height=500,
    width=800
)

# 图表2: 岭回归交叉验证误差随λ变化曲线
fig2 = go.Figure()

# 添加主曲线
fig2.add_trace(go.Scatter(
    x=np.log10(alphas),
    y=cv_scores,
    mode='lines',
    name='CV MSE',
    line=dict(color=ocean_colors[1], width=3),
    hovertemplate='log(λ): %{x:.4f}<br>MSE: %{y:.4f}<extra></extra>'
))

# 添加最优lambda线
fig2.add_vline(
    x=np.log10(best_alpha_ridge),
    line=dict(color=ocean_colors[6], dash='dash', width=2),
    annotation=dict(text=f"最优λ = {best_alpha_ridge:.4f}", 
                   font=dict(size=12, color=ocean_colors[6]))
)

# 添加最优点
fig2.add_trace(go.Scatter(
    x=[np.log10(best_alpha_ridge)],
    y=[min(cv_scores)],
    mode='markers',
    marker=dict(size=12, color=ocean_colors[6], symbol='circle'),
    name='最优λ'
))

fig2.update_layout(
    title='岭回归交叉验证误差随正则化参数λ变化',
    xaxis_title='log(λ)',
    yaxis_title='交叉验证MSE',
    template='plotly_white',
    height=600,
    width=800
)

# 图表3: Lasso回归正则化路径图
from sklearn.linear_model import lasso_path

alphas_lasso, coefs_lasso, _ = lasso_path(X_train_scaled, y_train, 
                                          alphas=np.logspace(-4, 2, 100))

fig3 = go.Figure()

# 只显示前10个变量的路径，避免图太拥挤
max_vars = min(10, coefs_lasso.shape[0])
for i in range(max_vars):
    fig3.add_trace(go.Scatter(
        x=np.log10(alphas_lasso),
        y=coefs_lasso[i, :],
        mode='lines',
        name=X.columns[i],
        line=dict(width=2),
        hovertemplate=f'变量: {X.columns[i]}<br>log(λ): %{{x:.4f}}<br>系数: %{{y:.4f}}<extra></extra>'
    ))

fig3.add_vline(
    x=np.log10(best_alpha_lasso),
    line=dict(color=ocean_colors[6], dash='dash', width=2),
    annotation=dict(text=f"最优λ = {best_alpha_lasso:.4f}", 
                   font=dict(size=12, color=ocean_colors[6]))
)

fig3.update_layout(
    title='Lasso回归正则化路径',
    xaxis_title='log(λ)',
    yaxis_title='回归系数',
    template='plotly_white',
    height=600,
    width=900,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.02,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='lightgray',
        borderwidth=1
    )
)

# 图表4: 弹性网模型非零回归系数条形图
coef_data = pd.DataFrame({
    '变量': X.columns,
    '系数': enet_final.coef_
})
coef_data = coef_data[coef_data['系数'] != 0].sort_values('系数', key=abs, ascending=False)

fig4 = go.Figure(data=[
    go.Bar(
        x=coef_data['变量'],
        y=coef_data['系数'],
        marker_color=[ocean_colors[1] if c > 0 else ocean_colors[5] for c in coef_data['系数']],
        text=[f'{c:.3f}' for c in coef_data['系数']],
        textposition='auto',
        hovertemplate='变量: %{x}<br>系数: %{y:.4f}<extra></extra>'
    )
])

fig4.update_layout(
    title='弹性网模型非零回归系数',
    xaxis_title='变量',
    yaxis_title='系数值',
    template='plotly_white',
    showlegend=False,
    height=500,
    width=900,
    xaxis=dict(tickangle=45)
)

# 图表5: KNN回归中MSE随K变化折线图
fig5 = go.Figure()

# 添加主曲线
fig5.add_trace(go.Scatter(
    x=k_results_df['k'],
    y=k_results_df['mse'],
    mode='lines+markers',
    name='交叉验证MSE',
    line=dict(color=ocean_colors[1], width=3),
    marker=dict(size=8, color=ocean_colors[1]),
    hovertemplate='K值: %{x}<br>MSE: %{y:.4f}<extra></extra>'
))

# 添加最优K值线
fig5.add_vline(
    x=best_k,
    line=dict(color=ocean_colors[6], dash='dash', width=2),
    annotation=dict(text=f"最优K = {best_k}", 
                   font=dict(size=12, color=ocean_colors[6]))
)

# 添加最优点
fig5.add_trace(go.Scatter(
    x=[best_k],
    y=[k_results_df['mse'].min()],
    mode='markers',
    marker=dict(size=15, color=ocean_colors[6], symbol='circle'),
    name='最优K值',
    hovertemplate='最优K值: %{x}<br>最小MSE: %{y:.4f}<extra></extra>'
))

fig5.update_layout(
    title='KNN回归中MSE随K值变化',
    xaxis_title='K值（近邻数量）',
    yaxis_title='交叉验证MSE',
    template='plotly_white',
    height=600,
    width=800,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# 图表6: 弹性网模型真实值vs预测值散点图
scatter_data = pd.DataFrame({
    '真实品质评分': y_test,
    '预测品质评分': y_pred_enet,
    '残差': y_test - y_pred_enet
})

# 计算回归线
z = np.polyfit(y_test, y_pred_enet, 1)
p = np.poly1d(z)
x_range = np.linspace(y_test.min(), y_test.max(), 100)

fig6 = go.Figure()

# 添加散点
fig6.add_trace(go.Scatter(
    x=scatter_data['真实品质评分'],
    y=scatter_data['预测品质评分'],
    mode='markers',
    name='预测点',
    marker=dict(
        size=8,
        color=scatter_data['残差'].abs(),
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title='|残差|'),
        opacity=0.7
    ),
    text=[f'真实: {a:.2f}<br>预测: {p:.2f}<br>残差: {r:.2f}' 
          for a, p, r in zip(scatter_data['真实品质评分'], 
                            scatter_data['预测品质评分'], 
                            scatter_data['残差'])],
    hovertemplate='%{text}<extra></extra>'
))

# 添加完美预测线
fig6.add_trace(go.Scatter(
    x=[y_test.min(), y_test.max()],
    y=[y_test.min(), y_test.max()],
    mode='lines',
    name='完美预测线',
    line=dict(color=ocean_colors[7], width=2, dash='solid'),
    hovertemplate='真实: %{x:.2f}<br>理想预测: %{y:.2f}<extra></extra>'
))

# 添加回归线
fig6.add_trace(go.Scatter(
    x=x_range,
    y=p(x_range),
    mode='lines',
    name='回归线',
    line=dict(color=ocean_colors[4], width=2, dash='dash'),
    hovertemplate='真实: %{x:.2f}<br>预测趋势: %{y:.2f}<extra></extra>'
))

# 添加统计信息标注
r2 = r2_score(y_test, y_pred_enet)
mae = mean_absolute_error(y_test, y_pred_enet)
rmse = np.sqrt(mse_enet)

fig6.add_annotation(
    x=0.02, y=0.98,
    xref="paper", yref="paper",
    text=f"R² = {r2:.4f}<br>MAE = {mae:.4f}<br>RMSE = {rmse:.4f}<br>α = {best_enet['alpha']:.2f}<br>λ = {best_enet['lambda']:.4f}",
    showarrow=False,
    font=dict(size=12, color=ocean_colors[7]),
    align="left",
    bgcolor="white",
    bordercolor=ocean_colors[1],
    borderwidth=2,
    borderpad=4
)

fig6.update_layout(
    title='弹性网模型：真实值 vs 预测值',
    xaxis_title='真实品质评分',
    yaxis_title='预测品质评分',
    template='plotly_white',
    height=700,
    width=800,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# 设置坐标轴比例一致
fig6.update_xaxes(
    scaleanchor="y",
    scaleratio=1,
    range=[y_test.min()*0.95, y_test.max()*1.05]
)
fig6.update_yaxes(
    range=[y_test.min()*0.95, y_test.max()*1.05]
)

# 图表7: LDA分类结果散点图
fig7 = go.Figure()

# 将正确和错误的分类分开
correct_mask = (y_test_numeric == y_pred_numeric)

# 正确分类的点
fig7.add_trace(go.Scatter(
    x=y_test_numeric[correct_mask],
    y=y_pred_numeric[correct_mask],
    mode='markers',
    name='正确分类',
    marker=dict(
        size=10,
        color=ocean_colors[2],
        opacity=0.7,
        line=dict(width=1, color='white')
    ),
    text=[f'真实: {a}<br>预测: {p}' 
          for a, p in zip(y_test_numeric[correct_mask], 
                         y_pred_numeric[correct_mask])],
    hovertemplate='%{text}<br>状态: 正确<extra></extra>'
))

# 错误分类的点
fig7.add_trace(go.Scatter(
    x=y_test_numeric[~correct_mask],
    y=y_pred_numeric[~correct_mask],
    mode='markers',
    name='错误分类',
    marker=dict(
        size=12,
        color=ocean_colors[6],
        opacity=0.7,
        symbol='x',
        line=dict(width=2, color='white')
    ),
    text=[f'真实: {a}<br>预测: {p}<br>误差: {abs(a-p)}' 
          for a, p in zip(y_test_numeric[~correct_mask], 
                         y_pred_numeric[~correct_mask])],
    hovertemplate='%{text}<br>状态: 错误<extra></extra>'
))

# 添加完美预测线
fig7.add_trace(go.Scatter(
    x=[3, 8],
    y=[3, 8],
    mode='lines',
    name='完美预测线',
    line=dict(color='gray', width=2, dash='solid'),
    hovertemplate='真实: %{x}<br>理想预测: %{y}<extra></extra>'
))

# 添加准确率标注
fig7.add_annotation(
    x=0.02, y=0.98,
    xref="paper", yref="paper",
    text=f"分类准确率: {accuracy:.2%}<br>MSE: {mse_lda:.4f}<br>MAE: {mae_lda:.4f}",
    showarrow=False,
    font=dict(size=12, color=ocean_colors[7]),
    align="left",
    bgcolor="white",
    bordercolor=ocean_colors[1],
    borderwidth=2,
    borderpad=4
)

fig7.update_layout(
    title='LDA分类结果：真实评分 vs 预测评分',
    xaxis_title='真实葡萄酒品质评分',
    yaxis_title='LDA预测评分',
    template='plotly_white',
    height=700,
    width=800,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# 设置坐标轴范围和刻度
fig7.update_xaxes(
    range=[2.5, 8.5],
    tickmode='array',
    tickvals=list(range(3, 9))
)
fig7.update_yaxes(
    range=[2.5, 8.5],
    tickmode='array',
    tickvals=list(range(3, 9))
)

# 图表8: LDA混淆矩阵热图
fig8 = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=list(range(conf_matrix.shape[1])),
    y=list(range(conf_matrix.shape[0])),
    colorscale='Blues',
    text=conf_matrix,
    texttemplate="%{text}",
    textfont={"size": 14},
    hovertemplate='真实类别: %{y}<br>预测类别: %{x}<br>样本数: %{z}<extra></extra>'
))

fig8.update_layout(
    title='LDA分类混淆矩阵',
    xaxis_title='预测类别',
    yaxis_title='真实类别',
    template='plotly_white',
    height=600,
    width=700
)

# 图表9: 所有模型MSE对比（包含LDA）
fig9 = go.Figure(data=[
    go.Bar(
        x=results_summary['模型'],
        y=results_summary['测试集MSE'],
        marker_color=ocean_colors[:len(results_summary)],
        text=results_summary['测试集MSE'],
        textposition='auto',
        texttemplate='%{text:.4f}',
        hovertemplate='模型: %{x}<br>MSE: %{y:.4f}<br>R²: %{customdata:.4f}<extra></extra>',
        customdata=results_summary['R²']
    )
])

fig9.update_layout(
    title='所有模型测试集MSE对比（包含LDA分类）',
    xaxis_title='模型',
    yaxis_title='测试集MSE',
    template='plotly_white',
    showlegend=False,
    height=500,
    width=800
)

# ==================== 10. 显示图表 ====================

print("\n正在显示图表...")

# 图表1
fig1.show()

# 图表2
fig2.show()

# 图表3
fig3.show()

# 图表4
fig4.show()

# 图表5
fig5.show()

# 图表6
fig6.show()

# 图表7
fig7.show()

# 图表8
fig8.show()

# 图表9
fig9.show()

# ==================== 11. 保存图表为HTML文件 ====================
print("\n正在保存图表为HTML文件...")

# 保存所有图表为HTML文件（不需要kaleido）
fig1.write_html("图1_模型MSE对比柱状图.html")
fig2.write_html("图2_岭回归CV误差曲线.html")
fig3.write_html("图3_Lasso正则化路径图.html")
fig4.write_html("图4_弹性网非零系数条形图.html")
fig5.write_html("图5_KNN_MSE变化折线图.html")
fig6.write_html("图6_真实值预测值散点图.html")
fig7.write_html("图7_LDA真实值预测值散点图.html")
fig8.write_html("图8_LDA混淆矩阵.html")
fig9.write_html("图9_所有模型MSE对比.html")

# ==================== 12. 保存结果 ====================
# 保存结果到CSV
results_summary.to_csv('模型比较结果.csv', index=False, encoding='utf-8-sig')

# 保存重要变量信息
important_vars_summary = {
    'Lasso': selected_vars,
    '弹性网': important_vars
}
import json
with open('重要变量.json', 'w', encoding='utf-8') as f:
    json.dump(important_vars_summary, f, ensure_ascii=False, indent=2)

# 保存LDA详细结果
lda_results = {
    'confusion_matrix': conf_matrix.tolist(),
    'accuracy': accuracy,
    'mse': mse_lda,
    'mae': mae_lda,
    'r2': r2_lda,
    'y_test': y_test_numeric.tolist(),
    'y_pred': y_pred_numeric.tolist()
}
with open('LDA_分析结果.json', 'w', encoding='utf-8') as f:
    json.dump(lda_results, f, ensure_ascii=False, indent=2)

print("\n所有分析完成！")
print("已生成以下文件：")
print("1. 图1-图9的HTML格式图表（交互式）")
print("2. 模型比较结果.csv")
print("3. 重要变量.json")
print("4. LDA_分析结果.json")
print("\n注意：如需保存图表为PNG格式，请安装kaleido包：pip install kaleido")
print("安装后，您可以在代码中添加：fig.write_image('filename.png')")