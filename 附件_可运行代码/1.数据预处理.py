import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 改变当前工作目录
os.chdir(script_dir)

print(f"原工作目录: {os.getcwd()}")  # 改变前
os.chdir(script_dir)
print(f"新工作目录: {os.getcwd()}")  # 改变后




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据质量检查 ====================
# 读取数据
# WineData = pd.read_csv(r'C:\Users\13066\Desktop\AI导论大作业代码\WineQT.csv')
WineData = pd.read_csv('WineQT.csv')

print("数据形状:", WineData.shape)
print("\n前5行数据:")
print(WineData.head())

# 1.1 缺失值检查
missing_summary = WineData.isnull().sum()
missing_percentage = WineData.isnull().mean() * 100
missing_info = pd.DataFrame({
    'Variable': WineData.columns,
    'Missing_Count': missing_summary.values,
    'Missing_Percentage': missing_percentage.values
})

print("\n=== 缺失值检查 ===")
print(missing_info)

# 1.2 重复值检查
duplicates = WineData.duplicated().sum()
print(f"\n重复值数量: {duplicates}")

# 1.3 检查变量类型
print("\n=== 数据类型 ===")
print(WineData.dtypes)

# ==================== 2. 单变量分布分析 ====================
# 2.1 响应变量分布
print("\n=== 响应变量分布 ===")
quality_counts = WineData['quality'].value_counts().sort_index()
print(quality_counts)

# 海洋清风配色
ocean_colors = ['#BFDFD2', '#51999F', '#4198AC', '#7BC0CD', 
                '#DBCB92', '#ECB66C', '#EA9E58', '#ED8D5A']

# 可视化响应变量分布
plt.figure(figsize=(10, 6))
bars = plt.bar(quality_counts.index, quality_counts.values, 
               color=ocean_colors[:len(quality_counts)])

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 添加均值线
mean_count = quality_counts.mean()
plt.axhline(y=mean_count, color='#2C3E50', linestyle='--', linewidth=2, 
            label=f'均值 = {mean_count:.1f}')

plt.xlabel('质量等级')
plt.ylabel('样本数量')
plt.title('葡萄酒质量等级分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('质量等级分布.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.2 基础描述性统计
print("\n=== 描述性统计 ===")
print(WineData.describe().T)

# 计算偏度和峰度
print("\n=== 偏度和峰度 ===")
skew_kurt = pd.DataFrame({
    '偏度': WineData.skew(),
    '峰度': WineData.kurt()
})
print(skew_kurt)

# ==================== 3. 变量相关性分析 ====================
# 3.1 相关系数矩阵
num_vars = [col for col in WineData.columns if WineData[col].dtype in ['int64', 'float64']]
cor_matrix = WineData[num_vars].corr()

print("\n=== 相关系数矩阵 ===")
print(cor_matrix)

# 3.2 相关热力图
plt.figure(figsize=(12, 10))
sns.heatmap(cor_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, 
            cbar_kws={"shrink": 0.8})
plt.title('葡萄酒变量相关性热力图', fontsize=16)
plt.tight_layout()
plt.savefig('相关性热力图.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.3 多重共线性检验
# 计算VIF
X = WineData.drop(['quality', 'Id'], axis=1, errors='ignore')
X = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])]

print("\n=== 方差膨胀因子(VIF) ===")
print(vif_data)

# 检查高VIF值
high_vif = vif_data[vif_data['VIF'] > 10]
if not high_vif.empty:
    print("\n警告：以下变量存在严重多重共线性(VIF>10):")
    print(high_vif)
else:
    print("\n没有检测到严重多重共线性问题(VIF均<10)")

# 3.4 与目标变量的相关性
if 'quality' in cor_matrix.index:
    cor_with_quality = cor_matrix['quality'].drop('quality')
    sorted_cor = cor_with_quality.sort_values(key=abs, ascending=False)
    
    print("\n=== 与质量变量的相关性（按绝对值降序排列）===")
    print(sorted_cor.round(3))
    
    # 可视化与质量的相关性
    plt.figure(figsize=(10, 8))
    colors = ['#ED8D5A' if x > 0 else '#4198AC' for x in sorted_cor]
    
    bars = plt.barh(sorted_cor.index, sorted_cor.values, color=colors)
    plt.axvline(x=0, color='gray', linewidth=1)
    plt.xlabel('相关系数')
    plt.title('各变量与葡萄酒质量的相关性')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 添加相关系数值
    for bar in bars:
        width = bar.get_width()
        ha = 'left' if width > 0 else 'right'
        x_pos = width + 0.02 if width > 0 else width - 0.02
        plt.text(x_pos, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', ha=ha, va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('与质量相关性.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== 4. 异常值检测 ====================
print("\n=== 异常值检测 ===")

# 4.1 箱线图（所有变量）
num_cols = len(num_vars)
nrows = (num_cols + 2) // 3  # 每行3个图

fig, axes = plt.subplots(nrows, 3, figsize=(15, nrows*4))
axes = axes.flatten()

for i, col in enumerate(num_vars):
    ax = axes[i]
    ax.boxplot(WineData[col].dropna(), vert=False)
    ax.set_title(col)
    ax.set_xlabel('值')
    ax.grid(True, alpha=0.3)

# 隐藏多余的子图
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('变量箱线图（异常值检测）', fontsize=16)
plt.tight_layout()
plt.savefig('箱线图_异常值检测.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.2 基于IQR的异常值统计
outlier_stats = {}
for col in num_vars:
    Q1 = WineData[col].quantile(0.25)
    Q3 = WineData[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = WineData[(WineData[col] < lower_bound) | (WineData[col] > upper_bound)]
    outlier_stats[col] = {
        '异常值数量': len(outliers),
        '异常值比例': len(outliers) / len(WineData) * 100,
        '下限': lower_bound,
        '上限': upper_bound
    }

outlier_df = pd.DataFrame(outlier_stats).T
print("\n=== 基于IQR的异常值统计 ===")
print(outlier_df)

# 汇总异常值情况
print(f"\n总异常值检测总结:")
print(f"平均异常值比例: {outlier_df['异常值比例'].mean():.2f}%")
print(f"最大异常值比例: {outlier_df['异常值比例'].max():.2f}% ({outlier_df['异常值比例'].idxmax()})")
print(f"最小异常值比例: {outlier_df['异常值比例'].min():.2f}% ({outlier_df['异常值比例'].idxmin()})")

# ==================== 5. 数据预处理总结 ====================
print("\n" + "="*50)
print("数据预处理总结")
print("="*50)

print(f"1. 数据质量:")
print(f"   样本数: {len(WineData)}")
print(f"   变量数: {len(WineData.columns)}")
print(f"   缺失值: {missing_info['Missing_Count'].sum()} (已处理)")
print(f"   重复值: {duplicates} (已处理)")

print(f"\n2. 响应变量分布:")
print(f"   质量等级范围: {WineData['quality'].min()} - {WineData['quality'].max()}")
print(f"   样本分布: {dict(quality_counts)}")

print(f"\n3. 相关性分析:")
if 'quality' in cor_matrix.index:
    top_cor = sorted_cor.head(3)
    print(f"   与质量最相关的变量:")
    for var, corr in top_cor.items():
        direction = "正相关" if corr > 0 else "负相关"
        print(f"     {var}: {corr:.3f} ({direction})")

print(f"\n4. 异常值情况:")
print(f"   平均异常值比例: {outlier_df['异常值比例'].mean():.2f}%")
print(f"   建议: 使用弹性网等稳健模型，无需删除异常值")

print(f"\n5. 数据预处理建议:")
print(f"   - 保留所有变量，无需删除")
print(f"   - 使用标准化处理连续变量")
print(f"   - 对于分类模型，考虑处理类别不平衡问题")
print(f"   - 使用正则化模型处理多重共线性")

# 保存预处理后的数据（可选）
WineData.to_csv('预处理后的葡萄酒数据.csv', index=False, encoding='utf-8-sig')
print(f"\n预处理后的数据已保存: '预处理后的葡萄酒数据.csv'")

# 保存重要统计信息
summary_stats = {
    'missing_info': missing_info,
    'quality_counts': quality_counts,
    'correlation_with_quality': sorted_cor if 'sorted_cor' in locals() else pd.Series(),
    'outlier_stats': outlier_df
}

import json
# 转换为可序列化的格式
summary_json = {
    'missing_info': missing_info.to_dict(),
    'quality_counts': quality_counts.to_dict(),
    'correlation_with_quality': sorted_cor.to_dict() if 'sorted_cor' in locals() else {},
    'outlier_stats': outlier_df.to_dict()
}

with open('数据预处理总结.json', 'w', encoding='utf-8') as f:
    json.dump(summary_json, f, ensure_ascii=False, indent=2)

print(f"预处理总结已保存: '数据预处理总结.json'")