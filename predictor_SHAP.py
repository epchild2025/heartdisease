# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:26:01 2026

@author: LENOVO
"""

import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import joblib
import shap
import matplotlib.pyplot as plt

######################## 1. 基础配置 ########################
# 加载训练好的最佳模型
model = joblib.load('xgb_model.pkl')

# 定义特征名称(替换为业务相关列名，与编码规则对应) 
feature_names = ['male','BPMeds','prevalentStroke','age','prevalentHyp','diabetes','cigsPerDay','sysBP','glucose','totChol','diaBP','BMI']

######################## 2. Streamlit页面配置 ########################
st.set_page_config(page_title="心脏病风险预测器", layout="wide") 
st.title("心脏病风险概率预测器")
st.markdown("请填写以下信息，点击预测获取心脏病风险评估结果")

######################## 3. 特征输入组件 ########################
#1.male(0：女性，1：男性) 
male = st.selectbox(
    "您的性别是?", 
    options = [0,1],
    format_func=lambda x:"女性" if x == 0 else"男性")

#2.BPMeds(0：未服用降压药，1：服用降压药) 
BPMeds = st.selectbox(
    "您是否正在服用降压药?", 
    options = [0,1],
    format_func=lambda x:"未服用降压药" if x == 0 else"服用降压药")

#3.prevalentStroke(0：无中风史，1：有中风史) 
prevalentStroke = st.selectbox(
    "您是否有过中风?", 
    options = [0,1],
    format_func=lambda x:"否" if x == 0 else"是")

# 4. age (连续变量：年龄)
age = st.number_input(
    "请输入您的年龄:",
    min_value=0,     
    max_value=120,   
    value=30,        
    step=1           
)

#5.prevalentHyp(0：无高血压史，1：有高血压史) 
prevalentHyp = st.selectbox(
    "您是否有过高血压?", 
    options = [0,1],
    format_func=lambda x:"否" if x == 0 else"是")

#6.diabetes(0：无糖尿病，1：有糖尿病) 
diabetes = st.selectbox(
    "您是否有糖尿病?", 
    options = [0,1],
    format_func=lambda x:"否" if x == 0 else"是")

# 7. cigsPerDay (连续变量：每天吸烟支数)
cigsPerDay = st.number_input(
    "您每天的吸烟数量（支）:",
    min_value=0,     
    max_value=120,   
    value=0,        
    step=1           
)

# 8. sysBP (连续变量：收缩压)
sysBP = st.number_input(
    "您的收缩压为:",
    min_value=0.0,     
    max_value=300.0,   
    value=120.0,        
    step=1.0          # 统一为浮点数
)

# 9. glucose (连续变量：血糖)
glucose = st.number_input(
    "您的血糖浓度为 (mmol/L):",
    min_value=0,     
    max_value=300,   
    value=120,        
    step=1           
)

# 10. totChol (连续变量：总胆固醇)
totChol = st.number_input(
    "您的总胆固醇为 (mmol/L):",
    min_value=0,     
    max_value=1000,   
    value=120,        
    step=1           
)

# 11. diaBP (连续变量：舒张压)
diaBP = st.number_input(
    "您的舒张压为:",
    min_value=0.0,     
    max_value=300.0,   
    value=120.0,        
    step=1.0  
)

# 12. BMI (连续变量：BMI)
BMI = st.number_input(
    "您的BMI为:",
    min_value=0.00,     
    max_value=60.00,   
    value=20.00,        
    step=0.01  
)

######################## 4. 数据处理与预测 ########################
feature_values = [male,BPMeds,prevalentStroke,age,prevalentHyp,diabetes,cigsPerDay,sysBP,glucose,totChol,diaBP,BMI]    
features = np.array([feature_values])

# 核心修改：将数组转为带有列名的 DataFrame，确保 SHAP 图能显示正确的临床指标名称
features_df = pd.DataFrame(features, columns=feature_names)

# 点击预测按钮后，执行下方所有缩进的代码
if st.button("预测"):
    # 模型预测
    predicted_class = model.predict(features)[0] 
    predicted_proba = model.predict_proba(features)[0] 

    # 显示预测结果
    st.subheader("预测结果")
    risk_label = "高风险" if predicted_class == 1 else "低风险" 
    st.write(f'**风险等级：{predicted_class} ({risk_label})**')
    
    st.write(f'**风险概率：**低风险概率 {predicted_proba[0]:.2%} | 高风险概率 {predicted_proba[1]:.2%}')
    
    # 生成个性化建议
    st.subheader("健康建议")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1: 
        advice = (
            f"模型预测您的心脏病风险为高风险(概率{probability:.1f}%)。 "
            "建议尽快前往医疗机构进行全面的心脏评估，"
            "同时可根据自身情况增加适宜的体育锻炼，改善生活习惯。"
        )
        st.warning(advice)
    else:
        advice = (
            f"模型预测您的心脏病风险为低风险(概率{probability:.1f}%)。 "
            "请继续保持良好的生活习惯，定期体检！"
        )
        st.success(advice)
        
    ######################## 5. SHAP (单样本特征贡献解释) ########################
    st.subheader("SHAP特征贡献解释 (瀑布图)") 

    # 1. 初始化 Explainer
    explainer = shap.TreeExplainer(model) 

    # 2. 计算当前患者的 SHAP 值 (传入带列名的 DataFrame)
    shap_values = explainer(features_df)

    # 3. 绘制并展示瀑布图
    fig, ax = plt.subplots(figsize=(8, 5))

    # 取第 0 个样本展示瀑布图
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)

    # 将 matplotlib 图像传递给 Streamlit 显示
    st.pyplot(fig)

    # 清理内存，防止连续点击预测导致图表重叠
    plt.clf()
