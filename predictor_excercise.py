# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:45:56 2026

@author: LENOVO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')
import shap
import joblib

######################## 1. 基础配置 ########################
# 加载训练好的最佳模型
model = joblib.load('xgb_model.pkl')

# 加载测试数据(用于LIME解释器，确保x_test.csv与脚本同目录) 
x_test = pd.read_csv('valdata.csv')

#定义特征名称(替换为业务相关列名，与编码规则对应) 
feature_names =['male','BPMeds','prevalentStroke','age','prevalentHyp','diabetes','cigsPerDay','sysBP','glucose','totChol','diaBP','BMI']

######################## 2. Streamlit页面配置 ########################
st.set_page_config(page_title="心脏病风险预测器", layout="wide") 
st.title("心脏病风险概率预测器")
st.markdown("请填写以下信息，点击预测获取心脏病风险评估结果")

######################## 3，特征输入组件(按编码规则设计) ########################
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
    min_value=0,     # 允许输入的最小值
    max_value=120,   # 允许输入的最大值
    value=30,        # 默认显示的数值
    step=1           # 每次加减的步长（整数）
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
    min_value=0,     # 允许输入的最小值
    max_value=120,   # 允许输入的最大值
    value=0,        # 默认显示的数值
    step=1           # 每次加减的步长（整数）
)

# 8. sysBP (连续变量：收缩压)
sysBP = st.number_input(
    "您的收缩压为:",
    min_value=0.0,     # 允许输入的最小值
    max_value=300.0,   # 允许输入的最大值
    value=120.0,        # 默认显示的数值
    step=0.1          # 每次加减的步长（整数）
)

# 9. glucose (连续变量：血糖)
glucose = st.number_input(
    "您的血糖浓度为 (mmol/L):",
    min_value=0,     # 允许输入的最小值
    max_value=300,   # 允许输入的最大值
    value=120,        # 默认显示的数值
    step=1           # 每次加减的步长（整数）
)

# 10. totChol (连续变量：总胆固醇)
totChol = st.number_input(
    "您的总胆固醇为 (mmol/L):",
    min_value=0,     # 允许输入的最小值
    max_value=1000,   # 允许输入的最大值
    value=120,        # 默认显示的数值
    step=1           # 每次加减的步长（整数）
)

# 11. diaBP (连续变量：舒张压)
diaBP = st.number_input(
    "您的舒张压为:",
    min_value=0.0,     # 允许输入的最小值
    max_value=300.0,   # 允许输入的最大值
    value=120.0,        # 默认显示的数值
    step=0.1          # 每次加减的步长（整数）
)

# 12. BMI (连续变量：BMI)
BMI = st.number_input(
    "您的BMI为:",
    min_value=0.00,     # 允许输入的最小值
    max_value=60.00,   # 允许输入的最大值
    value=20.00,        # 默认显示的数值
    step=0.01          # 每次加减的步长（整数）
)

######################## 4，数据处理与预测 ########################
# 整合用户特征
feature_values = [
    male,BPMeds,prevalentStroke,age,prevalentHyp,diabetes,cigsPerDay,sysBP,glucose,totChol,diaBP,BMI]    

# 转换为模型输入格式
features = np.array([feature_values])

# 预测按钮逻辑
if st.button("预测"):
    # 模型预测
    predicted_class = model.predict(features)[0] #0：低风险，1：高风险 
    predicted_proba = model.predict_proba(features)[0] #概率值

    # 显示预测结果
    st.subheader("预测结果")
    risk_label = "高风险" if predicted_class == 1 else "低风险" 
    st.write(f'**风险等级：{predicted_class}({risk_label})**')
    st.write(f'**风险概率，**低风险概率{predicted_proba[0]:,2%}|高风险概率{predicted_proba[1]:.2%}')
    
    # 生成个性化建议
    st.subheader("健康建议")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1: 
        advice = (
            f"模型预测您的心脏病风险险为高风险(概率{probability:.1f}%)。"
            "建议尽快前往医疗机构进行全面的心脏评估"
            "同时可根据自身情况增加适宜的体育锻炼，改善生活环境。"
            )

######################## 5.LIME	(适配业务特征) ########################
st.subheader("LIME特征贡献解释")	
lime_explainer = LimeTabularExplainer(
    training_data = x_test.values,
    feature_names = feature_names,
    class_names=['低心脏病风险','高心脏病风险'], #适配业务类别 
    mode='classification')

# 生成LIME解释
lime_exp = lime_explainer.explain_instance(
    data_row = features.flatten(), 
    predict_fn = model.predict_proba,
    num_features=10 #显示前10个重要特征
    )

#显示LIME解释(HTML格式)
lime_html = lime_exp.as_html(show_table=True)
st.components.v1.html(lime_html, height=600, scrolling=True)
