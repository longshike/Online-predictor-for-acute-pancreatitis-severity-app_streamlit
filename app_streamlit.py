import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import shap
from PIL import Image

# 设置页面标题和布局
st.set_page_config(
    page_title="Prediction of Acute Pancreatitis Severity",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 加载模型和特征名称
@st.cache_resource
def load_model():
    model = joblib.load('pancreatitis_model.pkl')
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, feature_names


model, feature_names = load_model()

# 标题和说明
st.title("🏥 Prediction Model for the Severity of Acute Pancreatitis")
st.markdown("""
This application uses machine learning models to predict the severity of acute pancreatitis.  
Please enter the patient's clinical indicators in the input field below, then click the "Predict" button.
""")

# 创建侧边栏
st.sidebar.header("Regarding")
st.sidebar.info("""
This is an acute pancreatitis severity prediction tool based on the LightGBM machine learning model.  
The model uses the following clinical indicators for prediction:
- PT
- α-HBDH
- CRP
- Glu
- Hb
- Ca
- ALB
- APTT
- WBC
- CO₂-CP
- MCH
""")

st.sidebar.header("Instructions for Use")
st.sidebar.markdown("""
1. Enter the patient's clinical indicator values in the input field
2. Click the "Predict" button to obtain the prediction results
3. The results will display the predicted severity level and corresponding probability
""")

# 创建输入表单
st.header("Patient Clinical Indicators Input")

# 使用两列布局，使输入框更紧凑
col1, col2 = st.columns(2)

input_data = {}
with col1:
    st.subheader("Blood indicators")
    input_data['PT'] = st.number_input("PT (s)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
    input_data['α-HBDH'] = st.number_input("α-HBDH (U/L)", min_value=0.0, max_value=1000.0, value=150.0, step=1.0)
    input_data['CRP'] = st.number_input("CRP (mg/L)", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
    input_data['Glu'] = st.number_input("Glu (mmol/L)", min_value=0.0, max_value=50.0, value=5.5, step=0.1)
    input_data['Hb'] = st.number_input("Hb (g/L)", min_value=0.0, max_value=200.0, value=130.0, step=1.0)

with col2:
    st.subheader("Biochemical indicators")
    input_data['Ca'] = st.number_input("Ca (mmol/L)", min_value=0.0, max_value=5.0, value=2.2, step=0.1)
    input_data['ALB'] = st.number_input("ALB (g/L)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
    input_data['APTT'] = st.number_input("APTT (s)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    input_data['WBC'] = st.number_input("WBC (×10⁹/L)", min_value=0.0, max_value=50.0, value=7.0, step=0.1)
    input_data['CO₂-CP'] = st.number_input("CO₂-CP (mmol/L)", min_value=0.0, max_value=40.0, value=24.0, step=0.1)
    input_data['MCH'] = st.number_input("MCH (pg)", min_value=0.0, max_value=50.0, value=30.0, step=0.1)

# 预测按钮
if st.button("Predict", type="primary"):
    # 将输入数据转换为DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # 进行预测
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # 显示结果
    st.header("Prediction Results")

    if prediction == 1:
        st.error(f"## Prediction Result: Severe Acute Pancreatitis")
        st.warning("⚠️ Immediate intensive care measures are recommended.")
    else:
        st.success(f"## Prediction result: Mild acute pancreatitis")
        st.info("ℹ️ Recommend routine treatment and observation.")

    # 显示概率
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of severe ", f"{probability[1]:.2%}")
    with col2:
        st.metric("Probability of mild", f"{probability[0]:.2%}")

    # 显示SHAP值解释
    st.subheader("Prediction Interpretation")

    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # 获取基准值
    if isinstance(explainer.expected_value, list):
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    # 如果是二分类问题，取第二个类的SHAP值
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    # 创建瀑布图 - 更可靠的方法
    st.subheader("Feature Contribution (Waterfall Plot)")

    # 获取当前样本的SHAP值
    sample_shap_values = shap_values[0]

    # 计算特征贡献的排序
    feature_order = np.argsort(-np.abs(sample_shap_values))
    sorted_features = [feature_names[i] for i in feature_order]
    sorted_shap_values = [sample_shap_values[i] for i in feature_order]

    # 创建瀑布图数据
    cumulative = base_value
    waterfall_data = []
    for i, (feature, value) in enumerate(zip(sorted_features, sorted_shap_values)):
        waterfall_data.append({
            'feature': feature,
            'value': value,
            'cumulative': cumulative,
            'start': cumulative,
            'end': cumulative + value
        })
        cumulative += value

    # 创建瀑布图可视化
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制基准线
    ax.axhline(y=base_value, color='gray', linestyle='--', alpha=0.7)
    ax.text(-1, base_value, f'Base Value: {base_value:.3f}',
            verticalalignment='center', fontweight='bold')

    # 绘制每个特征的贡献
    colors = ['red' if val > 0 else 'blue' for val in sorted_shap_values]
    bars = ax.bar(range(len(sorted_features)), sorted_shap_values,
                  bottom=[d['start'] for d in waterfall_data],
                  color=colors, alpha=0.7)

    # 添加数值标签
    for i, (bar, data) in enumerate(zip(bars, waterfall_data)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_y() + height / 2.,
                f'{data["value"]:.3f}',
                ha='center', va='center', fontweight='bold')

    # 设置图表属性
    ax.set_xlabel('Features')
    ax.set_ylabel('SHAP Value')
    ax.set_title('Waterfall Plot of Feature Contributions')
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=45, ha='right')

    # 添加最终预测值线
    final_value = base_value + sum(sorted_shap_values)
    ax.axhline(y=final_value, color='green', linestyle='-', linewidth=2)
    ax.text(len(sorted_features), final_value, f'Final Prediction: {final_value:.3f}',
            verticalalignment='center', fontweight='bold', color='green')

    plt.tight_layout()
    st.pyplot(fig)

    # 显示特征重要性条形图
    st.subheader("Feature Contribution (Bar Chart)")
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sample_shap_values
    }).sort_values('shap_value', key=abs, ascending=False)

    # 创建水平条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x > 0 else 'blue' for x in shap_df['shap_value']]
    ax.barh(shap_df['feature'], shap_df['shap_value'], color=colors)
    ax.set_xlabel('SHAP values (Impact on predictions)')
    ax.set_title('Contribution of Each Feature to the Prediction Results')
    plt.tight_layout()
    st.pyplot(fig)


    # 显示输入的数据
    st.subheader("Input data")
    st.dataframe(input_df.style.highlight_max(axis=0, color='#fffd75'), use_container_width=True)

# 在底部添加一些额外信息
st.markdown("---")
st.caption("""
Note: This prediction tool is for reference only and cannot replace professional medical diagnosis.  
Actual diagnosis should be based on clinical presentation and a doctor's professional judgment.
""")