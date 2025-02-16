import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
import joblib


flag = True


def check_invalid_input_val(val):
    return val is None or str(val).strip() == "" or str(val).strip().lower().startswith("please")


def setup_ml_model():
    global flag
    if flag:
        try:
            model = joblib.load('mxgb_ziti.pkl')
            print("成功从文件加载模型。")
            return model
        except FileNotFoundError:
            print("未找到保存的模型文件，将重新训练模型。")
            flag = False

    if not flag:
        data = pd.read_csv('pocd_ziti.csv')
        X = data[['Maternal Weight', 'Gestational Week', 'ASA PS', 'Anesthesia Method']]
        y = data['Neonatal combined endpoint']

        model = xgb.XGBClassifier()
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        with st.spinner('正在进行模型训练...'):
            for train_index, val_index in kfold.split(X, y):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                accuracy = np.mean(y_pred == y_val)
                cv_scores.append(accuracy)
        avg_accuracy = np.mean(cv_scores)
        print(f"5折交叉验证的平均准确率: {avg_accuracy}")
        model.fit(X, y)

        joblib.dump(model, 'mxgb_ziti.pkl')
        print("模型已保存到文件 mxgb_ziti.pkl")

        return model


def render_ui(model):
    col1, col2 = st.columns(2)

    with col1:
        # 产妇体重输入框
        weight = st.number_input("产妇体重（Kg）", min_value=0, value=None, placeholder="Please Input Weight")
        # 孕周输入框
        gw = st.number_input("孕周（W）", min_value=0, value=None, placeholder="Please Input GW")

    with col2:
        # 麻醉方式输入框
        anesthesia_options = ["Please Select Anesthesia", "1", "2", "3"]
        anesthesia = st.selectbox("麻醉方式", anesthesia_options, index=0)
        # ASA分级输入框
        asa_options = ["Please Select ASA", "1", "2", "3"]
        asa = st.selectbox("ASA分级", asa_options, index=0)

    result_col, button_col = st.columns([3, 1])

    with result_col:
        result_placeholder = st.empty()

    if button_col.button("开始计算"):
        valid_flag = True
        for v in [weight, gw, asa, anesthesia]:
            if check_invalid_input_val(v):
                st.warning("请完整填写所有输入项！")
                valid_flag = False
                break

        if valid_flag:
            input_data = pd.DataFrame({
                'Maternal Weight': [weight],
                'Gestational Week': [gw],
                'ASA PS': [int(asa)],
                'Anesthesia Method': [int(anesthesia)]
            })
            prediction = model.predict(input_data)
            result_text = f"新生儿复合终点事件: {prediction[0]}"
            result_col.markdown(f"<p style='text-align: right; margin-top: 8px;'>{result_text}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    st.image("logo.jpg", width=300, use_column_width='never')
    st.markdown(
        """
        <div style="
            height: 2px;
            border: none;
            border-radius: 5px;
            background: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
        " />
        """,
        unsafe_allow_html=True
    )
    st.title("围术期新生儿复合终点事件风险预测模型——预测工具（基于XGBoost）")
    model = setup_ml_model()
    render_ui(model)
