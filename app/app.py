from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import json

app = Flask(__name__)

# 加載模型和標準化器
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# 加載模擬數據
mydata = pd.read_excel('mydata_simulation.xlsx')
credit_data = pd.read_excel('CreditScore.xlsx')

def check_rules(user_data, loan_type):
    """
    檢查申請者是否符合政府規則。
    """
    with open('rules.json', 'r', encoding='utf-8') as file:
        rules = json.load(file)

    matched_rules = []
    for rule in rules:
        if rule["loan_type"] != loan_type:
            continue

        try:
            condition = rule["rule"]
            for key, value in user_data.items():
                condition = condition.replace(key, str(value))
            if eval(condition):
                matched_rules.append(rule["message"])
        except Exception as e:
            print(f"規則評估出錯：{rule['rule']}，錯誤信息：{e}")
    return matched_rules

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_auto', methods=['POST'])
def predict_auto():
    try:
        # 接收輸入數據
        name = request.form.get('name')
        id_number = request.form.get('id_number')
        loan_type = request.form.get('loan_type')
        loan_amount = float(request.form.get('loan_amount'))

        # 查詢 MyData 和徵信數據
        user_data_mydata = mydata[(mydata['Name'] == name) & (mydata['ID_Number'] == id_number)]
        user_data_credit = credit_data[(credit_data['Name'] == name) & (credit_data['ID_Number'] == id_number)]

        if user_data_mydata.empty or user_data_credit.empty:
            return render_template('result.html', decision="錯誤", reason="找不到匹配的資料", salary=None, credit_score=None, loan_amount=None, dti_ratio=None)

        # 提取必要數據
        monthly_salary = float(user_data_mydata['Income Information'].values[0])
        credit_score = float(user_data_credit['Credit Score'].values[0])
        assets_value = float(user_data_mydata['Asset Information'].values[0])
        employment_status_str = user_data_mydata['Employment Insurance Details'].values[0]

        # 就業狀態映射
        if 'Active' in employment_status_str:
            employment_status = 0  # Full-Time
        elif 'Inactive' in employment_status_str:
            employment_status = 2  # Unemployed
        else:
            employment_status = -1

        if employment_status == -1:
            raise ValueError(f"未知的就業狀態：{employment_status_str}")

        # 計算 DTI 比率
        dti_ratio = (loan_amount / (monthly_salary * 12)) * 100

        # 特徵標準化
        user_features = scaler.transform([[monthly_salary, credit_score, np.log1p(loan_amount), dti_ratio, employment_status, assets_value]])
        risk_prediction = model.predict(user_features)[0]
        risk_levels = {0: "低", 1: "中", 2: "高"}
        risk_level = risk_levels.get(risk_prediction, "未知")

        # 規則檢查
        user_data = {
            "salary": monthly_salary,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "DTI_ratio": dti_ratio,
            "employment_status": employment_status,
            "assets_value": assets_value
        }
        rule_results = check_rules(user_data, loan_type)

        return render_template(
            'result.html',
            decision=risk_level,
            reason="; ".join(rule_results) if rule_results else "符合政府與貸款規則。",
            salary=monthly_salary,
            credit_score=credit_score,
            loan_amount=loan_amount,
            dti_ratio=dti_ratio,
            name=name,
            id_number=id_number,
            loan_type=loan_type
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('result.html', decision="錯誤", reason="內部錯誤，請稍後再試。")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
