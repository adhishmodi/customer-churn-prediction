import sys
sys.path.insert(0, 'D:/customer-churn-prediction')
from src import load_raw_data
import pandas as pd

df = load_raw_data()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.loc[df['tenure'] == 0, 'TotalCharges'] = df.loc[df['tenure'] == 0, 'TotalCharges'].fillna(0)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
df['ChurnBinary'] = df['Churn'].map({'No': 0, 'Yes': 1})

print('=== KEY FINDINGS ===')
print()

# Churn distribution
print('1. CHURN DISTRIBUTION:')
churn_counts = df['Churn'].value_counts()
no_count = churn_counts['No']
yes_count = churn_counts['Yes']
print('   No: {} ({:.1f}%)'.format(no_count, no_count/len(df)*100))
print('   Yes: {} ({:.1f}%)'.format(yes_count, yes_count/len(df)*100))
print()

# Demographics
print('2. DEMOGRAPHICS - Churn Rate:')
for col in ['gender', 'SeniorCitizen', 'Partner', 'Dependents']:
    rate = df.groupby(col)['ChurnBinary'].mean()
    print('   {}: {}'.format(col, rate.to_dict()))
print()

# Tenure
print('3. TENURE:')
print('   Overall mean: {:.1f} months'.format(df['tenure'].mean()))
print('   Churn=No mean: {:.1f} months'.format(df[df['Churn']=='No']['tenure'].mean()))
print('   Churn=Yes mean: {:.1f} months'.format(df[df['Churn']=='Yes']['tenure'].mean()))
print()

# Contract
print('4. CONTRACT TYPE - Churn Rate:')
contract_churn = df.groupby('Contract')['ChurnBinary'].mean().sort_values(ascending=False)
for k, v in contract_churn.items():
    print('   {}: {:.1f}%'.format(k, v*100))
print()

# Payment Method
print('5. PAYMENT METHOD - Churn Rate:')
payment_churn = df.groupby('PaymentMethod')['ChurnBinary'].mean().sort_values(ascending=False)
for k, v in payment_churn.items():
    print('   {}: {:.1f}%'.format(k, v*100))
print()

# Charges
print('6. CHARGES:')
print('   MonthlyCharges mean: ${:.2f}'.format(df['MonthlyCharges'].mean()))
print('   TotalCharges mean: ${:.2f}'.format(df['TotalCharges'].mean()))
print('   MonthlyCharges by Churn:')
for churn_val in ['No', 'Yes']:
    subset = df[df['Churn'] == churn_val]['MonthlyCharges']
    print('     {}: mean=${:.2f}, median=${:.2f}'.format(churn_val, subset.mean(), subset.median()))
print('   Correlation Monthly vs Total: {:.3f}'.format(df['MonthlyCharges'].corr(df['TotalCharges'])))
print()

# Internet Service
print('7. INTERNET SERVICE - Churn Rate:')
is_churn = df.groupby('InternetService')['ChurnBinary'].mean().sort_values(ascending=False)
for k, v in is_churn.items():
    print('   {}: {:.1f}%'.format(k, v*100))
print()

# Online Security
print('8. ONLINE SECURITY - Churn Rate:')
os_churn = df.groupby('OnlineSecurity')['ChurnBinary'].mean().sort_values(ascending=False)
for k, v in os_churn.items():
    print('   {}: {:.1f}%'.format(k, v*100))
print()

# Tech Support
print('9. TECH SUPPORT - Churn Rate:')
ts_churn = df.groupby('TechSupport')['ChurnBinary'].mean().sort_values(ascending=False)
for k, v in ts_churn.items():
    print('   {}: {:.1f}%'.format(k, v*100))
print()

# Paperless Billing
print('10. PAPERLESS BILLING - Churn Rate:')
pb_churn = df.groupby('PaperlessBilling')['ChurnBinary'].mean()
for k, v in pb_churn.items():
    print('   {}: {:.1f}%'.format(k, v*100))
print()

# Correlation
print('11. KEY CORRELATIONS (with ChurnBinary):')
numeric_df = df.select_dtypes(include=['number']).drop(columns=['customerID'], errors='ignore')
corr_with_churn = numeric_df.corr()['ChurnBinary'].sort_values(key=abs, ascending=False)
for k, v in corr_with_churn.items():
    if k != 'ChurnBinary':
        print('   {}: {:.3f}'.format(k, v))
