import nbformat

cells = [
nbformat.v4.new_markdown_cell('# Customer Churn Prediction — Exploratory Data Analysis\n\nExplore the IBM Telco Customer Churn dataset before modeling.'),
nbformat.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom pathlib import Path\n\nDATA_PATH = Path('../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')\ndf = pd.read_csv(DATA_PATH)\nprint(f'Shape: {df.shape}')\ndisplay(df.head())\ndf.info()"),
nbformat.v4.new_markdown_cell('## Data quality checks\nCheck missing values, duplicates, and convert TotalCharges to numeric.'),
nbformat.v4.new_code_cell("display(df.isna().sum().sort_values(ascending=False))\nprint('Duplicate rows:', df.duplicated().sum())\ndf['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\nprint('TotalCharges missing after conversion:', df['TotalCharges'].isna().sum())\ndisplay(df[df['TotalCharges'].isna()][['customerID','tenure','MonthlyCharges','TotalCharges']])"),
nbformat.v4.new_markdown_cell('## Target balance'),
nbformat.v4.new_code_cell("display(df['Churn'].value_counts().to_frame('customers'))\nchurn_rate = df['Churn'].eq('Yes').mean() * 100\nprint('Churn rate: {:.2f}%'.format(churn_rate))\nsns.countplot(data=df, x='Churn')\nplt.title('Customer Churn Counts')\nplt.tight_layout()\nplt.show()"),
nbformat.v4.new_markdown_cell('## Numeric feature distributions'),
nbformat.v4.new_code_cell("num_cols = ['tenure','MonthlyCharges','TotalCharges']\ndisplay(df[num_cols].describe().T)\nfig, axes = plt.subplots(1, 3, figsize=(15,4))\nfor col, ax in zip(num_cols, axes):\n    sns.histplot(data=df, x=col, hue='Churn', kde=True, ax=ax, element='step')\n    ax.set_title('{} by churn'.format(col))\nplt.tight_layout()\nplt.show()"),
nbformat.v4.new_markdown_cell('## Churn by categorical features'),
nbformat.v4.new_code_cell("cat_cols = ['Contract','PaymentMethod','InternetService','TechSupport','OnlineSecurity']\nfor col in cat_cols:\n    rate = df.groupby(col)['Churn'].apply(lambda s: s.eq('Yes').mean()).sort_values(ascending=False)\n    display((rate*100).rename('Churn rate (%)').to_frame())\n    plt.figure(figsize=(9,4))\n    sns.barplot(x=rate.index.astype(str), y=rate.values*100)\n    plt.title('Churn rate by {}'.format(col))\n    plt.ylabel('Churn rate (%)')\n    plt.xticks(rotation=25, ha='right')\n    plt.tight_layout()\n    plt.show()"),
nbformat.v4.new_markdown_cell('## Numeric correlation'),
nbformat.v4.new_code_cell("plt.figure(figsize=(6,4))\nsns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)\nplt.title('Numeric Feature Correlation')\nplt.tight_layout()\nplt.show()"),
nbformat.v4.new_markdown_cell('## Initial modeling notes\n- Exclude customerID because it is an identifier.\n- Convert TotalCharges to numeric; blank strings become missing values.\n- Churn is the target. Consider class balance when selecting metrics.\n- Split data before fitting preprocessing to avoid data leakage.')
]
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
nbformat.write(nb, 'D:/customer-churn-prediction/notebooks/01_eda.ipynb')
print('Created notebooks/01_eda.ipynb')