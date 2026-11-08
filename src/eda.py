"""Exploratory Data Analysis for Customer Churn Prediction."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .config import REPORTS_DIR, TARGET_COLUMN, ID_COLUMN, RANDOM_SEED
from .data_loader import load_raw_data, get_data_summary

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

FIGURES_DIR = REPORTS_DIR / "figures"


def load_and_prepare_data() -> pd.DataFrame:
    """Load raw data and prepare for EDA (convert TotalCharges to numeric)."""
    df = load_raw_data()
    
    # Convert TotalCharges to numeric (blank strings become NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Only zero-tenure blank charges have a deterministic zero value.
    # Leave any other missing values visible; modeling imputes them inside CV folds.
    mask_tenure_zero = df['tenure'] == 0
    df.loc[mask_tenure_zero, 'TotalCharges'] = df.loc[mask_tenure_zero, 'TotalCharges'].fillna(0)
    
    # Convert SeniorCitizen to categorical for better plotting.
    df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})
    
    # Create binary target for easier analysis
    df['ChurnBinary'] = df[TARGET_COLUMN].map({'No': 0, 'Yes': 1})
    
    return df


def save_figure(fig, name: str, dpi: int = 150):
    """Save figure to reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_churn_distribution(df: pd.DataFrame):
    """Plot churn class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    churn_counts = df[TARGET_COLUMN].value_counts()
    axes[0].bar(churn_counts.index, churn_counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[0].set_title('Churn Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Customers')
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 50, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    
    axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90, textprops={'fontsize': 12})
    axes[1].set_title('Churn Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, '01_churn_distribution')
    return churn_counts.to_dict()


def plot_demographics(df: pd.DataFrame):
    """Plot churn by customer demographics."""
    demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(demo_cols):
        if col in df.columns:
            churn_rate = df.groupby(col)['ChurnBinary'].mean().sort_values(ascending=False)
            ax = axes[i]
            bars = ax.bar(churn_rate.index.astype(str), churn_rate.values * 100, 
                          color='#e74c3c', edgecolor='black', alpha=0.8)
            ax.set_title(f'Churn Rate by {col}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_ylim(0, max(churn_rate.values * 100) * 1.2)
            for bar, rate in zip(bars, churn_rate.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, '02_demographics_churn')
    
    return {col: df.groupby(col)['ChurnBinary'].mean().to_dict() for col in demo_cols if col in df.columns}


def plot_tenure_analysis(df: pd.DataFrame):
    """Analyze tenure distribution and its relationship with churn."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df['tenure'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Tenure Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Tenure (months)')
    axes[0, 0].set_ylabel('Count')
    
    sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[0, 1], palette=['#2ecc71', '#e74c3c'])
    axes[0, 1].set_title('Tenure by Churn Status', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Churn')
    axes[0, 1].set_ylabel('Tenure (months)')
    
    for churn_val, color in [('No', '#2ecc71'), ('Yes', '#e74c3c')]:
        subset = df[df['Churn'] == churn_val]['tenure']
        axes[1, 0].hist(subset, bins=20, alpha=0.5, label=churn_val, color=color, edgecolor='black')
    axes[1, 0].set_title('Tenure Distribution by Churn', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Tenure (months)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    df['tenure_bin'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                               labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
    tenure_churn = df.groupby('tenure_bin', observed=True)['ChurnBinary'].mean()
    axes[1, 1].bar(range(len(tenure_churn)), tenure_churn.values * 100, 
                   color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1, 1].set_xticks(range(len(tenure_churn)))
    axes[1, 1].set_xticklabels(tenure_churn.index.astype(str), rotation=45)
    axes[1, 1].set_title('Churn Rate by Tenure Group', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Churn Rate (%)')
    axes[1, 1].set_xlabel('Tenure (months)')
    for i, v in enumerate(tenure_churn.values):
        axes[1, 1].text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, '03_tenure_analysis')
    
    df.drop(columns=['tenure_bin'], inplace=True)
    
    return {
        'tenure_stats': df['tenure'].describe().to_dict(),
        'tenure_by_churn': df.groupby('Churn')['tenure'].describe().to_dict(),
        'churn_rate_by_tenure_bin': tenure_churn.to_dict()
    }


def plot_contract_payment(df: pd.DataFrame):
    """Analyze contract type, payment method, and churn."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    contract_counts = df['Contract'].value_counts()
    axes[0, 0].bar(contract_counts.index, contract_counts.values, color='#3498db', edgecolor='black')
    axes[0, 0].set_title('Contract Type Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(contract_counts.values):
        axes[0, 0].text(i, v + 20, str(v), ha='center', fontweight='bold')
    
    contract_churn = df.groupby('Contract')['ChurnBinary'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(contract_churn)), contract_churn.values * 100, 
                   color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[0, 1].set_xticks(range(len(contract_churn)))
    axes[0, 1].set_xticklabels(contract_churn.index, rotation=45, ha='right')
    axes[0, 1].set_title('Churn Rate by Contract Type', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(contract_churn.values):
        axes[0, 1].text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold')
    
    payment_counts = df['PaymentMethod'].value_counts()
    axes[1, 0].bar(range(len(payment_counts)), payment_counts.values, color='#9b59b6', edgecolor='black')
    axes[1, 0].set_xticks(range(len(payment_counts)))
    axes[1, 0].set_xticklabels(payment_counts.index, rotation=45, ha='right')
    axes[1, 0].set_title('Payment Method Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Count')
    for i, v in enumerate(payment_counts.values):
        axes[1, 0].text(i, v + 20, str(v), ha='center', fontweight='bold')
    
    payment_churn = df.groupby('PaymentMethod')['ChurnBinary'].mean().sort_values(ascending=False)
    axes[1, 1].bar(range(len(payment_churn)), payment_churn.values * 100, 
                   color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1, 1].set_xticks(range(len(payment_churn)))
    axes[1, 1].set_xticklabels(payment_churn.index, rotation=45, ha='right')
    axes[1, 1].set_title('Churn Rate by Payment Method', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(payment_churn.values):
        axes[1, 1].text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, '04_contract_payment_analysis')
    
    return {
        'contract_distribution': contract_counts.to_dict(),
        'churn_by_contract': contract_churn.to_dict(),
        'payment_distribution': payment_counts.to_dict(),
        'churn_by_payment': payment_churn.to_dict()
    }


def plot_charges_analysis(df: pd.DataFrame):
    """Analyze MonthlyCharges and TotalCharges."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    axes[0, 0].hist(df['MonthlyCharges'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Monthly Charges Distribution', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Monthly Charges ($)')
    axes[0, 0].set_ylabel('Count')
    
    axes[0, 1].hist(df['TotalCharges'], bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Total Charges Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Total Charges ($)')
    axes[0, 1].set_ylabel('Count')
    
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[0, 2], palette=['#2ecc71', '#e74c3c'])
    axes[0, 2].set_title('Monthly Charges by Churn', fontsize=11, fontweight='bold')
    
    sns.boxplot(data=df, x='Churn', y='TotalCharges', ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])
    axes[1, 0].set_title('Total Charges by Churn', fontsize=11, fontweight='bold')
    
    for churn_val, color in [('No', '#2ecc71'), ('Yes', '#e74c3c')]:
        subset = df[df['Churn'] == churn_val]
        axes[1, 1].scatter(subset['MonthlyCharges'], subset['TotalCharges'], 
                           alpha=0.3, label=churn_val, color=color, s=10)
    axes[1, 1].set_title('Monthly vs Total Charges', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Monthly Charges ($)')
    axes[1, 1].set_ylabel('Total Charges ($)')
    axes[1, 1].legend()
    
    df['monthly_bin'] = pd.qcut(df['MonthlyCharges'], q=5, duplicates='drop')
    monthly_churn = df.groupby('monthly_bin', observed=True)['ChurnBinary'].mean()
    axes[1, 2].bar(range(len(monthly_churn)), monthly_churn.values * 100, 
                   color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1, 2].set_xticks(range(len(monthly_churn)))
    axes[1, 2].set_xticklabels([str(i) for i in monthly_churn.index], rotation=45, ha='right')
    axes[1, 2].set_title('Churn Rate by Monthly Charges Quintile', fontsize=11, fontweight='bold')
    axes[1, 2].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(monthly_churn.values):
        axes[1, 2].text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, '05_charges_analysis')
    
    df.drop(columns=['monthly_bin'], inplace=True)
    
    return {
        'monthly_charges_stats': df['MonthlyCharges'].describe().to_dict(),
        'total_charges_stats': df['TotalCharges'].describe().to_dict(),
        'monthly_by_churn': df.groupby('Churn')['MonthlyCharges'].describe().to_dict(),
        'total_by_churn': df.groupby('Churn')['TotalCharges'].describe().to_dict(),
        'correlation_monthly_total': df['MonthlyCharges'].corr(df['TotalCharges'])
    }


def plot_internet_phone_services(df: pd.DataFrame):
    """Analyze internet and phone services."""
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()
    
    churn_rates = {}
    for i, col in enumerate(service_cols):
        if col in df.columns:
            rate = df.groupby(col)['ChurnBinary'].mean().sort_values(ascending=False)
            churn_rates[col] = rate.to_dict()
            ax = axes[i]
            bars = ax.bar(range(len(rate)), rate.values * 100, color='#e74c3c', edgecolor='black', alpha=0.8)
            ax.set_xticks(range(len(rate)))
            ax.set_xticklabels(rate.index.astype(str), rotation=45, ha='right', fontsize=8)
            ax.set_title(f'Churn Rate by {col}', fontsize=10, fontweight='bold')
            ax.set_ylabel('Churn Rate (%)')
            for bar, val in zip(bars, rate.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, '06_internet_phone_services')
    
    return churn_rates


def plot_paperless_billing(df: pd.DataFrame):
    """Analyze paperless billing."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    pb_counts = df['PaperlessBilling'].value_counts()
    axes[0].bar(pb_counts.index, pb_counts.values, color='#3498db', edgecolor='black')
    axes[0].set_title('Paperless Billing Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(pb_counts.values):
        axes[0].text(i, v + 20, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    
    pb_churn = df.groupby('PaperlessBilling')['ChurnBinary'].mean()
    axes[1].bar(range(len(pb_churn)), pb_churn.values * 100, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1].set_xticks(range(len(pb_churn)))
    axes[1].set_xticklabels(pb_churn.index)
    axes[1].set_title('Churn Rate by Paperless Billing', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(pb_churn.values):
        axes[1].text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, '07_paperless_billing')
    
    return {
        'distribution': pb_counts.to_dict(),
        'churn_rate': pb_churn.to_dict()
    }


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.drop(columns=[ID_COLUMN], errors='ignore')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Numeric Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '08_correlation_heatmap')
    
    return corr.to_dict()


def plot_churn_by_internet_service_detail(df: pd.DataFrame):
    """Detailed churn analysis by InternetService and related services."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    is_churn = df.groupby('InternetService')['ChurnBinary'].mean().sort_values(ascending=False)
    axes[0, 0].bar(range(len(is_churn)), is_churn.values * 100, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[0, 0].set_xticks(range(len(is_churn)))
    axes[0, 0].set_xticklabels(is_churn.index, rotation=45, ha='right')
    axes[0, 0].set_title('Churn Rate by Internet Service', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(is_churn.values):
        axes[0, 0].text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold')
    
    os_churn = df.groupby('OnlineSecurity')['ChurnBinary'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(os_churn)), os_churn.values * 100, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[0, 1].set_xticks(range(len(os_churn)))
    axes[0, 1].set_xticklabels(os_churn.index, rotation=45, ha='right')
    axes[0, 1].set_title('Churn Rate by Online Security', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(os_churn.values):
        axes[0, 1].text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold')
    
    ts_churn = df.groupby('TechSupport')['ChurnBinary'].mean().sort_values(ascending=False)
    axes[1, 0].bar(range(len(ts_churn)), ts_churn.values * 100, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1, 0].set_xticks(range(len(ts_churn)))
    axes[1, 0].set_xticklabels(ts_churn.index, rotation=45, ha='right')
    axes[1, 0].set_title('Churn Rate by Tech Support', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(ts_churn.values):
        axes[1, 0].text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold')
    
    df['Contract_Internet'] = df['Contract'] + ' / ' + df['InternetService']
    ci_churn = df.groupby('Contract_Internet')['ChurnBinary'].mean().sort_values(ascending=False)
    axes[1, 1].bar(range(len(ci_churn)), ci_churn.values * 100, color='#e74c3c', edgecolor='black', alpha=0.8)
    axes[1, 1].set_xticks(range(len(ci_churn)))
    axes[1, 1].set_xticklabels(ci_churn.index, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_title('Churn Rate by Contract + Internet Service', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Churn Rate (%)')
    for i, v in enumerate(ci_churn.values):
        axes[1, 1].text(i, v * 100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, '09_internet_service_detail')
    
    df.drop(columns=['Contract_Internet'], inplace=True)
    
    return {
        'churn_by_internet_service': is_churn.to_dict(),
        'churn_by_online_security': os_churn.to_dict(),
        'churn_by_tech_support': ts_churn.to_dict(),
        'churn_by_contract_internet': ci_churn.to_dict()
    }


def run_eda():
    """Run complete EDA and generate all visualizations."""
    print("=" * 60)
    print("STARTING EXPLORATORY DATA ANALYSIS (BATCH 2)")
    print("=" * 60)
    
    df = load_and_prepare_data()
    print(f"\nData loaded and prepared: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"TotalCharges missing after conversion: {df['TotalCharges'].isna().sum()}")
    
    results = {}
    
    print("\n1. Churn Distribution...")
    results['churn_distribution'] = plot_churn_distribution(df)
    
    print("2. Demographics Analysis...")
    results['demographics'] = plot_demographics(df)
    
    print("3. Tenure Analysis...")
    results['tenure'] = plot_tenure_analysis(df)
    
    print("4. Contract & Payment Analysis...")
    results['contract_payment'] = plot_contract_payment(df)
    
    print("5. Charges Analysis...")
    results['charges'] = plot_charges_analysis(df)
    
    print("6. Internet & Phone Services...")
    results['services'] = plot_internet_phone_services(df)
    
    print("7. Paperless Billing...")
    results['paperless'] = plot_paperless_billing(df)
    
    print("8. Correlation Heatmap...")
    results['correlation'] = plot_correlation_heatmap(df)
    
    print("9. Internet Service Detail...")
    results['internet_detail'] = plot_churn_by_internet_service_detail(df)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE - All figures saved to reports/figures/")
    print("=" * 60)
    
    return results, df


if __name__ == "__main__":
    results, df = run_eda()
