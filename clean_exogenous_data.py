#!/usr/bin/env python3
"""
Data Cleaning Script for Exogenous Variables Dataset
Labor Market Participation Study - Wave 2

This script cleans and processes the exogenous variables dataset by:
1. Handling missing values
2. Encoding categorical variables
3. Parsing and standardizing numeric variables
4. Creating derived variables (e.g., BMI)
5. Generating a data dictionary

Author: Data Processing Pipeline
Date: 2025-11-25
"""

import pandas as pd
import numpy as np
import re


def clean_risk_tolerance(series):
    """
    Clean Q84 (risk tolerance) - extract numeric values from mixed format
    Range: 0-10 where 0 = not willing to take risks, 10 = very willing
    """
    def extract_number(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        # Extract first number from string
        match = re.search(r'\d+', val_str)
        if match:
            return float(match.group())
        return np.nan

    return series.apply(extract_number)


def encode_respondent_work_hours(series):
    """
    Encode Q179 (respondent's work hours) to numeric ranges
    Returns midpoint of hour range for analysis
    """
    mapping = {
        '0 Stunden': 0,
        '1-10 Stunden': 5.5,
        '11-20 Stunden': 15.5,
        '21-30 Stunden': 25.5,
        '31-40 Stunden': 35.5,
        '41-50 Stunden': 45.5,
        'Mehr als 50 Stunden': 55,
        'Nicht erwerbstätig': 0
    }
    return series.map(mapping)


def encode_gender(series):
    """
    Encode Q256 (gender) - keep as categorical but standardize
    """
    mapping = {
        'Männlich': 'Male',
        'Weiblich': 'Female',
        'Divers': 'Diverse'
    }
    return series.map(mapping)


def encode_education(series):
    """
    Encode Q190 (highest school certificate) to ordinal scale
    1 = No certificate, 5 = Abitur/University entrance
    """
    mapping = {
        'Ohne allgemeinen Schulabschluss': 1,
        'Haupt- oder Volksschulabschluss (Abschluss der Pflichtschule)': 2,
        'Abschluss der Polytechnischen Oberschule der DDR': 3,
        'Mittleren Schulabschluss (z.B. Realschulabschluss)': 3,
        'Abitur oder Fachabitur (Höchster Schulabschluss/ Hochschulreife)': 4
    }
    return series.map(mapping)


def encode_health_status(series):
    """
    Encode Q211 (health status) to ordinal scale
    1 = Poor, 5 = Very good
    """
    mapping = {
        'Schlecht': 1,
        'Weniger gut': 2,
        'Zufriedenstellend': 3,
        'Gut': 4,
        'Sehr gut': 5
    }
    return series.map(mapping)


def encode_change_variables(series):
    """
    Encode change variables (Q219_3, Q219_4) to ordinal scale
    -2 = strongly decreased, 0 = no change, +2 = strongly increased
    """
    mapping = {
        'stark abgenommen': -2,
        'leicht abgenommen': -1,
        'sich nicht verändert': 0,
        'leicht zugenommen': 1,
        'stark zugenommen': 2
    }
    return series.map(mapping)


def parse_income_range(series):
    """
    Parse income ranges (Q86, Q87) and return midpoint in euros
    e.g., '1001-1500€' -> 1250.5
    """
    def extract_midpoint(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        # Extract numbers from range like '1001-1500€'
        numbers = re.findall(r'\d+', val_str)
        if len(numbers) >= 2:
            return (float(numbers[0]) + float(numbers[1])) / 2
        elif len(numbers) == 1:
            return float(numbers[0])
        return np.nan

    return series.apply(extract_midpoint)


def calculate_bmi(height_cm, weight_kg):
    """
    Calculate BMI from height (cm) and weight (kg)
    BMI = weight / (height_m)^2
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return bmi


def clean_exogenous_dataset(input_file, output_file, original_data_file='Data_Wave2 (1).csv'):
    """
    Main cleaning function - processes the entire dataset
    """
    print("=" * 80)
    print("DATA CLEANING PIPELINE")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(input_file)
    print(f"   Loaded {df.shape[0]} observations, {df.shape[1]} variables")

    # Load Q1 from original dataset to handle missing values properly
    print("   Loading Q1 (living situation) from original dataset...")
    df_original = pd.read_csv(original_data_file)
    df_original_data = df_original.iloc[2:].reset_index(drop=True)
    q1_data = df_original_data['Q1']

    # Create cleaned dataframe
    df_clean = pd.DataFrame()

    # Keep ID
    print("\n2. Preserving ID variable...")
    df_clean['ResponseId'] = df['ResponseId']

    # Add living situation indicator
    print("\n3. Adding living situation indicator (Q1)...")
    df_clean['lives_with_others'] = q1_data.map({'Ja': 'Yes', 'Nein': 'No'})

    # Clean respondent work hours
    print("\n4. Encoding respondent work hours (Q179)...")
    df_clean['respondent_work_hours'] = encode_respondent_work_hours(df['Q179'])
    df_clean['respondent_work_hours_cat'] = df['Q179']  # Keep original categories

    # Clean household composition
    print("\n5. Processing household composition variables...")
    df_clean['num_children'] = df['Q4_4']
    df_clean['num_partners'] = df['Q4_2']
    df_clean['num_parents'] = df['Q4_3']
    df_clean['num_siblings'] = df['Q4_5']

    # Fix missing values for people living alone
    print("   Fixing missing values: People living alone have 0 household members...")
    lives_alone = q1_data == 'Nein'
    df_clean.loc[lives_alone, 'num_children'] = df_clean.loc[lives_alone, 'num_children'].fillna(0)
    df_clean.loc[lives_alone, 'num_partners'] = df_clean.loc[lives_alone, 'num_partners'].fillna(0)
    df_clean.loc[lives_alone, 'num_parents'] = df_clean.loc[lives_alone, 'num_parents'].fillna(0)
    df_clean.loc[lives_alone, 'num_siblings'] = df_clean.loc[lives_alone, 'num_siblings'].fillna(0)

    before_fix = df['Q4_4'].isnull().sum()
    after_fix = df_clean['num_children'].isnull().sum()
    print(f"   Missing values reduced: {before_fix} → {after_fix} ({after_fix/len(df_clean)*100:.1f}%)")

    # Clean demographics
    print("\n6. Processing demographic variables...")
    df_clean['age'] = df['Q80_1']
    df_clean['gender'] = encode_gender(df['Q256'])
    df_clean['gender_code'] = df['Q256'].map({'Männlich': 1, 'Weiblich': 2, 'Divers': 3})
    df_clean['education_level'] = encode_education(df['Q190'])
    df_clean['education_cat'] = df['Q190']  # Keep original text
    df_clean['zip_code'] = df['Q120']

    # Clean vocational qualifications
    print("\n7. Processing vocational qualifications (Q191)...")
    df_clean['vocational_qualification'] = df['Q191']
    # Create binary indicator for university degree
    df_clean['has_university_degree'] = df['Q191'].str.contains(
        'Universität|Hochschule|Bachelor|Master|Diplom|Promotion',
        case=False,
        na=False
    ).astype(int)

    # Clean physical characteristics
    print("\n8. Processing physical characteristics...")
    df_clean['height_cm'] = df['Q82_1']
    df_clean['weight_kg'] = df['Q83_1']
    df_clean['bmi'] = calculate_bmi(df['Q82_1'], df['Q83_1'])

    # Clean risk tolerance
    print("\n9. Processing risk tolerance (Q84)...")
    df_clean['risk_tolerance'] = clean_risk_tolerance(df['Q84'])

    # Clean work/income change variables
    print("\n10. Processing work and income changes (Q219_3, Q219_4)...")
    df_clean['personal_income_change'] = encode_change_variables(df['Q219_3'])
    df_clean['partner_income_change'] = encode_change_variables(df['Q219_4'])

    # Clean household task division
    print("\n11. Processing household task division (Q243_*)...")
    df_clean['task_share_food'] = df['Q243_1']  # % share: food shopping & preparation
    df_clean['task_share_childcare'] = df['Q243_2']  # % share: childcare
    df_clean['task_share_education'] = df['Q243_3']  # % share: children's education
    df_clean['task_share_housework'] = df['Q243_9']  # % share: housework

    # Clean health status
    print("\n12. Processing health status (Q211)...")
    df_clean['health_status'] = encode_health_status(df['Q211'])
    df_clean['health_status_cat'] = df['Q211']  # Keep original text

    # Clean income variables
    print("\n13. Processing income variables (Q86, Q87)...")
    df_clean['household_income'] = parse_income_range(df['Q86'])
    df_clean['personal_income'] = parse_income_range(df['Q87'])
    df_clean['household_income_cat'] = df['Q86']  # Keep original categories
    df_clean['personal_income_cat'] = df['Q87']  # Keep original categories

    # Save cleaned data
    print(f"\n14. Saving cleaned data to {output_file}...")
    df_clean.to_csv(output_file, index=False)

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"\nOriginal dataset: {df.shape}")
    print(f"Cleaned dataset: {df_clean.shape}")
    print(f"\nMissing values by variable:")
    missing_pct = (df_clean.isnull().sum() / len(df_clean) * 100).round(1)
    for var, pct in missing_pct.items():
        if pct > 0:
            print(f"  {var:30s}: {pct:5.1f}%")

    # Generate data dictionary
    print("\n15. Generating data dictionary...")
    generate_data_dictionary(df_clean, output_file.replace('.csv', '_dictionary.txt'))

    print("\n" + "=" * 80)
    print("CLEANING COMPLETE!")
    print("=" * 80)

    return df_clean


def generate_data_dictionary(df, output_file):
    """
    Generate a data dictionary documenting all variables
    """
    dictionary = []
    dictionary.append("=" * 80)
    dictionary.append("DATA DICTIONARY - Cleaned Exogenous Variables Dataset")
    dictionary.append("=" * 80)
    dictionary.append("")

    var_descriptions = {
        'ResponseId': 'Unique survey response identifier',
        'lives_with_others': 'Lives with others in household (Yes/No)',
        'respondent_work_hours': 'Respondent\'s weekly work hours (numeric midpoint)',
        'respondent_work_hours_cat': 'Respondent\'s weekly work hours (categorical)',
        'num_children': 'Number of children in household (0 if living alone)',
        'num_partners': 'Number of partners in household (0 if living alone)',
        'num_parents': 'Number of parents in household (0 if living alone)',
        'num_siblings': 'Number of siblings in household (0 if living alone)',
        'age': 'Respondent\'s age in years',
        'gender': 'Gender (Male/Female/Diverse)',
        'gender_code': 'Gender code (1=Male, 2=Female, 3=Diverse)',
        'education_level': 'Education level (1=None to 4=Abitur)',
        'education_cat': 'Education level (categorical, German)',
        'zip_code': 'Zip code of residence',
        'vocational_qualification': 'Vocational qualification (text)',
        'has_university_degree': 'Has university degree (1=Yes, 0=No)',
        'height_cm': 'Height in centimeters',
        'weight_kg': 'Weight in kilograms',
        'bmi': 'Body Mass Index (calculated)',
        'risk_tolerance': 'Risk tolerance (0=not willing to 10=very willing)',
        'personal_income_change': 'Personal income change (-2=strongly decreased to +2=strongly increased)',
        'partner_income_change': 'Partner income change (-2=strongly decreased to +2=strongly increased)',
        'task_share_food': 'Personal share of food shopping/preparation (%)',
        'task_share_childcare': 'Personal share of childcare (%)',
        'task_share_education': 'Personal share of children\'s education (%)',
        'task_share_housework': 'Personal share of housework (%)',
        'health_status': 'Health status (1=Poor to 5=Very good)',
        'health_status_cat': 'Health status (categorical, German)',
        'household_income': 'Monthly household net income (euros, midpoint)',
        'personal_income': 'Monthly personal net income (euros, midpoint)',
        'household_income_cat': 'Monthly household net income (categorical)',
        'personal_income_cat': 'Monthly personal net income (categorical)'
    }

    for col in df.columns:
        dictionary.append(f"\n{col}")
        dictionary.append("-" * 40)
        dictionary.append(f"Description: {var_descriptions.get(col, 'No description')}")
        dictionary.append(f"Type: {df[col].dtype}")
        dictionary.append(f"Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
        dictionary.append(f"Non-missing: {df[col].notna().sum()}")

        if df[col].dtype in ['int64', 'float64']:
            dictionary.append(f"Mean: {df[col].mean():.2f}")
            dictionary.append(f"Std: {df[col].std():.2f}")
            dictionary.append(f"Min: {df[col].min()}")
            dictionary.append(f"Max: {df[col].max()}")
        elif df[col].dtype == 'object':
            dictionary.append(f"Unique values: {df[col].nunique()}")
            if df[col].nunique() <= 10:
                dictionary.append("Value counts:")
                for val, count in df[col].value_counts().items():
                    dictionary.append(f"  {val}: {count}")
        dictionary.append("")

    # Save dictionary
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(dictionary))

    print(f"   Data dictionary saved to {output_file}")


if __name__ == "__main__":
    # File paths
    input_file = "exogenous_variables_dataset.csv"
    output_file = "exogenous_variables_cleaned.csv"

    # Run cleaning
    df_cleaned = clean_exogenous_dataset(input_file, output_file)

    # Display first few rows
    print("\nFirst 5 rows of cleaned data:")
    print(df_cleaned.head())

    print("\nCleaned data shape:", df_cleaned.shape)
    print("\nCleaning complete! Files generated:")
    print(f"  - {output_file}")
    print(f"  - {output_file.replace('.csv', '_dictionary.txt')}")
