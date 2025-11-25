#!/usr/bin/env python3
"""
Regional Data Merging Script
Labor Market Participation Study - Wave 2

This script merges regional characteristics from Regions.csv with the
cleaned exogenous variables dataset by matching postal codes.

Author: Data Processing Pipeline
Date: 2025-11-25
"""

import pandas as pd
import numpy as np


def merge_regional_data(cleaned_file, regions_file, output_file):
    """
    Merge regional characteristics with cleaned dataset based on postal codes

    Parameters:
    -----------
    cleaned_file : str
        Path to cleaned exogenous variables dataset
    regions_file : str
        Path to regional characteristics dataset
    output_file : str
        Path for output merged dataset
    """
    print("=" * 80)
    print("REGIONAL DATA MERGING")
    print("=" * 80)

    # Load cleaned dataset
    print("\n1. Loading cleaned exogenous variables dataset...")
    df_clean = pd.read_csv(cleaned_file)
    print(f"   Loaded {df_clean.shape[0]} observations")
    print(f"   Postal codes: {df_clean['zip_code'].notna().sum()} non-missing")

    # Load regional data
    print("\n2. Loading regional characteristics dataset...")
    # Try different encodings for German characters
    try:
        df_regions = pd.read_csv(regions_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_regions = pd.read_csv(regions_file, encoding='latin-1')
        except UnicodeDecodeError:
            df_regions = pd.read_csv(regions_file, encoding='cp1252')
    print(f"   Loaded {df_regions.shape[0]} observations with regional data")

    # Check for encoding issues and clean column names
    print("\n3. Cleaning regional dataset...")
    # Remove empty columns
    df_regions = df_regions.loc[:, ~df_regions.columns.str.contains('^Unnamed')]
    df_regions = df_regions.dropna(axis=1, how='all')

    print(f"   Columns in regional dataset: {list(df_regions.columns)}")
    print(f"   Unique postal codes in regional data: {df_regions['PLZ'].nunique()}")

    # Aggregate regional data by postal code
    # Since multiple ResponseIds can have the same PLZ, we take the first occurrence
    # (all regional characteristics should be the same for the same PLZ)
    print("\n4. Aggregating regional data by postal code...")
    df_regions_agg = df_regions.groupby('PLZ').first().reset_index()
    print(f"   Unique postal codes after aggregation: {len(df_regions_agg)}")

    # Select relevant columns for merging (exclude ResponseId from regions)
    regional_cols = ['PLZ', 'Bundesland', 'Kreis', 'Stadt.Dummy',
                     'EW.km2', 'Rural.Dummy', 'EW', 'Metropol.Dummy']
    df_regions_merge = df_regions_agg[regional_cols].copy()

    # Rename columns for clarity
    print("\n5. Renaming regional variables...")
    df_regions_merge = df_regions_merge.rename(columns={
        'PLZ': 'region_plz',
        'Bundesland': 'federal_state',
        'Kreis': 'district',
        'Stadt.Dummy': 'is_city',
        'EW.km2': 'population_density',
        'Rural.Dummy': 'is_rural',
        'EW': 'population',
        'Metropol.Dummy': 'is_metropolitan'
    })

    # Merge with cleaned dataset
    print("\n6. Merging datasets on postal code...")
    print(f"   Matching zip_code (cleaned) with region_plz (regions)...")

    # Before merge statistics
    before_cols = df_clean.shape[1]

    # Perform left merge to keep all observations from cleaned dataset
    df_merged = df_clean.merge(
        df_regions_merge,
        left_on='zip_code',
        right_on='region_plz',
        how='left'
    )

    # Drop duplicate postal code column
    df_merged = df_merged.drop('region_plz', axis=1)

    # After merge statistics
    after_cols = df_merged.shape[1]
    print(f"   Variables added: {after_cols - before_cols}")
    print(f"   Total variables: {after_cols}")

    # Check merge success
    print("\n7. Checking merge results...")
    matched = df_merged['federal_state'].notna().sum()
    total_with_zip = df_clean['zip_code'].notna().sum()
    print(f"   Observations with postal code: {total_with_zip}")
    print(f"   Successfully matched to regional data: {matched}")
    print(f"   Match rate: {matched/total_with_zip*100:.1f}%")

    if matched < total_with_zip:
        unmatched = total_with_zip - matched
        print(f"   ⚠ Warning: {unmatched} postal codes could not be matched")
        # Show some unmatched postal codes
        unmatched_zips = df_merged.loc[
            df_merged['zip_code'].notna() & df_merged['federal_state'].isna(),
            'zip_code'
        ].unique()
        print(f"   Examples of unmatched postal codes: {unmatched_zips[:10].tolist()}")

    # Save merged dataset
    print(f"\n8. Saving merged dataset to {output_file}...")
    df_merged.to_csv(output_file, index=False)

    # Display summary
    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"\nOriginal cleaned dataset: {df_clean.shape}")
    print(f"Merged dataset: {df_merged.shape}")
    print(f"\nNew regional variables added:")
    new_vars = ['federal_state', 'district', 'is_city', 'population_density',
                'is_rural', 'population', 'is_metropolitan']
    for var in new_vars:
        missing = df_merged[var].isna().sum()
        missing_pct = missing / len(df_merged) * 100
        print(f"  {var:25s}: {missing:4d} missing ({missing_pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("MERGING COMPLETE!")
    print("=" * 80)

    return df_merged


def update_data_dictionary(df, output_file):
    """
    Update the data dictionary to include regional variables
    """
    dictionary = []
    dictionary.append("=" * 80)
    dictionary.append("DATA DICTIONARY - Cleaned Exogenous Variables with Regional Data")
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
        'personal_income_cat': 'Monthly personal net income (categorical)',
        'federal_state': 'Federal state (Bundesland) based on postal code',
        'district': 'District/county (Kreis) based on postal code',
        'is_city': 'City indicator (1=City, 0=Not city)',
        'population_density': 'Population density (inhabitants per km²)',
        'is_rural': 'Rural area indicator (1=Rural, 0=Not rural)',
        'population': 'Total population in the region',
        'is_metropolitan': 'Metropolitan area indicator (1=Metropolitan, 0=Not metropolitan)'
    }

    for col in df.columns:
        dictionary.append(f"\n{col}")
        dictionary.append("-" * 40)
        dictionary.append(f"Description: {var_descriptions.get(col, 'No description')}")
        dictionary.append(f"Type: {df[col].dtype}")
        dictionary.append(f"Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
        dictionary.append(f"Non-missing: {df[col].notna().sum()}")

        if df[col].dtype in ['int64', 'float64']:
            if df[col].notna().sum() > 0:
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

    print(f"\n9. Data dictionary updated: {output_file}")


if __name__ == "__main__":
    # File paths
    cleaned_file = "exogenous_variables_cleaned.csv"
    regions_file = "Regions.csv"
    output_file = "exogenous_variables_cleaned_with_regions.csv"
    dictionary_file = "exogenous_variables_cleaned_with_regions_dictionary.txt"

    # Merge regional data
    df_merged = merge_regional_data(cleaned_file, regions_file, output_file)

    # Update data dictionary
    update_data_dictionary(df_merged, dictionary_file)

    # Display sample
    print("\nFirst 5 rows with regional variables:")
    regional_vars = ['ResponseId', 'zip_code', 'federal_state', 'district',
                     'is_city', 'is_rural', 'is_metropolitan', 'population_density']
    print(df_merged[regional_vars].head())

    print(f"\nFiles generated:")
    print(f"  - {output_file}")
    print(f"  - {dictionary_file}")
