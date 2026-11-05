def electrode_likelihood_ratio_test(data, fband, base_vars, additional_vars, n_permutations=1000, 
                                   random_state=42, min_trials=10):
    """
    Perform permutation-corrected likelihood ratio test to assess whether adding 
    additional variables significantly improves model fit beyond a base model.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The chan_dfs_feedback dataframe containing TFR and behavioral data
    fband : str
        Frequency band to analyze (e.g., 'alpha', 'beta', 'hfa', 'high_theta', 'low_theta')
    base_vars : list
        List of variables in the base model (e.g., ['preward'])
    additional_vars : list
        List of variables to test for additional explanatory power (e.g., ['zrpe'])
    n_permutations : int, default=1000
        Number of permutations for z-scoring
    random_state : int, default=42
        Random seed for reproducibility
    min_trials : int, default=10
        Minimum number of trials required per electrode for analysis
        
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with likelihood ratio test results for each electrode
        
    Output Columns:
    ---------------
    - unique_elec_id: Electrode identifier
    - participant, region, channel, hemi: Electrode metadata
    - fband: Frequency band analyzed
    - n_trials: Number of trials used
    - base_vars: Variables in base model (comma-separated)
    - additional_vars: Variables tested for additional power (comma-separated)
    - base_r2: R² of base model
    - full_r2: R² of full model
    - lr_delta_r2: Difference in R² (full - base)
    - lr_z_score: Z-score of likelihood ratio against permutation distribution
    - lr_p_value: Uncorrected p-value (one-tailed)
    - lr_p_fdr: FDR-corrected p-value
    - {var}_beta: Beta coefficient for each additional variable
    
    Example:
    --------
    # Test if zrpe adds explanatory power beyond preward
    results = electrode_likelihood_ratio_test(
        data=chan_dfs_feedback,
        fband='hfa',
        base_vars=['preward'],
        additional_vars=['zrpe'],
        n_permutations=500,
        min_trials=15
    )
    
    # Filter for significant results
    sig_results = results[results['lr_p_fdr'] < 0.05]
    
    # Group by region to see which regions show improvement
    region_summary = results.groupby('region')[['lr_delta_r2', 'lr_z_score']].mean()
    """
    
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set random seed
    np.random.seed(random_state)
    
    # Filter data for specific frequency band
    data = data[data['fband'] == fband]
    
    if len(data) == 0:
        raise ValueError(f"No data found for frequency band '{fband}'")
    
    # Initialize results storage
    results = []
    
    # Get unique electrodes
    unique_electrodes = data['unique_elec_id'].unique()
    
    print(f"Analyzing {len(unique_electrodes)} electrodes for frequency band '{fband}'")
    print(f"Base model variables: {base_vars}")
    print(f"Additional variables: {additional_vars}")
    
    for i, elec_id in enumerate(unique_electrodes):
        if i % 50 == 0:  # Progress indicator
            print(f"Processing electrode {i+1}/{len(unique_electrodes)}")
            
        # Get data for this electrode
        elec_data = data[data['unique_elec_id'] == elec_id].copy()
        
        # Check minimum trial requirement
        if len(elec_data) < min_trials:
            continue
            
        # Prepare variables for regression
        y = elec_data['tfr'].values
        
        # Check for valid base variables
        valid_base_vars = []
        base_X_data = []
        
        for var in base_vars:
            if var in elec_data.columns:
                var_data = elec_data[var].values
                if np.sum(~np.isnan(var_data)) >= min_trials:
                    valid_base_vars.append(var)
                    base_X_data.append(var_data)
        
        # Check for valid additional variables
        valid_additional_vars = []
        additional_X_data = []
        
        for var in additional_vars:
            if var in elec_data.columns:
                var_data = elec_data[var].values
                if np.sum(~np.isnan(var_data)) >= min_trials:
                    valid_additional_vars.append(var)
                    additional_X_data.append(var_data)
        
        if len(valid_base_vars) == 0 or len(valid_additional_vars) == 0:
            continue
            
        # Stack variables
        base_X = np.column_stack(base_X_data)
        additional_X = np.column_stack(additional_X_data)
        full_X = np.column_stack([base_X, additional_X])
        
        # Remove rows with any NaN values
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(base_X), axis=1) | np.any(np.isnan(additional_X), axis=1))
        y_clean = y[valid_mask]
        base_X_clean = base_X[valid_mask]
        full_X_clean = full_X[valid_mask]
        
        if len(y_clean) < min_trials:
            continue
            
        # Perform actual regressions
        try:
            # Base model
            base_reg = LinearRegression().fit(base_X_clean, y_clean)
            base_r2 = base_reg.score(base_X_clean, y_clean)
            
            # Full model
            full_reg = LinearRegression().fit(full_X_clean, y_clean)
            full_r2 = full_reg.score(full_X_clean, y_clean)
            
            # Calculate actual likelihood ratio (R² difference)
            actual_lr = full_r2 - base_r2
            
            # Permutation testing
            permuted_lrs = []
            
            for perm in range(n_permutations):
                # Shuffle the dependent variable (TFR)
                y_shuffled = np.random.permutation(y_clean)
                
                # Fit base model on shuffled data
                base_reg_perm = LinearRegression().fit(base_X_clean, y_shuffled)
                base_r2_perm = base_reg_perm.score(base_X_clean, y_shuffled)
                
                # Fit full model on shuffled data
                full_reg_perm = LinearRegression().fit(full_X_clean, y_shuffled)
                full_r2_perm = full_reg_perm.score(full_X_clean, y_shuffled)
                
                # Calculate permuted likelihood ratio
                permuted_lr = full_r2_perm - base_r2_perm
                permuted_lrs.append(permuted_lr)
            
            # Convert to array for easier computation
            permuted_lrs = np.array(permuted_lrs)
            
            # Compute z-score for likelihood ratio
            perm_mean = np.mean(permuted_lrs)
            perm_std = np.std(permuted_lrs)
            
            if perm_std > 0:
                lr_z_score = (actual_lr - perm_mean) / perm_std
                # One-tailed p-value (testing if additional variables improve model)
                lr_p_value = 1 - stats.norm.cdf(lr_z_score)
            else:
                lr_z_score = 0
                lr_p_value = 1
            
            # Store results
            electrode_info = elec_data.iloc[0]  # Get electrode metadata
            
            result_row = {
                'unique_elec_id': elec_id,
                'participant': electrode_info['participant'],
                'region': electrode_info['region'],
                'channel': electrode_info['channel'],
                'hemi': electrode_info['hemi'],
                'fband': fband,
                'n_trials': len(y_clean),
                'base_vars': ', '.join(valid_base_vars),
                'additional_vars': ', '.join(valid_additional_vars),
                'base_r2': base_r2,
                'full_r2': full_r2,
                'lr_delta_r2': actual_lr,
                'lr_z_score': lr_z_score,
                'lr_p_value': lr_p_value,
            }
            
            # Add beta coefficients for additional variables
            additional_betas = full_reg.coef_[len(valid_base_vars):]
            for j, var in enumerate(valid_additional_vars):
                result_row[f'{var}_beta'] = additional_betas[j]
            
            results.append(result_row)
            
        except Exception as e:
            print(f"Error processing electrode {elec_id}: {e}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add FDR correction for multiple comparisons
    if len(results_df) > 0:
        from statsmodels.stats.multitest import multipletests
        
        # FDR correction for likelihood ratio p-values
        valid_p_values = results_df['lr_p_value'].dropna()
        if len(valid_p_values) > 0:
            _, p_corrected, _, _ = multipletests(valid_p_values, method='fdr_bh')
            results_df['lr_p_fdr'] = np.nan
            results_df.loc[valid_p_values.index, 'lr_p_fdr'] = p_corrected
        else:
            # If no valid p-values, create empty column
            results_df['lr_p_fdr'] = np.nan
    else:
        # If no results at all, create empty DataFrame with expected columns
        results_df = pd.DataFrame(columns=[
            'unique_elec_id', 'participant', 'region', 'channel', 'hemi', 'fband', 
            'n_trials', 'base_vars', 'additional_vars', 'base_r2', 'full_r2', 
            'lr_delta_r2', 'lr_z_score', 'lr_p_value', 'lr_p_fdr'
        ])
    
    print(f"Analysis complete. Results for {len(results_df)} electrodes.")
    return results_df


# Example usage in your notebook:
"""
# Test if zrpe adds explanatory power beyond preward for HFA
lr_results = electrode_likelihood_ratio_test(
    data=chan_dfs_feedback,
    fband='hfa',
    base_vars=['preward'],
    additional_vars=['zrpe'],
    n_permutations=500,
    min_trials=15
)

# View significant results
sig_electrodes = lr_results[lr_results['lr_p_fdr'] < 0.05]
print(f"Number of electrodes where zrpe adds significant explanatory power: {len(sig_electrodes)}")

# Summarize by region
region_summary = lr_results.groupby('region').agg({
    'lr_delta_r2': 'mean',
    'lr_z_score': 'mean',
    'lr_p_fdr': lambda x: (x < 0.05).sum()
}).rename(columns={'lr_p_fdr': 'n_significant'})
print("\nRegion Summary:")
print(region_summary.sort_values('lr_delta_r2', ascending=False))

# Test multiple additional variables at once
lr_results_multi = electrode_likelihood_ratio_test(
    data=chan_dfs_feedback,
    fband='beta',
    base_vars=['preward'],
    additional_vars=['zrpe', 'zQ'],
    n_permutations=500
)
"""

