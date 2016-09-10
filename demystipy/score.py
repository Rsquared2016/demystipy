from preppy.encoders import encode_df
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import itertools


def score_feature_groups(df):
    """
    Take a dataframe of components, where each row contains the explained 
    variance for each feature. Find the pairs for each feature with the largest 
    combined variance, and return the top feature pair -> explained variance 
    combinations.
    """

    pair_scores = {}
        
    for index,row in df.iterrows():
        _pair_scores = []
        row_dict = dict(row)
        
        for combo in itertools.combinations(df.columns, 2):
            combo = tuple(sorted(combo))
            combo_0 = combo[0]
            combo_1 = combo[1]
            key_tup = (combo_0,combo_1)
            if combo_0 == combo_1:
                continue
            
            val = abs(row_dict[combo[0]] + row_dict[combo[1]])
            if key_tup not in pair_scores:
                pair_scores[key_tup] = []
            
            pair_scores[key_tup].append(val)
        
    avg_scores = {}
    for key,val in pair_scores.iteritems():
        avg_scores[key] = max(val) + (sum(val) / float(len(val)))
    
    return sorted(avg_scores.items(),key=lambda x: x[1], reverse=True)


def column_scores(df,cls,numbers,datetimes,categories,n_components=5):
    """Get the components from each decomposition class, along with the 
    average amount of variance explained by each column in the original data. 
    
    Args:
        df: Pandas dataframe
        cls: decomposition class - eg PCA, FastICA
        numbers: list of header names of continuous columns
        datetimes: list of header names of datetime columns
        categories: list of header names of categorical columns
        n_components: Number of principal components for PCA and FastICA

    Returns:
        A pandas dataframe where columns are the same as the original dataset, 
        and rows represent the explained variance of each column for each 
        component.
    """

    m = cls(n_components=n_components)
    try:
        m.fit_transform(df)
    except:
        return None
    
    # Explained variance of each feature for every component
    indexes = ['c-' + unicode(i) for i in range(n_components)]
    df_components = pd.DataFrame(m.components_,columns=df.columns,index = indexes)
    
    df_scores = df_components[datetimes + numbers]
    for category in categories:
        cat_cols = [c for c in df_components.columns if c.startswith(category)]
        means = []
        for index,row in df_components[cat_cols].iterrows():
            means.append(np.absolute(row).mean())
        df_scores[category] = means

    return df_scores


def score_columns(df=None,columns=None,sample_size=500,n_components=5,
            column_types={}):
    """Given a pandas dataframe or list of columns, look for columns that 
    explain the most variance, under the assumption that these will be the 
    most meaningful (linear) combinations. 

    Args:
        columns: A list of (column_header,column_values) tuples
        df: Pandas dataframe
        column_types: Dict of column header -> semantic types
        sample_size: Max number of obvservations to use
        n_components: Number of principal components for PCA and FastICA

    Returns:
        Two values - first is a list of (column header, score) tuples, and 
        the second is a list of ((column header 1, column header 2), score)
        tuples, where (column header 1, column header 2) is a pair of 
        columns from the original dataset. The score represents the variance 
        these columns explain together.
    """

    if df is not None:
        df = df.dropna()

        sample_amount = int(len(df) * .2)
        if sample_size > sample_amount:
            sample_amount = sample_size
        
        if len(df) > sample_amount:
            sample_amount = len(df)

        if sample_amount > 1000:
            sample_amount = 1000
        
        df = df.sample(n=sample_size)
    
    elif columns:
        sample_amount = int(len(columns[0][1]) * .2)
        if sample_size > sample_amount:
            sample_size = sample_amount
        df = pd.DataFrame(np.array([c[1][:sample_size] for c in columns]).T)
        df.columns = [c[0] for c in columns]
        df = df.dropna()
    
    else:
        raise Exception('Either a dataframe or columns are required')
    
    df, categories, numbers, datetimes = encode_df(df,data_types=data_types)

    df_scores = None
    for cls in [PCA, FastICA, FactorAnalysis]:
        _df_scores = column_scores(df,cls,numbers,datetimes,categories,
                                    n_components=n_components)

        if df_scores is None:
            df_scores = _df_scores
        else:
            df_scores = df_scores.append(_df_scores,ignore_index=True)

    scores = sorted(df_scores.abs().mean().to_dict().items(),key=lambda x: x[1],reverse=True)
    pair_scores = score_feature_groups(df_scores)
    return scores, pair_scores

    