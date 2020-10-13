#!/usr/bin/env python
# coding: utf-8

# # Combinatorial Optimization for Digital Office Solutions Research

# This document serves to provide the mathematical formulation and implementation code for conducting a combinatorial optimization analysis for research on portential digital office solutions.

# ## Mathematical Formulation

# \begin{align}
# \text{optimal solution} = \min\ \text{sort}(A, n)
# \end{align}

# Where:

# \begin{align}
# \text{sort} &:= f: A, n \rightarrow y, \text{where } |y| \le n, n \in \mathbb{N}, n \le |I| \\
# y &:= \text{list result of a first-element-keyed top-down sort of list-casted set } A
# \end{align}

# Given definitions:

# \begin{align}
# A &:= \left(\sum_{i \in I}u_ix_i, |X_i|\right) \text{ s.t. } \sum_{i \in I}w_ix_i \le W \wedge \sum_{i \in I}u_ix_i \ge 0 \wedge \hat{c} = \text{True, } \forall \hat{c} \in C \\
# I &:= I_\text{new} \cap I_\text{existing} \\
# I_\text{new} &:= \text{discrete set of potential tech solutions (e.g. tools, processes)} \\
# I_\text{existing} &:= \text{discrete set of existing tech solutions (e.g. tools, processes)} \\
# x_i &:=
#     \begin{cases}
#     x_\text{new},& \text{if } i \in I_\text{new}\\
#     x_\text{existing},& \text{if } i \in I_\text{existing}
#     \end{cases} \\
# x_\text{new} &:=
#     \begin{cases}
#     1,& \text{if $i$ is adopted, where } i \in I_\text{new} \\
#     0,& \text{otherwise}
#     \end{cases} \\
# x_\text{existing} &:=
#     \begin{cases}
#     1.5,& \text{if $i$ is adopted, where } i \in I_\text{existing}\\
#     0,& \text{otherwise}
#     \end{cases} \\
# X_i &:= \{x_i | x_i = 1\}, i \in I \\
# u_i &:= \text{utility of } i \in I,\ u_i \in \mathbb{R}^+ \\
# w_u &:= \text{cost of } i \in I, w_i \in \mathbb{R}^+ \\
# W &:= \text{budget (cost upper-bound)} \\
# C &:= \text{discrete set of constraints}
# \end{align}

# ## Implementation

# This implementation solution of the modified combinatorial optimization problem uses a dynamic programming approach.

# __Setup__

# In[ ]:


# !pip install pandas


# In[2]:


import ast
import pandas as pd

from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, NewType, NoReturn, Union


# In[3]:


Tech = NewType('Tech', Dict[str, Union[bool, pd.DataFrame]])
TThruple = NewType('TThruple', Tuple[pd.DataFrame, List[int], int])


# In[4]:


class Rank(Enum):
    NA   = 0.00
    LOW  = 0.33
    MED  = 0.67
    HIGH = 1.00


# In[5]:


def get_row_values(df: pd.DataFrame, rows: List[str]
        ) -> List[Tuple[str, List[int]]]:
    """
    Returns a list of tuples containing row labels (left)
    and a list of row values (right).
    
    """
    row_val_pairs = []
    for row in rows:
        vals = [Rank[ast.literal_eval(datapoint)['rank']].value
                    for datapoint in df.loc[row].values]
        row_val_pairs.append((row, vals))

    return row_val_pairs


# In[6]:


def avg_row_ranks(df: pd.DataFrame, rows: List[str]
        ) -> List[Tuple[str, int]]:
    """
    Returns a list of tuples containing row labels (left) 
    and sum of row ranks (right).

    """
    rank_avgs = []
    
    row_val_pairs = get_row_values(df, rows)
    for pair in row_val_pairs:
        row, vals = pair
        rank_avgs.append((row, sum(vals) / len(vals)))

    return sorted(rank_avgs, key=lambda tup: tup[1], reverse=True)


# __Data Setup__

# In[7]:


all_df            = pd.read_csv('data/metrics-all.csv', header=[0, 1, 2], index_col=0)
browser_compat_df = pd.read_csv('data/browser-compatibility.csv', index_col=0)
opsys_compat_df   = pd.read_csv('data/os-compatibility.csv', index_col=0)
prop_df           = pd.read_csv('data/tech-properties.csv', index_col=0)


# In[8]:


technologies = list(all_df.index)
props        = prop_df.to_dict('index')
browsers     = browser_compat_df.to_dict('index')
opsys        = opsys_compat_df.to_dict('index')


# In[9]:


tech_utilities = {key: val for key, val in avg_row_ranks(all_df, technologies)}


# In[10]:


tech_properties = {}

for tech in props:
    props_dict = props[tech]
    props_dict['browsers'] = browsers[tech]
    props_dict['operating systems'] = opsys[tech]
    
    tech_properties[tech] = props_dict


# ### Dynamic Programming Build

# __Helper Functions__

# In[11]:


def bitpad_int(intkey: int, length: int) -> str:
    """
    Given an integer and length of binary string, returns a 
    binary string of the integer with padded zeros.
    
    """
    if intkey > 2**length - 1:
        raise RuntimeError("Given int has more bits than given length")
        
    return format(intkey, '0{}b'.format(length))


# In[12]:


def serialize(binarr: List[int]) -> str:
    """
    Given a binary arry, returns a binary string.
    
    """
    return ''.join([str(x) for x in binarr])


# In[13]:


def deserialize(binstr: str) -> List[int]:
    """
    Given a binary string, returns a binary array.
    
    """
    return list(map(int, list(binstr)))


# In[14]:


def lookup_items(binstr: str, items: List[Any]) -> List[Any]:
    """
    Given a binary string and its represented lookup table of items, 
    returns a list of items of which the binary string flags.
    
    """
    binarr = deserialize(binstr)
    indices = [i for i in range(len(binarr)) if binarr[i] == 1]
    return [items[i] for i in indices]


# In[15]:


def count_unadopted(df: pd.DataFrame, items: List[Any]) -> int:
    """
    Returns the number of items in the given list that have not yet been
    adopted.
    
    """
    counter = 0
    
    for i in items:
        if not df['properties'][i][0]['adopted']:
            counter += 1 
    
    return counter


# In[16]:


def get_util(df: pd.DataFrame,
             metrics_df: pd.DataFrame,
             binarr: List[int], 
             i: int, 
             W: float, 
             C: List[Callable[[TThruple], bool]]
        ) -> float:
    """
    Given a Dataframe, metrics, a binary array, an index i, a weight upper-bound W,
    a list of constraint lambdas C, returns the utility of the item in index i.
    
    """
    indicator = -1
    properties = df['properties'].iloc[i]
    c_results = [f(df, metrics_df, binarr, i) for f in C]
    weight_sum = sum([df['weight'].iloc[j] for j in range(0, i)
                                           if binarr[j] == 1])
    
    if (not all(c_results)) or (weight_sum > W):
        indicator = 0.0
    elif properties['adopted']:
        indicator = 1.5
    else:
        indicator = 1.0
    
    return (df['utility'].iloc[i] * indicator)


# __Dynamic Programming Functions__

# In[17]:


def set_utils_dp(table: Dict[str, float],
                 df: pd.DataFrame,
                 metrics_df: pd.DataFrame,
                 key: str, 
                 i: int, 
                 W: float, 
                 C: List[Callable[[TThruple], bool]]
        ) -> str:
    """
    Given a serial lookup table, a DataFrame, metrics, a binary string key, 
    an index i, a weight upper-bound W, and a list of constraint 
    lambdas C, recursively sets the utility of the given key and its 
    binary substrings into the table by reference using dynamic 
    programming.
    
    """
    # Base Case (last digit)
    if i == len(df.index) - 1:
        binarr = deserialize(key)
        
        if binarr[i] == 0:
            table[key] = 0.0
        else:
            table[key] = get_util(df, metrics_df, binarr, i, W, C)
    
    # Memoization Case
    if key not in table:
        binarr = deserialize(key)
        next_binarr = [0 if j <= i else binarr[j] for j in range(len(binarr))]
        next_binstr = serialize(next_binarr)
        
        if binarr[i] == 0:
            table[key] = table[set_utils_dp(table, df, metrics_df, 
                                            next_binstr, i + 1, W, C)]
        else:
            util = get_util(df, metrics_df, binarr, i, W, C)
            if util == 0: # unsatisfied constraints
                table[key] = 0.0
            else:
                table[key] = util + table[set_utils_dp(table, df, metrics_df, 
                                                       next_binstr, i + 1, W, C)]
    
    return key


# In[18]:


def memoize_all(df: pd.DataFrame,
                metrics_df: pd.DataFrame,
                W: float, 
                C: List[Callable[[TThruple], bool]]
        ) -> Dict[str, float]:
    """
    Conducts combinatorial optimization calculations using dynamic programming 
    and returns the memoized results.
    
    """
    memo_table = {}
    n = len(df.index)

    for i in range(2**n - 1, 0, -1):
        key = bitpad_int(i, n)
        set_utils_dp(memo_table, df, metrics_df, key, 0, W, C)
    
    return memo_table


# ### Dynamic Programming Solution

# __Dataset__

# In[19]:


tech_metadata = []
tech_cols = ['utility', 'weight', 'properties']

for tech in technologies:
    tech_metadata.append([tech_utilities[tech], -1, tech_properties[tech]])

tech_meta_df = pd.DataFrame(data=tech_metadata, index=[technologies], 
                            columns=tech_cols)
metrics_df = all_df # renaming for clarity


# __Weight Bound__

# In[20]:


weight_upper_bound = 100 
    # n.b. arbitrarily selected since current problem has no weight bound


# __Constraints__

# In[21]:


def has_good_tag_combo(df: pd.DataFrame, metrics_df: pd.DataFrame, 
                      binarr: List[int], i: int) -> bool:
    """
    Called in a constraints lambda, returns true if the given configurations
    contain an acceptable combination of tags.
    
    """
    items = lookup_items(binarr, technologies) 
    tags = [ast.literal_eval(df['properties'][item][0]['tags']) 
                for item in items]
    
    num_3da = 0
    num_2da = 0
    num_2dh = 0
    num_vc = 0
    
    for t in tags:
        if '3D OFFICE ANALOGUE' in t:
            num_3da += 1
        if '2D OFFICE ANALOGUE' in t:
            num_2da += 1
        if '2D OFFICE HYBRID' in t:
            num_2dh += 1
        if 'VIDEO CONFERENCING' in t:
            num_vc += 1
    
    return ((num_3da == 1 and num_2da == 0 and num_2dh == 0 and num_vc < 2) or
            (num_3da == 0 and num_2da == 1 and num_2dh == 0 and num_vc < 2) or
            (num_3da == 0 and num_2da == 0 and num_2dh == 1 and num_vc < 2) or
            (num_3da == 0 and num_2da == 0 and num_2dh == 0 and num_vc == 1))


# In[22]:


def has_acc_sec(df: pd.DataFrame, metrics_df: pd.DataFrame, i: int) -> bool:
    """
    Returns true if the given item has acceptable security levels. Called in a
    constraints lambda.
    
    """
    item = df.index.to_list()[i][0]
    
    transmission_sec = ast.literal_eval(metrics_df[('communication', 
                                                    'transmission', 
                                                    'secure')][item])['rank']
    storage_sec = ast.literal_eval(metrics_df[('communication', 
                                               'storage', 
                                               'secure')][item])['rank']
    
    return ((transmission_sec == 'MED' or transmission_sec == 'HIGH') and 
            (storage_sec == 'MED' or storage_sec == 'HIGH'))


# In[23]:


constraints = [
    lambda df, metrics_df, binarr, i: \
        not df['properties'].iloc[i]['beta'],
    
    lambda df, metrics_df, binarr, i: \
        (df['properties'].iloc[i]['browsers']['Google Chrome'] or 
         df['properties'].iloc[i]['browsers']['Microsoft Edge'] or
         df['properties'].iloc[i]['operating systems']['Windows']),
    
    lambda df, metrics_df, binarr, i: \
        has_good_tag_combo(df, metrics_df, binarr, i),
    
    lambda df, metrics_df, binarr, i: \
        has_acc_sec(df, metrics_df, i)
]


# __Results Sorting__

# In[24]:


results = memoize_all(tech_meta_df, metrics_df, weight_upper_bound, constraints)


# In[25]:


pruned_results = [(key, results[key], lookup_items(key, technologies)) 
                      for key in results if results[key] != 0.0]


# In[26]:


top_5_results = sorted(pruned_results, key=lambda tup: tup[1], reverse=True)[:5]


# In[27]:


top_w_icount = [(key, results, items, count_unadopted(tech_meta_df, items)) 
                    for key, results, items in top_5_results]


# In[28]:


final_results = sorted(top_w_icount, key=lambda tup: tup[3])


# ## Findings

# In[29]:


print("==================================================")
print("Recommended Configuration of Technologies to Adopt")
print("==================================================\n")
print("{:50}: Utility Measure".format('Configuration'))
print("-------------------------------------------------------------------")
for config in final_results:
    config_str = ', '.join(config[2])
    print("{:50}: {}".format(config_str, config[1]))


# ### Time Complexity

# The time complexity of a dynamic programming solution is equal to the number of subproblems times the number of operations per subproblem. 
# 
# A brute-force, recursive approach to this combinatorial optimization problem contains $n$ subproblems, each requiring $\log_2^2n$ operations (the sum of utility measures for each item in a configuration ($\log_2n$) where each sum is evaluated against a list of constraints, and where some constraints require $\log_2n$ operations). Therefore, a bruteforce approach would have a time-complexity of $O(n\log^2n)$. 
# 
# The memoization component of dynamic programming removes redundent operations. In the case of this combinatorial optimization problem, the redudant operations are in the summation of utility measures for each configuration item. Therefore, the dynamic programming solution to this combinatorial optimization problem has a time complexity of $O(n\log n * |A|)$, where $|A|$ is the size of the memoized summations.
# 
# *N.B.* The dynamic programming solution can be further optimized for greater efficiency. For example, the solution can also memoize the infeasible region with constant-time checking to reduce redundant constraint checking, thus reducing the time complexity to $O(2*|A|*|I|)$, where $|I|$ is the size of the infeasible region.
