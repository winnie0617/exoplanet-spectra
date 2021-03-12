#%% Initial set up
import numpy as np
import pandas as pd

#%% 
surfaces = ['biota', 'snow', 'sand', 'seawater', 'basalt', 'cloud']
columns = ['biota_type', 'biota', 'snow', 'sand', 'seawater', 'basalt', 'cloud']
biotas = ["Agrococcus", "Geodermatophilus", "Bark", "Lichen", "Aspen-leaf", "Leafy-Spurge"]
# Values of percentage coverage
values = np.array(range(0, 101, 5))
print("values", values)

# Find all possible spectra (combinations of percentage coverage of 6 surfaces)

def get_all_sum_to(total, length, values):
    '''Returns a list of lists (with specified length) of possible combinations of the values that sums to total'''
    res = np.empty((0, length), int)
    # Already obtained the target sum - only possibility is a list of zeros
    if total == 0:
        res = np.append(res, np.zeros((1, length)), axis=0)
        return res 
    # Only one more element needed - only one possibility
    if length == 1:
        res = np.append(res, np.array([[total]]), axis=0)
        return res

    # Consider all possible first element for the list
    for head in values:
        if head <= total:
            all_tails = get_all_sum_to(total-head, length-1, values)
            all_heads = np.full((np.shape(all_tails)[0], 1), head) # matches with tails dimension
            combo_with_head = np.append(all_heads, all_tails, axis=1)
            res = np.append(res, combo_with_head, axis=0)
    
    return res

final = get_all_sum_to(100, len(surfaces), values) / 100 # convert to decimal
num_spectra = final.shape[0]
assert num_spectra == 53130 # TODO: don't hard code
df = pd.DataFrame(final, columns=surfaces)
# Add biota label
biota_list = np.concatenate(list(np.repeat(b, num_spectra) for b in biotas))
df = pd.concat([df]*6)
df.reset_index(inplace=True, drop=True)
df.insert(0, 'biota_type', biota_list)

# df.to_csv('surface_composition.csv', index=False)

\

# %%
