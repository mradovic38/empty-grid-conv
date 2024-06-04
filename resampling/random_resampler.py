import numpy as np

# Resampling the Data - Balancing the Classes
def resample_data(observations, actions):
    unique_classes, counts = np.unique(actions, return_counts=True) # Get the unique classes and their counts
    max_count = counts.max() # Get the number of examples of the class with the most examples
    
    resampled_observations = []
    resampled_actions = []
    
    for cls in unique_classes:
        class_indices = np.where(actions == cls)[0] 
        class_observations = observations[class_indices]
        class_actions = actions[class_indices]
        
        # Calculate the number of samples needed to reach max_count
        num_samples_needed = max_count - len(class_observations)
        
        if num_samples_needed > 0:
            # Randomly sample with replacement
            sampled_indices = np.random.choice(class_indices, size=num_samples_needed, replace=True)
            sampled_observations = observations[sampled_indices]
            sampled_actions = actions[sampled_indices]
            
            # Append original and sampled data
            resampled_observations.append(np.concatenate((class_observations, sampled_observations)))
            resampled_actions.append(np.concatenate((class_actions, sampled_actions)))
        else:
            resampled_observations.append(class_observations)
            resampled_actions.append(class_actions)
    
    # Concatenate all the resampled data
    resampled_observations = np.concatenate(resampled_observations)
    resampled_actions = np.concatenate(resampled_actions)
    
    return resampled_observations, resampled_actions
