# 5ARIP10
Interdisciplinary team project

Team members:
Jingyuan Han - 1803859


Things to caution:
1) Before you do any operation, please create a new branch first. After you finish this step, then push it to the main branch. Do not operate directly on main branch!

Packages to download:
-matplotlib
-numpy 
-scipy
-librosa
-latex

# How to run Genetic Algorithm (GA)

## Relevant files
Genetic_algorithms.ipynb  
GA_feature_extraction.py

## Steps to run
1. Open Genetic_algorithms.ipynb  
2. Run modules "Import libaraies" and "Load in the data". This step would generate a database of test data first, and will then load in the microphone and current test data.  
Things to be adjusted: 
```python
# adjust the data directory to the one on your own PC please
HD_dl = dl.HD(r'E:\5ARIP0_Interdisciplinary_team_project\data_analyse\5ARIP10\HD_Model_NEW\HD_data')
```
3. Run module "Extract features". This step will extract features of test data, and form a feature map of the features.  
Things to be adjusted:
```python
# adjust the x's in the below code to change the lengths of time windows during feature extraction

mic_time_feats = fe.extract_time_features(data = mic_data, x = 44100)   # use fs as x
mic_fft_feats = fe.extract_fft_features(data=mic_data, x = 44100)   # use fs as x
# using x = (current_min_len/60) could produce the same n_time_windows as mic data
current_time_feats = fe.extract_time_features(data=current_data, x = int(current_min_len/60))
```
4. Run module "Perform GA". This step would select the fittest features, and draw a plot between normal and anomalous data with the features (if the number of selected features is set to 2).  
Things to be adjusted:
```python
# choose featuers to be put into genetic_algorithm 
# change the columns of feature_map_norm to choose different features: [:15]->Mic(time); [15:20]->Mic(freq); [20:]->Cur(time)
feature_map_chosen = feature_map_norm[:, :15]

# n_goal_features, num_genes need to be adjusted; population_size, mutation_rate, elitism_rate, num_generations could be tuned. After trials, population_size should increase with num_genes increasing, while other parameters are less likely to be tuned
n_goal_features = 2
data = feature_map_chosen
# if want to cover all combinations of features, then set 2 to the n_genes th, note that the running tiem would be very long 
population_size =  10000  
num_genes = 15  
mutation_rate=0.01
elitism_rate=0.1
num_generations= 5

features_chosen = genetic_algorithm(data, n_goal_features, population_size, num_genes, calculate_fitness, mutation_rate, 
                                    elitism_rate, num_generations)
```

```python
# note that number of normal test files and anomalous test files should be the same to run this line. Or, adjust n_norm and run.
features_plot(features, n_norm = int(features_chosen.shape[0]/2))
```
