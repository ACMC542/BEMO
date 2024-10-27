# BEMO Framework

The first step in this study was to understand the MAFAULDA dataset and perform an initial analysis. The dataset is comprised of sensor data. At this stage, identifying key variables such as vibration, temperature, and pressure is crucial. These variables, along with their units and ranges, help define the scope of the study, that is, whether it is anomaly detection.  

Following the initial analysis, data preprocessing was conducted to improve data quality and relevance. This process involves cleaning the data by removing missing values, duplicates, and inconsistencies. Additionally, new features were created using statistical metrics, enhancing the dataset’s informational depth and capturing important patterns within the data. The file used to create the new features is available at [create_features.py](https://github.com/ACMC542/BEMO/blob/main/create_features.py).


Once the data have been preprocessed, the dataset is split into subsets for training, validation, and testing for each experiment.  The [runs_mafaulda.ipynb](https://github.com/ACMC542/BEMO/blob/main/runs_mafaulda.ipynb) was used to generate the data combinations needed for the experiments.

For each data subset used in the experiments we evaluated the use of eight algorithms, which are: SVC, XGBClassifier, LGBMClassifier, CatBoostClassifier, DecisionTreeClassifier, KNeighborsClassifier, LogisticRegression and RandomForestClassifier. These experiments can be accessed at [experiments.ipynb](https://github.com/ACMC542/BEMO/blob/main/experiments.ipynb). Additionally, an experiment was created involving the OCSVM (One-Class Support Vector Machines) algorithm, which can be observed in [Testes_one_class (1).ipynb](https://github.com/ACMC542/BEMO/blob/main/Testes_one_class%20(1).ipynb).

Finally, with the results of the experiments we created some graphs to illustrate this work that are available at [graficos.ipynb](https://github.com/ACMC542/BEMO/blob/main/Testes_one_class%20(1).ipynb).

## Link to dataset

- [MAFAULDA. Machinery Fault Database [Online]](http://www02.smt.ufrj.br/~offshore/mfs/page_01.html) (13Gb)


