import pandas as pd

from collections import Counter
import random
import numpy as np

from sklearn import cluster

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SVMSMOTE

class Data:

    """ 
    Loads up and cleans the data.

    data_num - indicates which dataset we want to load. 
        
    use_pre_process - will pre process data by using averages over the given data period (e.g. 24 hours market price become one column). Also removes constant values

    command_column - which of the command columns to use. Possible values are 1 and 2. The not chosen one is dropped from the dataset alltogether.

    average_interval - this is how many columns will be aggregated together in the pre processing (e.g.)
    
    normalize - will normalize all columns (e.g. squeeze it between 0 and 1)
    
    imb_learning - applies imbalanced learning algorithm to the dataset. Available ones are "Under", "Over" and "SMOTE" sampling (https://imbalanced-learn.org/). If None, this step is skipped.

    sdandard_scaler - applies standard scaling to the data rows
    
    feature_agglomeration / n-clusters - apply feature agglomeration with n number of clusters
    
    use_pca / pca_components - principal component analysis - when turned on, will agglomerate features into pca_components number 

    debug - will print some extra information about the dataset if set to True
    
    """

    def __init__(self, data_num = 2, command_column = 1, use_pre_process = True, include_std_dev = False, average_interval = 1, 
                 normalize = False, imb_learning = None, standard_scaler = False, feature_agglomeration = False, n_clusters = 4, use_pca = False, pca_components = 4,
                 debug = False):

        # Set the variables to be accessed all over the class
        self.data_num = data_num
        self.command_column = command_column
        self.average_interval = average_interval
        self.imb_learning = imb_learning
        self.debug = debug

        # Datasets have different number of command columns - defined here for easy access
        self.num_commands = 1 if self.data_num in [1,10,11,12,15,17] else 2

        #   Read the appropriate data set
        self.data = pd.read_csv("Data/Data" + str(data_num) + ".csv", header='infer')

        # Set the correct number of reservoirs
        if data_num in [1]:                        
            self.num_res = 1
            
        elif data_num in [2,3,4,5,6,7,8,9,13,14,15,16,18,19,20]:
            self.num_res = 4

        elif data_num in [10, 17]:
            self.num_res = 2

        elif data_num == 12:
            self.num_res = 6
            
        elif data_num == 11:
            self.num_res = 13

        # Save the original data
        self.original_data = self.data.copy()

        #   Remove the run sequence for the pre processing
        self.data = self.data.drop(columns = ["0"])

        # Pre Process it if needed
        if use_pre_process: 
            self.data = self.PreProcess(self.data, include_std_dev)            
                        
        # Grab the data (X) and labels(Y) from the DataHandler
        self.X, self.Y = self.GetData() 

        # Normalize if specified
        if normalize:
            self.X.apply(lambda x: x/x.max(), axis=0)

        # Use Standard Scaler if specified
        if standard_scaler:
            scaler = StandardScaler()
            scaler.fit(self.X)
            self.X = scaler.transform(self.X)
            
        if feature_agglomeration:
            agglo = cluster.FeatureAgglomeration(n_clusters)
            agglo.fit(self.X)
            self.X = agglo.transform(self.X)

        if use_pca :
            pca = PCA(pca_components)
            pca.fit(self.X)
            self.X = pca.transform(self.X)


        # Print out some debug info about the dataset
        if self.debug:
            print("Columns in the dataset")
            print(self.data.columns)
            
            print("\nColumns in X ")
            print(self.X.columns)        

            print("\nColumns in Y " + self.Y.name + "\n\n")


    def PreProcess(self, data, include_std_dev = True):

        # Set the right time horizon
        time_horizon = 24

        if self.data_num in [4,6,8,9,12,18,19,20]:
            time_horizon = 72
        elif self.data_num in [7, 13, 15, 16, 17]:
            time_horizon = 168
        elif self.data_num in [10]:
            time_horizon = 210
        elif self.data_num in [11]:
            time_horizon = 240
        elif self.data_num == 14:
            time_horizon = 336            
            
        slices = 1    
        # Make sure it is possible to divide the time horizons into intervals
        if time_horizon % self.average_interval != 0:
            print("Average over value results in unused columns. Exiting.")
            raise
        else:
            slices = int(time_horizon/self.average_interval)

        #   Calculate the average of the PPH (price per hour) and standard deviation if needed
        PPH = []
        STD_PPH = []

        for i in range(self.average_interval):
            PPH.append([])
            
            for index, row in data.iloc[:, i*slices : slices+i*slices].iterrows():
                PPH[i].append(np.average(row))

                if include_std_dev:
                    STD_PPH.append([])
                    STD_PPH[i].append(np.std(row))

        #   Calculate the average for the inflow per reservoir
        IPH = []
        STD_IPH = []
        
        for reservoir in range(self.num_res):
            IPH.append([])
            STD_IPH.append([])                                                        

            for i in range(self.average_interval):
                IPH[reservoir].append([])
                STD_IPH[reservoir].append([])    
                
                for index, row in data.iloc[: , time_horizon*reservoir + slices*i: time_horizon*reservoir + slices*i + slices ].iterrows():
                    IPH[reservoir][i].append(np.average(row))

                    if include_std_dev:
                        STD_IPH[reservoir][i].append(np.std(row))


        #   Processed dataframe
        proc_data = pd.DataFrame()

        # Include processed data
        
        #   Price per hour          
        for i in range(len(PPH)):
           proc_data['PPH_int_'+ str(i)] = PPH[i]
           if include_std_dev:
               proc_data['PPH_STD' + str(i)] = STD_PPH[i]        
        
        
        #   Inflow per hour
        for reservoir in range(len(IPH)):
            for i in range(len(IPH[reservoir])):
                proc_data['IPHres' + str(reservoir) + "Int" + str(i)] = IPH[reservoir][i]
            
                if include_std_dev:
                    proc_data['IPHres' + str(reservoir) + "Int" + str(i) + 'STD'] = STD_IPH[reservoir][i]

        
        #   Sort out the rest of the data
        data = data[data.columns[time_horizon+time_horizon*self.num_res:]]
        
        #   Remove the empty/constant values
        data = data[data.columns[data.nunique()>1]]
        
        #   Concatenate the rest
        proc_data = pd.concat( [proc_data, data ], axis = 1)

        return proc_data

        
    # Returns the data and the labels
    def GetData(self):
        
        # Raise an error if a command column does not exist (e.g. wrong input)
        if self.command_column > self.num_commands: 
            print("The selected dataset does not have that many command columns, check input.")
            raise

        X = self.data[self.data.columns[:-self.num_commands]]

        if self.num_commands == 1:
            Y = self.data[self.data.columns[-1]]
        else:
            Y = self.data[self.data.columns[-1]] if self.command_column == 2 else self.data[self.data.columns[-2]]

        # Cast labels as int
        Y = Y.astype('int32')
        
        return X, Y


    # Splits the data into a training and testing/evaluation set. iloc_test is for testing specificially chosen rows.
    def Split(self, test_size = 0.25, stratify_y = True, iloc_test = None):
        
        # Turn self.x into the dataframe
        self.X = pd.DataFrame(self.X)
        
        # If no specific indexes are given, then we just carry on
        if iloc_test is None:
        
            st = self.Y if stratify_y else None
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, stratify = st, test_size = test_size, random_state = 42 )
            
            # Imbalanced learning
            if self.imb_learning is not None:
                # Random Under Sampling
                if self.imb_learning == "Under":
                    rus = RandomUnderSampler()
                    self.X_train, self.y_train = rus.fit_resample(self.X_train, self.y_train)
                    
                # Random Over Sampling
                elif self.imb_learning == "Over":
                    ros = RandomOverSampler()
                    self.X_train, self.y_train = ros.fit_resample(self.X_train, self.y_train)
                    
                # SMOTENC
                elif self.imb_learning == "SMOTE":
                    smote_nc = SVMSMOTE()
                    self.X_train, self.y_train = smote_nc.fit_resample(self.X_train, self.y_train)
                    
                else:
                    print("imb_learning got an unexcpeted value. Exiting.")
                    raise
            
        # Else, we separate the sets manually
        else:
            iloc_test = [i + 1 for i in iloc_test]
            
            self.X_train = self.X.drop(iloc_test)
            self.X_test = self.X.iloc[iloc_test]

            self.y_train = self.Y.drop(iloc_test)
            self.y_test = self.Y.iloc[iloc_test]
            
            print(self.y_test)

        if self.debug:
            print("Training data")
            print(Counter(self.y_train))
            print("Testing data")
            print(Counter(self.y_test))

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    
    # Returns feature ranking - where True is which column to keep. Estimator is used to select these features, keep_n indicates how many to keep in the evaluation phase.
    def GetFeatureRanking(self, estimator, keep_n = 8, verbose = False):
        
        selector = RFE(estimator, n_features_to_select = keep_n, step = 1)
        
        X,y = self.GetData()        

        selector = selector.fit(X,y)

        if verbose:
            print(selector.support_)
            print(selector.ranking_)

        return selector.support_


    # Keeps the specified columns in the X dataframe, and drops the rest. 
    def KeepColumns(self, binary_array):
        
        # Get a list of all columns
        columns = list(self.X)
        columns_to_delete = [i for i,j in zip(columns, binary_array) if not j ]

        self.X = self.X.drop(columns = columns_to_delete)

    # Saves the predictions into a file
    def SavePredictions(self, predictions):

        new_index = [i-1 for i in self.y_test.index]

        # Grab the rows from the original datasets                
        orig_df = self.original_data.iloc[new_index]
                
        print(new_index)
        
        # Add the predictions
        orig_df["ML output"] = predictions
        
        # Save the csv file
        orig_df.to_csv("Predictions.csv", index = False)
        
        
        
        
