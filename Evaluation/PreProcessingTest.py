import Evaluation
import Data
import RandomForest

import matplotlib.pyplot as plt



# Evaluates the model(s), and each of their command columns test n_runs number of times.
def preprocess_testing(data_num, command_column, horizon, name, batch_runs = 100):    

    results = []
    
    testing_list = None
    
    if horizon == 168:
        testing_list = [2, 4, 8, 12, 24,56, 84, 168]    # For 168 hour horizons
    elif horizon == 72:
        testing_list = [2, 4, 8, 12, 24, 36, 72]         # For 72 hour
    elif horizon == 210:
        testing_list = [2, 3, 5, 10, 21, 30, 70, 210]    # For 210 hour
    elif horizon == 240:
        testing_list = [2, 4, 8, 10, 24, 48, 120, 240]
    elif horizon == 336:
        testing_list = [2, 4, 8, 14, 24, 48, 84, 336]
        
    # One run with no pre processing
    data = Data.Data(data_num = data_num, command_column = command_column, 
                     use_pre_process = False)

    mean_accuracy = Evaluation.run_test_batch_ba(RandomForest.RF, data, n_runs=batch_runs)

    results.append((mean_accuracy, "Pre_process: False"))

    # All the other runs
    for av_int in testing_list:
        for std in [True, False]:
            data = Data.Data(data_num = data_num, command_column = command_column, 
                             use_pre_process = True, include_std_dev = std, average_interval = av_int)
                
            mean_accuracy = Evaluation.run_test_batch_ba(RandomForest.RF, data, n_runs=batch_runs)
            results.append((mean_accuracy, (av_int, std)))

    # Print out the results, and then start plotting    
    print(name)
    for i in results:
        print(i)    
    
    plt.figure()
    
    # Plotting where std is True
    x = [str(i[1][0]) for i in results if i[1][1]]
    y = [i[0] for i in results if i[1][1]]
    
    plt.plot(x,y, label = "STD: True", color="orange")
    
    # Plotting where std is False
    x = [str(i[1][0]) for i in results if not i[1][1]]
    y = [i[0] for i in results if not i[1][1]]
    
    plt.plot(x,y, label = "STD: False", color="blue")
        
    # Plot it
    
    plt.title(name)
    plt.legend()
    plt.ylim(top=100)
    plt.xlabel("Averaging Interval")
    plt.ylabel("Balanced Accuracy")
    
    plt.plot(x,y)
    plt.show()


preprocess_testing(data_num=10, command_column=1, horizon=210, name = "Statkraft")

"""
# All dataset testing
preprocess_testing(data_num=17, command_column=1, horizon=168, name = "Skagerak")
preprocess_testing(data_num=15, command_column=1, horizon=168, name = "E-CO")
preprocess_testing(data_num=12, command_column=1, horizon=72, name = "TronderEnergi")
preprocess_testing(data_num=11, command_column=1, horizon=240, name = "Hydro")
preprocess_testing(data_num=10, command_column=1, horizon=210, name = "Statkraft")
preprocess_testing(data_num=14, command_column=1, horizon=336, name = "BKK, Command 1")
preprocess_testing(data_num=14, command_column=2, horizon=336, name = "BKK, Command 2")

# BKK varied horizons 
preprocess_testing(data_num=18, command_column=1, horizon=72, name = "BKK 72, Command 1")
preprocess_testing(data_num=18, command_column=2, horizon=72, name = "BKK 72, Command 2")

preprocess_testing(data_num=13, command_column=1, horizon=168, name = "BKK 168, Command 1")
preprocess_testing(data_num=13, command_column=2, horizon=168, name = "BKK 168, Command 2")

preprocess_testing(data_num=14, command_column=1, horizon=336, name = "BKK 336, Command 1")
preprocess_testing(data_num=14, command_column=2, horizon=336, name = "BKK 336, Command 2")
"""

"""
Info dumb from the runs above.

All datasets:

Skagerak
(58.9077922077922, 'Pre_process: False')
(62.16493506493508, (2, True))
(63.27662337662338, (2, False))
(60.20259740259741, (4, True))
(61.774025974025975, (4, False))
(59.85584415584416, (8, True))
(63.037662337662326, (8, False))
(59.85194805194806, (12, True))
(60.4935064935065, (12, False))
(58.335064935064935, (24, True))
(59.06363636363637, (24, False))
(57.394805194805194, (56, True))
(58.72727272727273, (56, False))
(57.11818181818183, (84, True))
(59.688311688311686, (84, False))
(58.423376623376626, (168, True))
(57.91298701298702, (168, False))

E-CO
(43.229026044004016, 'Pre_process: False')
(46.78563363822598, (2, True))
(46.61082496358673, (2, False))
(46.032669037413186, (4, True))
(46.2002469064482, (4, False))
(44.95542334817155, (8, True))
(45.57866536348408, (8, False))
(44.33631196559018, (12, True))
(45.145714700441914, (12, False))
(44.155877953912515, (24, True))
(44.634031471713605, (24, False))
(43.64343189995511, (56, True))
(44.21352228332913, (56, False))
(43.24105151994004, (84, True))
(44.20953693573144, (84, False))
(43.869353204832706, (168, True))
(43.79426093455237, (168, False))

TronderEnergi
(44.720272518236236, 'Pre_process: False')
(45.22309723380626, (2, True))
(45.6714390382143, (2, False))
(42.99657393796684, (4, True))
(43.89019859025342, (4, False))
(42.88343555455204, (8, True))
(43.02952127291914, (8, False))
(42.19345845640584, (12, True))
(43.26314706185037, (12, False))
(40.16218997633015, (24, True))
(42.60719096705421, (24, False))
(40.293825715510124, (36, True))
(42.69627318793498, (36, False))
(42.17587425638096, (72, True))
(42.20375814659337, (72, False))

Hydro
(60.31730769230769, 'Pre_process: False')
(58.57486263736264, (2, True))
(58.113324175824175, (2, False))
(58.754120879120876, (4, True))
(59.21771978021979, (4, False))
(59.894230769230774, (8, True))
(60.2129120879121, (8, False))
(59.948489010989015, (10, True))
(60.309752747252745, (10, False))
(61.607142857142854, (24, True))
(60.73763736263737, (24, False))
(61.423076923076934, (48, True))
(61.010302197802204, (48, False))
(61.37431318681319, (120, True))
(62.25961538461539, (120, False))
(61.8434065934066, (240, True))
(62.18475274725276, (240, False))

Statkraft
(37.605232592736854, 'Pre_process: False')
(40.407090846140946, (2, True))
(38.82612563632331, (2, False))
(39.47903527706099, (3, True))
(38.793256190120694, (3, False))
(38.29982445320826, (5, True))
(38.460045387907016, (5, False))
(37.05409816554955, (10, True))
(37.38770577383948, (10, False))
(37.394436600288884, (21, True))
(37.71218332603015, (21, False))
(37.204959129244635, (30, True))
(37.951928580316036, (30, False))
(36.55124505682955, (70, True))
(37.56341485845356, (70, False))
(37.67289871301751, (210, True))
(37.50812032393658, (210, False))

BKK, Command 1
(62.27396548138907, 'Pre_process: False')
(75.49766063630693, (2, True))
(76.86650031191516, (2, False))
(70.98882304013307, (4, True))
(73.20461634435432, (4, False))
(65.08078602620088, (8, True))
(67.55224578914536, (8, False))
(61.762736535662306, (14, True))
(64.24880432522355, (14, False))
(60.17243709710959, (24, True))
(63.32397587856103, (24, False))
(59.07615928467457, (48, True))
(62.015439800374295, (48, False))
(58.245217300894154, (84, True))
(61.163235599916824, (84, False))
(59.76876689540444, (336, True))
(59.58640049906425, (336, False))

BKK, Command 2
(52.92583732057415, 'Pre_process: False')
(55.87571770334928, (2, True))
(55.79611244019138, (2, False))
(54.670514354067, (4, True))
(54.844467703349274, (4, False))
(53.706459330143545, (8, True))
(54.404724880382794, (8, False))
(53.776525119617226, (14, True))
(54.38283492822967, (14, False))
(53.61668660287082, (24, True))
(53.761842105263156, (24, False))
(53.01892942583731, (48, True))
(53.11991626794259, (48, False))
(52.737709330143545, (84, True))
(53.347398325358846, (84, False))
(51.76985645933013, (336, True))
(52.23229665071769, (336, False))



BKK varied horizons:

BKK 72, Command 1
(51.891742081447966, 'Pre_process: False')
(52.742024886877815, (2, True))
(54.04830316742081, (2, False))
(51.00141402714931, (4, True))
(52.69756787330316, (4, False))
(51.321380090497726, (8, True))
(52.05390271493212, (8, False))
(51.37884615384614, (12, True))
(51.92590497737556, (12, False))
(51.65955882352941, (24, True))
(51.968608597285076, (24, False))
(51.22070135746606, (36, True))
(51.63546380090499, (36, False))
(51.06001131221719, (72, True))
(50.764705882352935, (72, False))

BKK 72, Command 2
(56.779957582184515, 'Pre_process: False')
(61.731954754330154, (2, True))
(62.5949098621421, (2, False))
(58.81997172145635, (4, True))
(60.38769883351008, (4, False))
(57.68161894662424, (8, True))
(59.28989042064333, (8, False))
(57.86359137504419, (12, True))
(59.18900671615412, (12, False))
(56.86843407564511, (24, True))
(58.571120537292316, (24, False))
(56.00664545775892, (36, True))
(57.925556733828216, (36, False))
(55.75309296571227, (72, True))
(56.973983739837394, (72, False))
   

BKK 168, Command 1
(61.0395652173913, 'Pre_process: False')
(70.79878260869566, (2, True))
(70.99869565217391, (2, False))
(66.91130434782607, (4, True))
(68.2315652173913, (4, False))
(68.0135652173913, (8, True))
(67.22321739130433, (8, False))
(66.568, (12, True))
(67.04208695652173, (12, False))
(66.8915652173913, (24, True))
(65.65799999999999, (24, False))
(63.78713043478261, (56, True))
(64.32504347826088, (56, False))
(62.51078260869564, (84, True))
(62.95321739130435, (84, False))
(59.575913043478266, (168, True))
(61.836000000000006, (168, False))
    
BKK 168, Command 2
(53.256395628152774, 'Pre_process: False')
(58.342691568580356, (2, True))
(58.33644607254384, (2, False))
(56.169979582032184, (4, True))
(58.01285130915205, (4, False))
(54.69096805188567, (8, True))
(56.59464328609175, (8, False))
(54.68664424693729, (12, True))
(56.90892985827528, (12, False))
(54.201987749219306, (24, True))
(55.846655056449684, (24, False))
(53.51411241892866, (56, True))
(54.76753543117944, (56, False))
(53.59914724957963, (84, True))
(54.23795940427577, (84, False))
(53.845844343021845, (168, True))
(53.61641244294979, (168, False))


BKK 336, Command 1
(61.427323767935114, 'Pre_process: False')
(75.80302557704304, (2, True))
(77.43636930754835, (2, False))
(70.91999376169682, (4, True))
(73.44650655021834, (4, False))
(64.4284154709919, (8, True))
(67.56175920149721, (8, False))
(61.98279268039093, (14, True))
(64.02973591183199, (14, False))
(59.81191515907674, (24, True))
(63.65751715533375, (24, False))
(58.92456851736327, (48, True))
(61.95669577874819, (48, False))
(58.51050114368893, (84, True))
(61.34102724059056, (84, False))
(59.50051985859845, (336, True))
(59.15642545227697, (336, False))
 
BKK 336, Command 2
(52.77706339712918, 'Pre_process: False')
(55.42619617224881, (2, True))
(55.75191387559809, (2, False))
(54.42338516746411, (4, True))
(55.07772129186603, (4, False))
(54.16480263157895, (8, True))
(54.32263755980862, (8, False))
(53.67284688995216, (14, True))
(54.009330143540666, (14, False))
(53.71169258373206, (24, True))
(53.5238038277512, (24, False))
(53.42293660287081, (48, True))
(53.62637559808613, (48, False))
(52.79542464114832, (84, True))
(52.7532894736842, (84, False))
(51.841297846889944, (336, True))
(52.31097488038278, (336, False))

"""








