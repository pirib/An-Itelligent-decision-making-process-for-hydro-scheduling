import Evaluation
import Data
import RandomForest

import matplotlib.pyplot as plt



def imbalanced_learn_testing(data_num, command_column, horizon, name, batch_runs = 100):
    
    results = []
     
    # One run with no pre processing
    data = Data.Data(data_num = data_num, command_column = command_column, 
                     use_pre_process = False)

    mean_accuracy = Evaluation.run_test_batch_ba(RandomForest.RF, data, n_runs=batch_runs)

    results.append((mean_accuracy, False, False))
    
    
    # One run with only pre processing
    data = Data.Data(data_num = data_num, command_column = command_column, 
                     use_pre_process = True, average_interval = 2 )

    mean_accuracy = Evaluation.run_test_batch_ba(RandomForest.RF, data, n_runs=batch_runs)

    results.append((mean_accuracy, False, True))
    
    
    # Exhaustive search with imbalaned learn algorithms and pre processing 
    for imb_learn in ["Under", "Over", "SMOTE"]:
        for pre_process in [True, False]:
            data = Data.Data(data_num = data_num, command_column = command_column, 
                             use_pre_process = pre_process,
                             average_interval= 2,
                             imb_learning = imb_learn) 
            
            mean_accuracy = Evaluation.run_test_batch_ba(RandomForest.RF, data, n_runs=batch_runs)
            results.append((mean_accuracy, imb_learn, pre_process))
        
    # For debugging
    print(name)
    
    for i in results:
        print(i)

    # Time to plot!
        
    plt.figure()

    # Plotting where std is False
    x = [str(i[1]) for i in results if not i[2]]
    y = [i[0] for i in results if not i[2]]
    
    plt.plot(x,y, label = "Pre-process: False", color="orange")
        
    # Plotting where pre process is True
    x = [str(i[1]) for i in results if i[2]]
    y = [i[0] for i in results if i[2]]
    
    plt.plot(x,y, label = "Pre-process: True", color="blue")
    
    
    # Plot it    
    plt.title(name)
    plt.ylim(top=100)

    plt.legend()
    plt.xlabel("Imbalanced Learn Algorithm")
    plt.ylabel("Balanced Accuracy")
    
    plt.plot(x,y)
    plt.show()



# Run the tests for the datasets
imbalanced_learn_testing(data_num=11, command_column=1, horizon=240, name = "Hydro")

"""
imbalanced_learn_testing(data_num=10, command_column=1, horizon=210, name = "Statkraft")

imbalanced_learn_testing(data_num=15, command_column=1, horizon=168, name = "E-CO")
imbalanced_learn_testing(data_num=12, command_column=1, horizon=72, name = "TronderEnergi")
imbalanced_learn_testing(data_num=17, command_column=1, horizon=168, name = "Skagerak")


imbalanced_learn_testing(data_num=18, command_column=1, horizon=72, name = "BKK 72, Command 1")
imbalanced_learn_testing(data_num=18, command_column=2, horizon=72, name = "BKK 72, Command 2")

imbalanced_learn_testing(data_num=14, command_column=1, horizon=336, name = "BKK 336, Command 1")
imbalanced_learn_testing(data_num=14, command_column=2, horizon=336, name = "BKK 336, Command 2")
"""

"""
Results:
    
Skagerak
(59.696103896103885, False, False)
(61.959740259740265, False, True)
(63.42207792207793, 'Under', True)
(59.79220779220778, 'Under', False)
(63.48311688311687, 'Over', True)
(59.69610389610391, 'Over', False)
(64.43896103896104, 'SMOTE', True)
(59.45064935064934, 'SMOTE', False)

Statkraft
(37.58029872708289, False, False)
(39.12153545287827, False, True)
(39.65178278721781, 'Under', True)
(38.87848294031409, 'Under', False)
(39.811849543686535, 'Over', True)
(38.5654759779498, 'Over', False)
(39.991880217039665, 'SMOTE', True)
(38.65073951452792, 'SMOTE', False)

E-CO
(43.18832008767624, False, False)
(46.68844761660756, False, True)
(46.491439853689926, 'Under', True)
(43.359414937185186, 'Under', False)
(46.63127280240123, 'Over', True)
(43.469337074216774, 'Over', False)
(47.660738861009946, 'SMOTE', True)
(44.57152183854997, 'SMOTE', False)

TronderEnergi
(43.98951829306693, False, False)
(45.803520987221965, False, True)
(47.616950751030274, 'Under', True)
(47.09243685862138, 'Under', False)
(48.4472850320795, 'Over', True)
(46.41490678745819, 'Over', False)
(50.255877144072784, 'SMOTE', True)
(47.71468836736355, 'SMOTE', False)

Hydro
(60.868131868131876, False, False)
(58.33928571428571, False, True)
(63.083104395604394, 'Under', True)
(64.75961538461539, 'Under', False)
(61.0315934065934, 'Over', True)
(61.92719780219779, 'Over', False)
(60.68200549450549, 'SMOTE', True)
(63.61607142857143, 'SMOTE', False)

BKK 72, Command 1
(51.11402714932127, False, False)
(53.46764705882353, False, True)
(55.84383484162896, 'Under', True)
(52.80418552036199, 'Under', False)
(55.28936651583711, 'Over', True)
(52.53772624434389, 'Over', False)
(55.11725113122173, 'SMOTE', True)
(52.471493212669685, 'SMOTE', False)

BKK 72, Command 2
(56.42467302933899, False, False)
(62.80583244962885, False, True)
(66.31000353481797, 'Under', True)
(62.477129727819026, 'Under', False)
(65.18677978084128, 'Over', True)
(58.357723577235774, 'Over', False)
(65.40466595970308, 'SMOTE', True)
(61.81431601272535, 'SMOTE', False)

BKK 336, Command 1
(61.10428363485133, False, False)
(77.20342066957787, False, True)
(78.59721355791224, 'Under', True)
(67.86894364732792, 'Under', False)
(81.03904138074444, 'Over', True)
(66.05687253067167, 'Over', False)
(81.55650862965274, 'SMOTE', True)
(69.99022665834893, 'SMOTE', False)

BKK 336, Command 2
(53.021321770334936, False, False)
(55.88107057416268, False, True)
(56.24084928229665, 'Under', True)
(52.85765550239235, 'Under', False)
(56.29261363636363, 'Over', True)
(53.16097488038277, 'Over', False)
(56.62449162679425, 'SMOTE', True)
(53.12840909090908, 'SMOTE', False)


"""
