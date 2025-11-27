# Порівняння результатів
Рекорд: 0.8862 за F7.1.
[Датасет](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - стандартний для цієї задачі, анонімізовані дані банківських транзакцій, шарайський та не шахрайських.
Оцінка - F7.1 (FN в 50 разів гірший за FP) (правильність беззмістовна - дуже незбалансовані дані).
Для кожної моделі шукаю в кінці найкращий treshold (наскільки треба впевненість щоб зарахувати як 1).

Моделі я не комітила, крім фінальних ансамблів GB та NN, щоб не смітити. (Напишіть, якщо вам треба.) Результати їх оцінювання в основному лежать по папках, а код - в кореневій папці.

## Logistic Regression - 0.8511
Взагалі не оптимізована, це базовий варіант.
```--- F7.1-Score Optimization ---
Optimal Threshold: 0.9839
Best F7.1-Score as per the matrix: 0.8511
--- Classification Report for threshold 0.9839---  
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.63      0.86      0.72        98

    accuracy                           1.00     56962
   macro avg       0.81      0.93      0.86     56962
weighted avg       1.00      1.00      1.00     56962

--- Confusion Matrix ---
[[56814    50]
 [   14    84]]```

## Random Forest - 0.8753
Теж не підкручувала, просто на пробу. Тим не менш має один із найкращих результатів.
```--- F7.1-Score Optimization ---
Optimal Threshold: 0.0700
Best F7.1-Score as per the matrix: 0.875293
--- Classification Report for threshold 0.0700---  
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.77      0.88      0.82        98

    accuracy                           1.00     56962
   macro avg       0.89      0.94      0.91     56962
weighted avg       1.00      1.00      1.00     56962

--- Confusion Matrix ---
[[56839    25]
 [   12    86]]```

Схоже, я ще пробувала об'єднувати їх в ансамбль, але результат вийшов гірший - 0.8746.
```(4 models)
--- F7.1-Score Optimization ---
Optimal Threshold: 0.0700
Best F7.1-Score as per the matrix: 0.8746
--- Classification Report for threshold 0.0700---  
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.75      0.88      0.81        98

    accuracy                           1.00     56962
   macro avg       0.87      0.94      0.90     56962
weighted avg       1.00      1.00      1.00     56962

--- Confusion Matrix ---
[[56835    29]
 [   12    86]]```

## Gradient Boosting (LightGBM) - 0.8844
Це перша модель, яку я всерйоз підкурчувала. Дала сама по собі мало не найкращий рещультат, але не пропорційний кількості зусиль. Результати індивідуальних моделей не зберіглись, бо вони були гірші. (Лежить ще якийсь ансамбль в miracle, але воно старе і гірше.)
```THIS RESULT USES np.max INSTEAD OF np.mean TO CONSOLIDATE THE ENSEMBLE SCORES
Loaded ensemble of 4 models with following params:           
{'boosting_type': 'dart', 'class_weight': 'balanced', 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 600, 'n_jobs': None, 'num_leaves': 31, 'objective': None, 'random_state': 61, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'force_col_wise': True}
--- F7.1-Score Optimization ---
Optimal Threshold: 0.0411
Best F7.1-Score as per the matrix: 0.8844
--- Classification Report for threshold 0.0411---
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864        
           1       0.74      0.89      0.81        98        

    accuracy                           1.00     56962        
   macro avg       0.87      0.94      0.90     56962        
weighted avg       1.00      1.00      1.00     56962        

--- Confusion Matrix ---
[[56834    30]
 [   11    87]]```

## Multi-layer Perceptron - 0.8733
Я трошки встигла попідкручувати, але не сильно, бо там складно і довго і результатів силььно не було. Стабільності не досягла, просто отримала якісь три чудо-моделі, привожу тут найкращу.
```CONFIG = {
     'lr': 0.0005,             
     'hidden_units': 60,      
     'dropout': 0.3,       
     'weight_decay': 0.0001,
     'pos_weight_mult': 1.0, 
     'max_epochs': 50,      
     'batch_size': 1024, 
}
--- F7.1-Score Optimization ---
Best F7.1-Score as per the matrix: 0.8733
--- Classification Report for threshold 0.8856---
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     56864
         1.0       0.37      0.90      0.52        98

    accuracy                           1.00     56962
   macro avg       0.68      0.95      0.76     56962
weighted avg       1.00      1.00      1.00     56962

--- Confusion Matrix ---
[[56712   152]
 [   10    88]]```

Тут теж робила ансамбль з усіх трьох (ймовірності об'єднані за допомогою np.average), але він дав трохи гірші результати (хоча напевно він більш стабільний).
```--- F7.1-Score Optimization ---
Best F7.1-Score as per the matrix: 0.8725
--- Classification Report for threshold 0.8578---
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     56864
         1.0       0.36      0.90      0.51        98

    accuracy                           1.00     56962
   macro avg       0.68      0.95      0.76     56962
weighted avg       1.00      1.00      1.00     56962

--- Confusion Matrix ---
[[56707   157]
 [   10    88]]```

## Final combination of GB and NN examples - 0.8862
Врешті-решт найкращий результат вийшов, коли я об'єднала ймовірності ансамблів GB та NN з коефіцієнтами 0.85 та 0.15.
```--- F7.1-Score Optimization ---
Best F7.1-Score as per the matrix: 0.8862
--- Classification Report for threshold 0.9979---
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     56864
         1.0       0.81      0.89      0.85        98

    accuracy                           1.00     56962
   macro avg       0.91      0.94      0.92     56962
weighted avg       1.00      1.00      1.00     56962

--- Confusion Matrix ---
[[56844    20]
 [   11    87]]```
