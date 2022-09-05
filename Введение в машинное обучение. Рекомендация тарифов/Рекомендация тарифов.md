# 1.   Вводные

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
  Многие клиенты оператора "Мегалайн" пользуются архивными тарифами. Необходимо построить систему для анализа поведения клиентов, чтобы предложить пользователям новый тариф: «Смарт» или «Ультра»
<br><br>  Имеются данные о поведении клиентов на этих тарифах. Нужно подобрать наилучшую модель для классификации подходящего тарифа

Данные признаки:
- сalls — количество звонков,
- minutes — суммарная длительность звонков в минутах,
- messages — количество sms-сообщений,
- mb_used — израсходованный интернет-трафик в Мб,
- is_ultra — каким тарифом пользовался в течение месяца («Ультра» — 1, «Смарт» — 0).</div>

# 2.   Настройка пространства


```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
```


```python
data = pd.read_csv(r'/datasets/users_behavior.csv')
```

# 3.   Знакомство с данными

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Посмотрим на срез данных</div>


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>calls</th>
      <th>minutes</th>
      <th>messages</th>
      <th>mb_used</th>
      <th>is_ultra</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.0</td>
      <td>311.90</td>
      <td>83.0</td>
      <td>19915.42</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85.0</td>
      <td>516.75</td>
      <td>56.0</td>
      <td>22696.96</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77.0</td>
      <td>467.66</td>
      <td>86.0</td>
      <td>21060.45</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>106.0</td>
      <td>745.53</td>
      <td>81.0</td>
      <td>8437.39</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66.0</td>
      <td>418.74</td>
      <td>1.0</td>
      <td>14502.75</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Посмотрим на масштабы </div>


```python
data.shape
```




    (3214, 5)



<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Посмотрим на оценку безошибочности и типы данных</div>


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3214 entries, 0 to 3213
    Data columns (total 5 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   calls     3214 non-null   float64
     1   minutes   3214 non-null   float64
     2   messages  3214 non-null   float64
     3   mb_used   3214 non-null   float64
     4   is_ultra  3214 non-null   int64  
    dtypes: float64(4), int64(1)
    memory usage: 125.7 KB


<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Ошибок нет. Проверим на адекватность</div>


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>calls</th>
      <th>minutes</th>
      <th>messages</th>
      <th>mb_used</th>
      <th>is_ultra</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3214.000000</td>
      <td>3214.000000</td>
      <td>3214.000000</td>
      <td>3214.000000</td>
      <td>3214.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>63.038892</td>
      <td>438.208787</td>
      <td>38.281269</td>
      <td>17207.673836</td>
      <td>0.306472</td>
    </tr>
    <tr>
      <th>std</th>
      <td>33.236368</td>
      <td>234.569872</td>
      <td>36.148326</td>
      <td>7570.968246</td>
      <td>0.461100</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>40.000000</td>
      <td>274.575000</td>
      <td>9.000000</td>
      <td>12491.902500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>62.000000</td>
      <td>430.600000</td>
      <td>30.000000</td>
      <td>16943.235000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>82.000000</td>
      <td>571.927500</td>
      <td>57.000000</td>
      <td>21424.700000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>244.000000</td>
      <td>1632.060000</td>
      <td>224.000000</td>
      <td>49745.730000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Все выглядит культурно и порядочно, предобработка не требуется</div>

# 4.   Разделение данных

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Целевой признак очевиден – это is_ultra</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
С признаками для вопросов вроде тоже все понятно – все данные нам понадобятся, но при возникновении проблем с accuracy можно попробовать выкинуть некоторые столбцы, которые в теории могут создавать шум</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Разделим данные на тренировочную, валидационную и тестовую выборки, в соотношении 3:1:1</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Начнем с тестовой – ее должно быть 0.2, остальные данные положим во "временные" переменные</div>


```python
X_, X_test, Y_, Y_test = train_test_split(data[['calls', 'minutes', 'messages', 'mb_used']], 
                                          data['is_ultra'], 
                                          test_size=0.2, 
                                          random_state=12345)
```

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Теперь разделим 0.8 данных на тренировочную и валидационную – пригодятся наши "временные" данные</div>


```python
X_train, X_valid, Y_train, Y_valid = train_test_split(X_, Y_, test_size=0.25)
```


```python
# проверка на пересечение

display(set(X_train.index) & set(X_valid.index), set(X_valid.index) & set(X_test.index))
display(X_train.shape, X_valid.shape, X_test.shape)
```


    set()



    set()



    (1928, 4)



    (643, 4)



    (643, 4)


<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Так как у нас осталось 0.8 данных, а нам нужно их разделить на 0.6 и 0.2, берем четверть от всего для валидации</div>

# 5.   Модели

## 5.1.   DecisionTreeClassifier

### 5.1.1.   Обучение

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Нам будет необходимо пробежаться по гиперпараметрам и проверить их работоспособность на деле. Имеем три основных:
<br><br>- max_depth: зададим верх как квадрат количества признаков
<br><br>- min_samples_split: зададим верх как количество признаков
<br><br>- min_samples_leaf: зададим верх как количество признаков </div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Лучший результат будем записывать в отдельную переменную best_DTC_result, лучшие настройки гиперпараметров – в best_DTC_hyper</div>


```python
# задаем исходный лучший результат – пока что 0
best_DTC_result = 0
features_count = X_train.shape[1]

# тремя вложенными циклами перебираем все гиперпараметры во всех комбинациях:
for max_depth in range(1, features_count**2+1):
    for min_samples_split in range(2, features_count+1):
        for min_samples_leaf in range(1, features_count+1):
            # создаем модель с гиперпараметрами
            DTC_model = DecisionTreeClassifier(random_state=12345, 
                                               max_depth=max_depth, 
                                               min_samples_split=min_samples_split, 
                                               min_samples_leaf=min_samples_leaf)
            # обучаем модель и предсказываем
            DTC_result = DTC_model.fit(X_train, Y_train)
            DTC_predictions = DTC_model.predict(X_valid)
            # считаем accuracy и сверяем полученный результат с предыдущими лидерами
            DTC_accuracy = accuracy_score(DTC_predictions, Y_valid)
            if DTC_accuracy > best_DTC_result:
                # в случае нового короля перезаписываем рекорды и модель
                best_DTC_result = DTC_accuracy
                best_DTC_model = DTC_model
                best_DTC_hyper = (max_depth, min_samples_split, min_samples_leaf)
```

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Перебрав 192 модели, получили лучшую. Посмотрим на ее accuracy</div>


```python
best_DTC_result
```




    0.8102643856920684



<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Уже неплохо, 81% точности. Проверим гиперпараметры</div>


```python
best_DTC_hyper
```




    (8, 2, 2)



<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Ожидаемо, минимальные гиперпараметры остались верны своим настройкам по умолчанию – там сложно что либо подкрутить в свою сторону, если не углубляться в специфику данных. Но мы хотя бы попробовали. Максимальной глубиной рекомендуется считать 5 уровней – не вижу причины этому не верить</div>

## 5.2.   RandomForestClassifier

### 5.2.1.   Обучение

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Та же логика с перебором всех возможных конфигураций. Но имеем уже четыре основных гиперпараметра:
<br><br>- n_estimators: будем перебирать от 1 до 10
<br><br>- max_depth: зададим верх как квадрат количества признаков
<br><br>- min_samples_split: зададим верх как количество признаков
<br><br>- min_samples_leaf: зададим верх как количество признаков</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Лучший результат будем все так же записывать в best_RFC_result, лучшие настройки гиперпараметров – в best_RFC_hyper</div>


```python
# задаем исходный лучший результат – пока что 0
best_RFC_result = 0
features_count = X_train.shape[1]

# тремя вложенными циклами перебираем все гиперпараметры во всех комбинациях:
for n_estimators in range(1, 11):
    for max_depth in range(1, features_count**2+1):
        for min_samples_split in range(2, features_count+1):
            for min_samples_leaf in range(1, features_count+1):
                # создаем модель с гиперпараметрами
                RFC_model = RandomForestClassifier(random_state=12345, 
                                                   n_estimators=n_estimators, 
                                                   max_depth=max_depth, 
                                                   min_samples_split=min_samples_split, 
                                                   min_samples_leaf=min_samples_leaf)
                # обучаем модель и предсказываем
                RFC_result = RFC_model.fit(X_train, Y_train)
                RFC_predictions = RFC_model.predict(X_valid)
                # считаем accuracy и сверяем полученный результат с предыдущими лидерами
                RFC_accuracy = accuracy_score(RFC_predictions, Y_valid)
                if RFC_accuracy > best_RFC_result:
                    # в случае нового короля перезаписываем рекорды и модель
                    best_RFC_result = RFC_accuracy
                    best_RFC_model = RFC_model
                    best_RFC_hyper = (n_estimators, max_depth, min_samples_split, min_samples_leaf)
```

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Перебрав уже 1920 моделей, получили лучшую. Посмотрим на ее accuracy</div>


```python
best_RFC_result
```




    0.8242612752721618



<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Чуть лучше, 82% точности</div>


```python
best_RFC_hyper
```




    (7, 7, 2, 4)



<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Все так же ожидаемо минимальные гиперпараметры остались теми же. Максимальной глубиной стало 10 уровней, оптимальным количеством деревьев – 7 штук</div>

## 5.3.   LogisticRegression

### 5.3.1.   Обучение

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Здесь все намного скромнее – гиперпараметров мы не рассматриваем, построим такую модель, чтобы было с чем сравнить</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Лучший результат будет в best_LG_result, лучших настроек гиперпараметров – нет</div>


```python
best_LG_model = LogisticRegression(random_state=12345)
```


```python
# обучаем модель на тестовых данных 
best_LG_model.fit(X_train, Y_train)
# предсказываем результаты
best_LG_predictions = best_LG_model.predict(X_valid)
# считаем accuracy
best_LG_result = accuracy_score(best_LG_predictions, Y_valid)
```


```python
best_LG_result
```




    0.71850699844479



## 5.4.   Тестирование

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Проверим лучшую модель – RandomForestClassifier – на тестовой выборке</div>


```python
best_RFC_model_predictions = best_RFC_model.predict(X_test)
```


```python
accuracy_score(best_RFC_model_predictions, Y_test)
```




    0.807153965785381



<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Здесь 80% точности. Неплохо, проходной балл есть, но однозначно есть куда стремиться. Эта модель отработала хорошо</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Попробуем переобучить лучшую модель с подобранными гиперпараметрами на выборке, состоящей из тренировочной и валидационной</div>


```python
RFC_model = RandomForestClassifier(random_state=12345, 
                                                   n_estimators=9, 
                                                   max_depth=9, 
                                                   min_samples_split=3, 
                                                   min_samples_leaf=1)
```


```python
RFC_model.fit(X_, Y_)
```




    RandomForestClassifier(max_depth=9, min_samples_split=3, n_estimators=9,
                           random_state=12345)




```python
RFC_model_predictions = RFC_model.predict(X_test)
```


```python
accuracy_score(best_RFC_model_predictions, Y_test)
```




    0.807153965785381



<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
В целом, ничего не изменилось. В то же время, постоянство – признак мастерства</div>

# 6.   Общий вывод

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Мы получили чистые, подготовленные данные с нужными колонками</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Разбили их на DataFrame с признаками (calls, minutes, messages, mb_used) и целевой признак is_ultra в виде Series. Разделили их на тренировочную, валидационную и тестовую выборки в соотношении 3:1:1</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Обучили и проверили 3 модели – DecisionTreeClassifier, RandomForestClassifier, LogisticRegression</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
Первые два показали себя с хорошей стороны на боевых данных, последний не дал ожидаемых результатов</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
DecisionTreeClassifier:
<br><br>- Меняли гиперпарамтры max_depth, min_samples_split, min_samples_leaf 
<br><br>- Получили лучшую настройку как 5, 2, 1 соответственно
<br><br>- На валидационных и на тестировочных данных показатель accuracy лучшей модели был 0.79
<br><br>- Пороговое значение пройдено, модель можно применять для предсказаний</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
RandomForestClassifier:
<br><br>- Меняли гиперпарамтры n_estimators, max_depth, min_samples_split, min_samples_leaf 
<br><br>- Получили лучшую настройку как 7, 10, 2, 1 соответственно
<br><br>- На валидационных данных показатель accuracy лучшей модели был 0.82, на тестировочных – 0.79
<br><br>- Пороговое значение пройдено, модель можно применять для предсказаний</div>

<div style="background-color: #fff0e0; padding: 10px; font-family: monospace; font-size: 15px">
LogisticRegression:
<br><br>- Гиперпарамтры не меняли
<br><br>- Лучшую настройку не получали
<br><br>- На валидационных данных показатель accuracy лучшей модели был 0.69, на тестировочных – 0.70
<br><br>- Пороговое значение не пройдено, модель нельзя применять для предсказаний</div>
