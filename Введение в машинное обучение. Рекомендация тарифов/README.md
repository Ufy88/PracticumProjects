#   Рекомендация тарифов

  Многие клиенты оператора "Мегалайн" пользуются архивными тарифами. Необходимо построить систему для анализа поведения клиентов, чтобы предложить пользователям новый тариф: «Смарт» или «Ультра»

  Имеются данные о поведении клиентов на этих тарифах. Нужно подобрать наилучшую модель для классификации подходящего тарифа

Данные признаки:
- сalls — количество звонков
- minutes — суммарная длительность звонков в минутах
- messages — количество sms-сообщений
- mb_used — израсходованный интернет-трафик в Мб
- is_ultra — каким тарифом пользовался в течение месяца («Ультра» — 1, «Смарт» — 0)


#   Используемые модули:

```python
pandas 
sklearn
joblib
```