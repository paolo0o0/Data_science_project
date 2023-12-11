
"""
Original file is located at
    https://colab.research.google.com/drive/12IWOLdCm4rhJTz2Q5uq8Vt43WkcmoJ5d

# Тестовое задание М.Тех
**Data Science**
"""

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.stats.proportion import proportions_ztest
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_icon=":bar_chart",
    layout="wide"
)

st.title("Проверка гипотез")

with st.sidebar:
    st.header("Исследуемые данные")
    uploaded_file = st.file_uploader("Загрузить csv")

if uploaded_file is None:
    st.info("Пожалуйста, загрузите csv-файл")
    st.stop()

data = pd.read_csv(uploaded_file, encoding='cp1251')
names = data.columns.str.split(',').tolist()
data = data.iloc[:,0].str.split(',', expand=True)
data.columns = names
names = names[0]

for i in range(data.shape[1]):
    names[i] = names[i].replace('"','')
data.columns = names
for i, col in enumerate(data.columns):
    data.iloc[:, i] = data.iloc[:, i].str.replace('"', '')
for column in data.columns[:-1]:
    data[column] = data[column].astype('int64')
data['Пол'] = data['Пол'].astype('string')

with st.expander("Предпросмотр данных"):
    st.write(data)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Первая гипотеза")
    st.write("Мужчины пропускают в течение года более work_days рабочих дней по болезни значимо чаще женщин")

    workdays = st.slider("Установите параметр 'workdays'", 0, 8, 2)
    st.write(f'workdays = {workdays}')

    significance = st.selectbox('Задайте уровень статистической значимости', [0.01, 0.05, 0.10])

    data['Более workdays пропусков'] = (data['Количество больничных дней'] > workdays).astype(int)

    st.write('Будем считать успешным событие, когда количество пропусков в году для конкретного человека > workdays')
    st.write('Добавим в таблицу новый столбец, в котором будет храниться 1, '
             'если у человека более workdays пропусков в год (успех), и 0, если меньше или равно (неудача)')

    with st.expander("Предпросмотр столбца"):
        st.write(data['Более workdays пропусков'])

    data['Пол'] = data['Пол'].map({'М': 1, 'Ж': 0})

    male_data = data[data['Пол'] == 1]
    female_data = data[data['Пол'] == 0]

    male_success = male_data['Более workdays пропусков'].sum()
    male_unsuccess = male_data.shape[0] - male_success
    female_success = female_data['Более workdays пропусков'].sum()
    female_unsuccess = female_data.shape[0] - female_success

    colors = ["#4682B4", "#87CEEB"]
    fig = go.Figure()
    fig.add_trace(go.Pie(values=[male_unsuccess, male_success], labels=[0, 1], hole=0.3,
                         marker=dict(colors=colors)))
    fig.update_layout(title='Соотношение "успехов" и "неудач" в группе мужчин')
    st.plotly_chart(fig, use_container_width=True)

    colors = ["#FFA500", "#FFE4B5"]
    fig = go.Figure()
    fig.add_trace(go.Pie(values=[female_unsuccess, female_success], labels=[0, 1], hole=0.3,
                         marker=dict(colors=colors)))
    fig.update_layout(title='Соотношение "успехов" и "неудач" в группе женщин')
    st.plotly_chart(fig, use_container_width=True)

    st.write('Построим диаграмму, показывающую разность между долями в двух выборках, '
             'а также вычислим доверительные интервалы для этих разностей')

    p1 = male_data['Более workdays пропусков'].mean()
    p2 = female_data['Более workdays пропусков'].mean()
    p = data['Более workdays пропусков'].mean()

    st.write('Доля успешных событий для группы мужчин: ', format(p1, '.3f'))
    st.write('Доли успешных событий для группы женщин: ', format(p2, '.3f'))
    st.write('Комбинированная доля успешных событий: ', format(p, '.3f'))

    z = st.selectbox('Задайте z критическое значение', [1.645, 1.960, 2.580])
    if z == 1.645:
        conf_level = 0.90
    elif z == 1.960:
        conf_level = 0.95
    else:
        conf_level = 0.99

    st.write('Уровень достоверности:', conf_level)

    SE = np.sqrt(p * (1 - p) * (1 / male_data.shape[0] + 1 / female_data.shape[0]))
    conf_int = z * SE

    st.write('Доверительный интервал: ', format(conf_int, '.3f'))
    st.write('Этот доверительный интервал предоставляет оценку диапазона, в пределах которого мы ожидаем, '
             'что истинная разность между долями будет с высокой вероятностью находиться.')

    difference = p1 - p2
    x = ['Доля в выборке мужчин', 'Доля в выборке женщин', 'Разность']
    y = [p1, p2, difference]
    err = [0, 0, conf_int]

    fig = px.bar(x=x, y=y, title='Доли выборок и их разность с доверительным интервалом')
    fig.update_traces(marker_color=['#4682B4', '#DB7093', '#8FBC8F'])
    fig.add_shape(type="line", x0=x[2], y0=difference - err[2],
                      x1=x[2], y1=difference + err[2], line=dict(color="black", width=2))
    st.plotly_chart(fig)

    st.write('Проведём двухвыборочный биномиальный тест')

    male_successes = male_data['Более workdays пропусков'].sum()
    female_successes = female_data['Более workdays пропусков'].sum()

    successes = np.array([male_successes, female_successes])
    trials = np.array([male_data.shape[0], female_data.shape[0]])

    z_stat, p_value = proportions_ztest(successes, trials, alternative='larger')

    st.write('z-статистика: ', format(z_stat, '.3f'))
    st.write('p_value: ', format(p_value, '.3f'))

    if p_value > significance:
        st.subheader("Первая гипотеза отвергнута")
    else:
        st.write("Имеет место статистически значимая связь между переменными")
        st.write("Таким образом, действительно, мужчины пропускают в течение года более work_days "
                 "рабочих дней по болезни значимо чаще женщин")
        st.subheader("Первая гипотеза подтверждена")

with col2:
    st.subheader("Вторая гипотеза")
    st.write("Работники старше age лет пропускают в течение года более "
             "workdays рабочих дней по болезни значимо чаще своих более молодых коллег")

    age = st.slider("Установите параметр 'age'", 23, 60, 35, key='slider1')
    st.write(f'age = {age}')

    workdays = st.slider("Установите параметр 'workdays'", 0, 8, 2, key='slider2')
    st.write(f'workdays = {workdays}')

    significance = st.selectbox('Задайте уровень статистической значимости', [0.01, 0.05, 0.10], key='select3')

    data['Более workdays пропусков'] = (data['Количество больничных дней'] > workdays).astype(int)

    st.write('Будем считать успешным событие, когда количество пропусков в году для конкретного человека > workdays')
    st.write('Добавим в таблицу новый столбец, в котором будет храниться 1, '
             'если у человека более workdays пропусков в год (успех), и 0, если меньше или равно (неудача)')

    with st.expander("Предпросмотр столбца"):
        st.write(data['Более workdays пропусков'])

    young_data = data[data['Возраст'] <= age]
    old_data = data[data['Возраст'] > age]

    young_success = young_data['Более workdays пропусков'].sum()
    young_unsuccess = young_data.shape[0] - young_success
    old_success = old_data['Более workdays пропусков'].sum()
    old_unsuccess = old_data.shape[0] - old_success

    colors = ["#2E8B57", "#8FBC8F"]
    fig = go.Figure()
    fig.add_trace(go.Pie(values=[young_unsuccess, young_success], labels=[0, 1], hole=0.3,
                         marker=dict(colors=colors)))
    fig.update_layout(title='Соотношение "успехов" и "неудач" в молодой группе')
    st.plotly_chart(fig, use_container_width=True)

    colors = ["#A0522D", "#DEB887"]
    fig = go.Figure()
    fig.add_trace(go.Pie(values=[old_unsuccess, old_success], labels=[0, 1], hole=0.3,
                         marker=dict(colors=colors)))
    fig.update_layout(title='Соотношение "успехов" и "неудач" в возрастной')
    st.plotly_chart(fig, use_container_width=True)

    st.write('Построим диаграмму, показывающую разность между долями в двух выборках, '
             'а также вычислим доверительные интервалы для этих разностей')

    p1 = young_data['Более workdays пропусков'].mean()
    p2 = old_data['Более workdays пропусков'].mean()
    p = data['Более workdays пропусков'].mean()

    st.write('Доля успешных событий для молодой группы: ', format(p1, '.3f'))
    st.write('Доли успешных событий для возрастной группы: ', format(p2, '.3f'))
    st.write('Комбинированная доля успешных событий: ', format(p, '.3f'))

    z = st.selectbox('Задайте z критическое значение', [1.645, 1.960, 2.580], key='select4')
    if z == 1.645:
        conf_level = 0.90
    elif z == 1.960:
        conf_level = 0.95
    else:
        conf_level = 0.99

    st.write('Уровень достоверности:', conf_level)

    SE = np.sqrt(p * (1 - p) * (1 / young_data.shape[0] + 1 / old_data.shape[0]))
    conf_int = z * SE

    st.write('Доверительный интервал: ', format(conf_int, '.3f'))
    st.write('Этот доверительный интервал предоставляет оценку диапазона, в пределах которого мы ожидаем, '
             'что истинная разность между долями будет с высокой вероятностью находиться.')

    difference = p1 - p2
    x = ['Доля в молодой группе', 'Доля в возрастной группе ', 'Разность']
    y = [p1, p2, difference]
    err = [0, 0, conf_int]

    fig = px.bar(x=x, y=y, title='Доли выборок и их разность с доверительным интервалом')
    fig.update_traces(marker_color=['#4682B4', '#DB7093', '#8FBC8F'])
    fig.add_shape(type="line", x0=x[2], y0=difference - err[2],
                      x1=x[2], y1=difference + err[2], line=dict(color="black", width=2))
    st.plotly_chart(fig)

    st.write('Проведём двухвыборочный биномиальный тест')

    old_successes = old_data['Более workdays пропусков'].sum()
    young_successes = young_data['Более workdays пропусков'].sum()

    successes = np.array([old_successes, young_successes])
    trials = np.array([old_data.shape[0], young_data.shape[0]])

    z_stat, p_value = proportions_ztest(successes, trials, alternative='larger')

    st.write('z-статистика: ', format(z_stat, '.3f'))
    st.write('p_value: ', format(p_value, '.3f'))

    if p_value > significance:
        st.subheader("Вторая гипотеза отвергнута")
    else:
        st.write("Имеет место статистически значимая связь между переменными")
        st.write("Таким образом, действительно,   работники старше age лет пропускают в течение года более "
                 "workdays рабочих дней по болезни значимо чаще своих более молодых коллег")
        st.subheader("Первая гипотеза подтверждена")