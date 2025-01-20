#!/usr/bin/env python
# coding: utf-8


# # INTRODUCCION
# 

# Trabajo en una empresa emergente que vende productos alimenticios. Investigare el comportamiento del usuario para la aplicación de la empresa.

# ## Análisis exploratorio de datos (Python)

# In[1]:


#importar librerias para hacer analisis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


#Cargar dataset
log = pd.read_csv('/datasets/logs_exp_us.csv', sep='\t')


# In[3]:


# Inspeccionar primeras filas del dataset
print("Primeras filas de log")
print(log.head(10))


# In[4]:


# Inspeccionar la informacion de los data sets

print(log.info())


# In[5]:


# Resumen de estadisticas para cada Dataset
print("\nSummary statistics of log:")
print(log.describe())


# In[6]:


# Revisar por valores nulos
print("\nValores nulos en log:")
print(log.isnull().sum())


# In[7]:


# Verificar si hay valores duplicados
print(log.duplicated().sum())


# In[8]:


#imprimir los valores duplicados del dataset log
print(log[log.duplicated()])


# Para facilitar el analisis, optaremos por eliminar los valores duplicados

# In[9]:


# Drop duplicates
log = log.drop_duplicates().reset_index(drop=True)
print(log.info())


# In[10]:


#cambio a datetime

log['EventTimestamp'] = pd.to_datetime(log['EventTimestamp'], unit='s')
print(log['EventTimestamp'])


# In[11]:


# Revisar tipos de datos
print("\nData types in log:")
print(log.dtypes)


# In[12]:


# Rename columns
log.columns = ['event_name', 'device_id', 'event_timestamp', 'exp_id']



# In[13]:


# Add separate columns for date and time
log['event_date'] = log['event_timestamp'].dt.date
log['event_time'] = log['event_timestamp'].dt.time



# In[14]:


# Display the first few rows of the modified data
print(log.head(10))


# ## Estudiar y comprobar los datos

# In[15]:


# Contar el numero de eventos unicos
num_events = log['event_name'].nunique()

# Presenta el resultado
print(f"Numero de eventos unicos: {num_events}")


# In[16]:


# Contar el numero de usuarios unicos
num_users = log['device_id'].nunique()

# Display the result
print(f"Numero de usuarios unicos: {num_users}")


# In[17]:


# Contar el numero de eventos por usuario
events_per_user = log.groupby('device_id')['event_name'].count()

#Calcular el promedio de numero de eventos por usuario
average_events_per_user = events_per_user.mean()

# Display the result
print(f"Promedio de numero de eventos por usuario: {average_events_per_user:.2f}")


# In[18]:


# Find the minimum and maximum dates
min_date = log['event_timestamp'].min()
max_date = log['event_timestamp'].max()

# Display the date range
print(f"Date range: {min_date} to {max_date}")


# In[19]:


log['event_date_hour'] = log['event_timestamp'].dt.floor('H')
log2=log['event_date_hour'].value_counts().sort_index()


# In[20]:


# Plot histogram of events by date and hour using a different plotting method
plt.figure(figsize=(14, 8))

log2.plot(kind='hist', bins=50)
plt.title('Histogram of Events by Date and Hour')
plt.xlabel('Date and Hour')
plt.ylabel('Number of Events')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()




# In[21]:


# Determine the completeness of the data using a rolling window
log['event_count'] = log.groupby('event_date_hour')['event_name'].transform('count')
rolling_event_count = log.set_index('event_date_hour')['event_count'].rolling('24H').sum()



# In[22]:


# Plot the rolling event count
plt.figure(figsize=(14, 8))
rolling_event_count.plot()
plt.title('Rolling Event Count (24H)')
plt.xlabel('Date and Hour')
plt.ylabel('Number of Events')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



#  <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo con el filtro, esto te dejará muestras de datos más consistentes
# </div>

# In[23]:


# Analyze the rolling event count to find the date when the data starts to be complete
threshold = rolling_event_count.mean() * 0.5  # Example threshold: 50% of the mean rolling event count
complete_data_start_date = rolling_event_count[rolling_event_count > threshold].index.min()

# Filter the data to include only the complete period
filtered_log = log[log['event_date_hour'] >= complete_data_start_date]

# Find the new minimum and maximum dates in the filtered data
new_min_date = filtered_log['event_timestamp'].min()
new_max_date = filtered_log['event_timestamp'].max()




# In[24]:


# Display the new date range
print(f"New date range: {new_min_date} to {new_max_date}")


# In[25]:


# Calculate the number of events and users in the original and filtered data
original_event_count = log.shape[0]
filtered_event_count = filtered_log.shape[0]
lost_event_count = original_event_count - filtered_event_count

original_user_count = log['device_id'].nunique()
filtered_user_count = filtered_log['device_id'].nunique()
lost_user_count = original_user_count - filtered_user_count

# Display the results
print(f"Original number of events: {original_event_count}")
print(f"Filtered number of events: {filtered_event_count}")
print(f"Lost number of events: {lost_event_count}")
print()
print(f"Original number of users: {original_user_count}")
print(f"Filtered number of users: {filtered_user_count}")
print(f"Lost number of users: {lost_user_count}")


# In[26]:


# Ensure that there are users from all three experimental groups
experimental_groups = filtered_log['exp_id'].unique()
print(f"Experimental groups: {experimental_groups}")


# Check if all three groups are present
if set([246, 247, 248]).issubset(experimental_groups):
    print("All three experimental groups are present.")
else:
    print("Not all experimental groups are present.")


# Podemos comprobar que la perdida de usuarios despues del filtrado es minima, por lo cual seguiremos el analisis con los datos filtrados, ya que ademas se encuentran presentes los 3 grupos experimentales.

# ##  Estudiar el embudo de eventos
# 
# 

# In[27]:


# Numero de eventos 
num_events = filtered_log['event_name'].unique()
print(f"Name of events: {num_events}")


# In[28]:


# Count the frequency of each event
event_frequencies = filtered_log['event_name'].value_counts()

# Sort the events by their frequency in descending order
sorted_event_frequencies = event_frequencies.sort_values(ascending=False)

# Display the sorted event frequencies
print(sorted_event_frequencies)


# In[29]:


# Count the number of unique users who performed each event
users_per_event = filtered_log.groupby('event_name')['device_id'].nunique().reset_index()
users_per_event.columns = ['event_name', 'user_count']


# In[30]:


# Sort the events by the number of users in descending order
sorted_users_per_event = users_per_event.sort_values(by='user_count', ascending=False)


# In[31]:


# Calculate the proportion of users who performed each event at least once
total_users = filtered_log['device_id'].nunique()
sorted_users_per_event['user_proportion'] = sorted_users_per_event['user_count'] / total_users


# In[32]:


# Display the sorted events with user counts and proportions
print(sorted_users_per_event)



# In[33]:


# Identify the sequence of events (example: A → B → C)
event_sequence = ['Tutorial', 'MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful']

# Filter for the events in the sequence
users_per_event = users_per_event[users_per_event['event_name'].isin(event_sequence)]

# Sort the events by the sequence
users_per_event['event_order'] = users_per_event['event_name'].apply(lambda x: event_sequence.index(x))
users_per_event = users_per_event.sort_values(by='event_order')

# Calculate the proportion of users who move from one event to the next
users_per_event['proportion'] = users_per_event['user_count'].shift(-1) / users_per_event['user_count']

# Display the funnel proportions
print(users_per_event[['event_name', 'user_count', 'proportion']])


# In[34]:


# Identify the stage with the most drop-off
users_per_event['drop_off'] = 1 - users_per_event['proportion']
max_drop_off_stage = users_per_event.loc[users_per_event['drop_off'].idxmax()]



# In[35]:


# Display the funnel proportions and the stage with the most drop-off
print(users_per_event[['event_name', 'user_count', 'proportion', 'drop_off']])


# In[36]:


# Calculate the percentage of users who complete the entire journey
users_completing_journey = log[log['event_name'] == event_sequence[-1]]['device_id'].nunique()
percentage_completing_journey = (users_completing_journey / total_users) * 100

# Display the funnel proportions and the percentage of users completing the journey
print(users_per_event[['event_name', 'user_count', 'proportion']])
print(f"Percentage of users completing the entire journey: {percentage_completing_journey:.2f}%")


# Podemos comprobar que en este embudo de ventas, casi 1 de cada 2 personas que entran a la plataforma, se convierte en cliente. Podemos corroborar que el tutorial es evitado en su mayoria, y que donde mas perdida tenemos es cuando sale la pantalla de oferta.

# ##   Estudiar los resultados del experimento

# In[37]:


# Count the number of unique users in each experimental group
users_per_group = filtered_log.groupby('exp_id')['device_id'].nunique().reset_index()
users_per_group.columns = ['exp_id', 'user_count']

# Display the number of users in each group
print(users_per_group)


#  <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Muy bien este paso, siempre hay que revisar que todos los grupos experimentales estén en igualdad de condiciones para poder generar conclusiones correctas
# </div>

# In[38]:


# Filter data to include only the control groups (246 and 247)
control_groups = filtered_log[filtered_log['exp_id'].isin([246, 247])]


# In[39]:


# Calculate conversion rate for each control group
conversion_rate_246 = control_groups[control_groups['exp_id'] == 246]['device_id'].nunique() / control_groups['device_id'].nunique()
conversion_rate_247 = control_groups[control_groups['exp_id'] == 247]['device_id'].nunique() / control_groups['device_id'].nunique()



# In[40]:


# Perform t-test to determine the significance of the difference in conversion rates
conversion_rate_246_data = control_groups[control_groups['exp_id'] == 246]['device_id']
conversion_rate_247_data = control_groups[control_groups['exp_id'] == 247]['device_id']
t_stat, p_value = stats.ttest_ind(conversion_rate_246_data, conversion_rate_247_data, equal_var=False)



# In[41]:


# Display the results
print(f"Conversion rate for group 246: {conversion_rate_246:.2f}")
print(f"Conversion rate for group 247: {conversion_rate_247:.2f}")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")



# In[42]:


# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("The difference in conversion rates between the control groups is statistically significant.")
else:
    print("The difference in conversion rates between the control groups is not statistically significant.")


# In[43]:


# Identify the most popular event
most_popular_event = filtered_log['event_name'].value_counts().idxmax()



# In[44]:


#Total Unique users of each group
total_users_246 = filtered_log[filtered_log['exp_id'] == 246]['device_id'].nunique()
total_users_247 = filtered_log[filtered_log['exp_id'] == 247]['device_id'].nunique()
total_users_248 = filtered_log[filtered_log['exp_id'] == 248]['device_id'].nunique()



# Los grupos estan divididos satisfactoriamente.

# In[45]:


# Function to calculate user counts and proportions for an event
def calculate_event_stats(event_name):
    event_data = filtered_log[filtered_log['event_name'] == event_name]
    users_246 = event_data[event_data['exp_id'] == 246]['device_id'].nunique()
    users_247 = event_data[event_data['exp_id'] == 247]['device_id'].nunique()
    users_248 = event_data[event_data['exp_id'] == 248]['device_id'].nunique()
    proportion_246 = users_246 / total_users_246
    proportion_247 = users_247 / total_users_247
    proportion_248 = users_248 / total_users_248
    return users_246, users_247, users_248, proportion_246, proportion_247, proportion_248



# In[46]:


# Calculate stats for the most popular event
users_246, users_247, users_248, proportion_246, proportion_247, proportion_248 = calculate_event_stats(most_popular_event)



# In[47]:


# Perform chi-square test to determine the significance of the difference
contingency_table = pd.DataFrame({
    'Group 246': [users_246, total_users_246 - users_246],
    'Group 247': [users_247, total_users_247 - users_247],
    'Group 248': [users_248, total_users_248 - users_248]
})
chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)



# In[48]:


# Display the results for the most popular event
print(f"Most popular event: {most_popular_event}")
print(f"Users in Group 246: {users_246}, Proportion: {proportion_246:.2f}")
print(f"Users in Group 247: {users_247}, Proportion: {proportion_247:.2f}")
print(f"Users in Group 248: {users_248}, Proportion: {proportion_248:.2f}")
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p_value}")



# In[49]:


# Interpret the results
alpha = 0.01
if p_value < alpha:
    print("The difference in proportions between the control groups is statistically significant.")
else:
    print("The difference in proportions between the control groups is not statistically significant.")



# In[50]:


# Function to repeat the procedure for all events
def test_all_events():
    event_names = filtered_log['event_name'].unique()
    results = []
    for event_name in event_names:
        users_246, users_247, users_248, proportion_246, proportion_247, proportion_248 = calculate_event_stats(event_name)
        contingency_table = pd.DataFrame({
            'Group 246': [users_246, total_users_246 - users_246],
            'Group 247': [users_247, total_users_247 - users_247],
            'Group 248': [users_248, total_users_248 - users_248]
        })
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        results.append((event_name, users_246, users_247, users_248, proportion_246, proportion_247, proportion_248, chi2, p_value))
    return pd.DataFrame(results, columns=['Event', 'Users 246', 'Users 247', 'Users 248', 'Proportion 246', 'Proportion 247', 'Proportion 248', 'Chi2', 'P-value'])



# In[51]:


# Test all events and display the results
all_event_results = test_all_events()
print(all_event_results)


# Acorde a los experimentos realizados, pese a los cambios que hicimos entre los grupos 246,247 y 248, Estadisiticamente no es significativo. Asi que recomiendo hacer otra cosa para incrementar la conversión a compra, el cual es el objetivo. Vere el numero de hipotesis realizadas para ver el nivel de significancia, y ver si debemos realizar de nuevo la prueba, para descartar un falso positivo.



# In[52]:


# Calculate the number of hypothesis tests performed
num_tests = len(all_event_results)
print(f"Number of hypothesis tests performed: {num_tests}")



# In[53]:


# Adjust the significance level using the Bonferroni correction
adjusted_alpha = 0.05 / num_tests
print(f"Adjusted significance level: {adjusted_alpha}")



# In[54]:


# Re-run the tests with the adjusted significance level and check conclusions
all_event_results['Significant'] = all_event_results['P-value'] < adjusted_alpha
print(all_event_results)


# Ajustando el nivel de significancia a 0.01 obtenemos los mismos resultados, por lo cual doy por terminado el experimento, y concluyo que esta no es la manera de obtener mas ganancias, hay que volver con el departamento de Marketing a sugerir nuevas alternativas, para realizar otro experimento, y poder lograr el objetivo de incrementar las ventas y la conversion de usuarios.


