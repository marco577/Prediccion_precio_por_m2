#!/usr/bin/env python
# coding: utf-8

# # Predicción de precio por $m^2$ de un inmueble con base en sus características

# ## Entendimiento del negocio

# Se proporciona un conjunto de datos que enlista una serie de inmuebles con sus características geoespaciales y amenidades, el objetivo es poder predecir el precio por $m^2$. Los datos han sido extraídos de diversas páginas de anunciantes de inmuebles que se pueden encontrar en Internet. La metodología planteada para resolver este problema es la propuesta por IMB llamada CRISP-DM.
# En este cuaderno se puede encontrar de forma detallada la elaboración de la metodología a seguir para este proyecto y la base de la automatización de un nuevo conjunto de datos.

# In[1]:


# librerías
# análisis de datos
import pandas as pd
import numpy as np
# graficación
import matplotlib.pyplot as plt
import seaborn as sns
# pruebas estadísticas
import statsmodels.api as sm
from scipy import stats


# In[2]:


data = pd.read_csv('reto_precios.csv')


# ## Entendimiento de los datos

# In[3]:


# descripción del tipo de datos
data.info()


# In[4]:


# se despliegan los primos 5 registros para conocer el formato de estos
data.iloc[:,10:]


# Primero, describimos los campos que contiene la tabla y el tipo de dato que contiene con el fin de conocer la información:
# 
# 1.	main_name: object nombre del anuncio
# 
# 2.	subtitle: object tipo de anuncio
# 
# 3.	link: link del anuncio
# 
# 4.	location: object dirección del inmueble
# 
# 5.	price: object precio de lista en pesos
# 
# 6.	attributes: object  metros cuadrados y número de recámaras
# 
# 7.	timestamp: object fecha
# 
# 8.	id:int64  id del registro
# 
# 9.	address: object dirección/ubicación del inmueble
# 
# 10.	since: object fecha de publición y anunciante
# 
# 11.	description: object descripción del inmueble o anuncio
# 
# 12.	vendor: object nombre del vendedor
# 
# 13.	lat: float64 latitud 
# 
# 14.	lon: float64 longitud
# 
# 15.	price_mod: float64 precio
# 
# 16.	price_currency: object moneda
# 
# 17.	since_period: object perdio de tiempo en días, semanas, etc desde el anuncio
# 
# 18.	since_value: int64  número de días, semanas, desde el anuncio
# 
# 19.	days_on_site: float64 días desde la fecha de publicación
# 
# 20.	amenities: float64 número de amenidades, es decir de facilidades que proporciona el inmueble, como alberca, lobby, seguridad, etc.
# 
# 21.	age_in_years: float64 antiguedad del departamento en años
# 
# 22.	bathrooms: float64 número de baños
# 
# 23.	cellars: float64 número de bodegas 
# 
# 24.	num_floors: float64 número de pisos donde se tiene un departamento en renta
# 
# 25.	monthly_fee: object impuesto mensual
# 
# 26.	apartments_per_floor: float64 número de departamentos por piso disponibles en el edificio
# 
# 27.	disposition: object disposición del departamento, frente si da a la calle, interno si no da a la calle y contrafente si esta ubicado en la parte trasera del edificio
# 
# 28.	parking_lots: int64 número de lugares de estacionamiento
# 
# 29.	floor_situated: float64 piso en el que se encuentra
# 
# 30.	orientation: object orientación
# 
# 31.	num_bedrooms: float64 número de cuartos
# 
# 32.	department_type: object tipo de departamentos, contiene solo tipo 'loft'
# 
# 33.	m2: float64 metros cuadrados del departamento
# 
# 34.	final_price: float64 suma de precio_mod + monthly_fee
# 
# 35.	price_square_meter: float6 precio por metro cuadrado

# ## Preparación de los datos

# En esta sección, se hará un esfuerzo para mejorar la comprensión de los datos realizando exploraciones estadísticas y numéricas para visualizar mejor la información que se tiene. El objetivo de esta sección es tener un conocimiento de la estructura
# del modelo de datos, la información que contiene la tabla y conocer más a fondo qué contienen cada registros (departamento en renta), para posteriormente continuar con la propuesta de variables que alimenten al modelo.

# In[5]:


# notemos que el calculo del precio final es igual al precio listado, por lo tanto sólo se utiliza un campo de los dos
# esto pues final_price se calcula a partir de price_mod por lo que solo debe ser considerado alguna  de las dos
data[['final_price','price_mod']].corr()


# In[6]:


# Primero obtenemos los nombres las las columnas categóricas y numéricas
# Se excluyen la columna id, final_price
# Veamos que price_square_meter se calcula a partir del precio final y el núm de m2, por lo tanto se descarta
# since_value  se descarta pues solo consideraremos los días desde la fecha de publicación
# col_num=numéricas
col_num=data.drop(['id','price_square_meter','final_price','since_value'], axis=1).select_dtypes(exclude=['object']).columns.tolist()
# ahora veremos los valores nulos que se tienen en el data set para estas variables 
data[col_num].info()


# In[7]:


# conteo de porcentaje de valores nulos en campos numericos y de valores 
for i in col_num:
    print()
    print('*** columna: \'{}\' ***'.format(i))
    porcentaje_nulos = data[i].isnull().mean() * 100
    print('valores nulos: ',porcentaje_nulos,'%')
    print()
    print(data[i].value_counts())


# In[8]:


# verifiquemos que todos los registros pertenezcan a la ciudad de México
data.drop(data[data['lat'] > 20].index, inplace=True)


# In[9]:


data


# In[10]:


# las columnas lon, lat, price_mod se van a considerar como variables numéricas continuas

# para la columna days_on_site se hara uso de la distribución para considerar si se transforma en 
# variable categórica o se mantiene como variable continua

# Notemos que la columna 'amenities' solo contiene ocho valores distintos de tipo entero, 
# por lo que se puede considerar transformar en una variable de tipo categórico

# Todos los valores de la columna age_in_years son cero, por esto noo se utilizará en el modelo

# Para la columna bathrooms se tienen seis tipos de datos distintos, por lo que puede considerarse como categórica
# se va a verificar si el dato de tener 23 baños es correcto

# La columna cellars que contiene el número de bodegas solo contiene dos tipos de registros, una y dos bodegas
# sin embargo, como se tiene un porcentaje muy significativo de valores nulos y no existe una forma de rescatar la información
# no se utilizará en el modelo

# la variable num_floors contine un gran porcentaje de valores nulos, pero representa cuantos departamentos 
# se tienen en el inmueble, por tanto no será utilizada

# El número de lugares de estacionamiento es uno, dos o tres a lo mucho, se analiza tomarla como tipo categórico

# floor_situated se considera como categórica 

# num_bedrooms se considera como categórica 

# Por último, m2 se va a considerar como variable continua


# In[11]:


# Un problema que se tiene es la cantidad de valores faltantes en amenities
# se hará la supoción de que si el anunciante no menciona alguna amedidad es por que
# el inmueble no cuenta con ninguna, así los valores nulos 
data['amenities'].fillna(0, inplace=True)


# In[12]:


# Para la columna bathrooms se tienen seis tipos de datos distintos, por lo que puede considerarse como categórica
# Primero revisemos que el registro que tenga 23 baños sea correcto
index = data['bathrooms'].idxmax()
print(index)


# In[13]:


# se buscó el anunciante y se conluye que el dato correcto es 2 baños, no 23, se realiza esta correción
data.iloc[index,:]


# In[14]:


# se reemplaza el valor por el dato correcto
data['bathrooms'] = data['bathrooms'].replace(23,2)
# y verificamos que se halla realizado el cambio
data['bathrooms'][index] 


# In[15]:


# ahora se trabajará con las variables categoricas
# no se consideran:
# 'main_name','subtitle', pues es el nombre del anuncio y la categoría
# 'link' no se utilizará como variable
# 'location','address' pues ya se tiene los campos de longitud y latitud que ubican el inmueble
# 'price','attributes','since', 'since_period' pues ya existen otros campos que considerarn esta información de forma numérica
col_cat=data.drop(['main_name','subtitle','link','location','price','attributes','address','since','since_period'], axis=1).select_dtypes(['object']).columns.tolist()
data[col_cat].info()


# In[16]:


# conteo de porcentaje de valores nulos
for i in col_cat:
    print('*** columna: \'{}\' ***'.format(i))
    porcentaje_nulos = data[i].isnull().mean() * 100
    print('valores nulos: ',round(porcentaje_nulos,2),'%')
    print()


# In[17]:


# analicemos el tipo de datos que se tienen en la columna vendor
data['vendor'].value_counts()
# notemos que se tienen muchos registros por lo cual no es conveniente tomarla como categorica
# es por eso que se va a tomar como TopVendor si tiene más de cierta cantidad de anuncios


# In[18]:


# Calcular la frecuencia de cada valor en la columna
frecuencias = data['vendor'].value_counts()

# Calcular el porcentaje que representa cada valor respecto al total de registros
porcentajes = frecuencias / len(data) * 100

porcentajes
# notamos que tres registros acumulan al menos 30% del porcentaje total de anuncios
# esto se logra al tener más de 96 ventas  por lo tanto dichos registros se considerarn como
# top vendedores 


# In[19]:


# lo anterior se realiza con el siguiente código
data['top_vendor'] = data['vendor'].map(lambda x: True if frecuencias[x] > 96 else False)


# In[20]:


# notemos que todos los precios están en moneda nacional 
# por tanto no es necesario realizar ninguna conversión y no se utilizará este campo
data['price_currency'].value_counts()


# In[21]:


# veamos los valores que se tienen en la columna de monthly_fee
data['monthly_fee'].value_counts()
# Notemos que esta información contribuye a que la variable de price_mod y final_price sean iguales pues 
# final_price = price_mod + monthly_fee
# es por ello, que sólo se utilizará price_mod


# In[22]:


# veamos los valores que contiene la columna de disposición
# esta variable sería interesante de analizar, por el momento no se centrara en hallar los datos faltantes
# una propuesta para poder completar esta información analizando la imagen del departamento y clasifcarla
data['disposition'].value_counts()


# In[23]:


# prientación del departamento 
data['orientation'].value_counts()


# In[24]:


# Se observa que para el tipo de departamento sólo se tiene de tipo loft
# tradicionalmente se sabe que un loft puede elevar el precio de un departamento,
# es por ello que esta columna se utilizará para crear una vaRibles de tipo booleana
# que indique si es loft o no
data['department_type'].value_counts()


# ##  Construcción de Analytical Base Table (ABT)

# In[25]:


# Creamos la tabla analytics basic table la cual será la tabla con la que se estará trabajando
# primero guardamos las columnas a las que no se les aplicará ninguna transformación
ABT = data[['id', 'price_mod','lon', 'lat','days_on_site','m2','bathrooms','amenities','parking_lots','num_bedrooms','top_vendor']]
# se promueve la columna 'id' como índice de la tabla
ABT = ABT.set_index('id')


# In[26]:


# visualicemos la tabla que se acaba de crear
ABT


# In[27]:


# Esta será la tabla con la cual se hará el modelo, entonces, realicemos algunas pruebas estadísticas para
# visualizar la información y conocer qué algoritmos y métricas serían adecuados
ABT.info()


# In[28]:


# Primero realicemos un análisis de correlación
matriz_corr = ABT[['price_mod','lon', 'lat','days_on_site','m2','bathrooms','amenities','parking_lots','num_bedrooms']].corr()

sns.heatmap(matriz_corr)
# notamos que no existe mucha correlación entre las variables por lo que no hay variables que sean combinaciones lineales
# y puedan ocasionar ruido al modelo

# las variables que están más correlacionadas son el número de baños, amenidades, lugares de estacionamineto y 
# número de recámaras con el precio, inclusive más que con la cantidad de metros cuadrados.


# In[29]:


# Convertimos las varibles numericas que se acumulan en pocos valores, a categóricas
#ABT['bathrooms'] = ABT['bathrooms'].astype('object')
#ABT['amenities'] = ABT['amenities'].astype('object')
#ABT['parking_lots'] = ABT['parking_lots'].astype('object')
#ABT['num_bedrooms'] = ABT['num_bedrooms'].astype('object')


# In[30]:


col_cat_ABT=ABT.select_dtypes(['object']).columns.tolist()
col_cat_ABT


# In[31]:


# se analiza el comportamiento de cada variable
for i in col_cat_ABT:
    plt.hist(ABT[i])
    plt.xlabel(i)
    plt.ylabel('Frecuencia')
    plt.title('Histograma de {}'.format(i))
    plt.show()    


# In[32]:


# Nos interesa saber a que distribución se parecen más, las distribuciones de las columnas.
# Para ello usamos los qq-plots, este tipo de plots, están en la librería stats models
# hacemos los qq-plots de las columnas numericas, mientras los puntos azules, me parezcan
# más a la linea roja, la variable tendra más similitud a una normal con los mismos parametros
# que la muestra. Es una forma visual de saber si algo se compart normal
col_num_ABT = ABT.select_dtypes(exclude=['object','bool']).columns.tolist()
for i in col_num_ABT:
    print('##########################  %s ########################## '%i)
    sm.qqplot( ABT[i], fit   = True, line  = 'q', alpha = 0.4, lw    = 2)
    plt.show()
    
# notemos que price_mod se asemeja de forma de la identida, por lo que hay sospecha de relación lineal
# hagamos el mismo análisis pero ahora removiendo outliers


# In[33]:


for i in col_num_ABT:
    print('##########################  %s ########################## '%i)
    display(ABT[i].describe())
    plt.rcParams['figure.figsize']=(8,5)
    Q1 = ABT[i].quantile(0.25)
    Q3 = ABT[i].quantile(0.75)
    IQR = Q3 - Q1
    filter = (ABT[i] >= Q1 - 1.5 * IQR) & (ABT[i] <= Q3 + 1.5 *IQR)
    sm.qqplot( ABT.loc[filter][i].dropna(), fit   = True, line  = '45', alpha = 0.4, lw    = 2)    
    plt.show()
    
# Con esto notamos que la distribución de price_mod y m2 se asemejan más a una normal al remover outliers
# la forma de latitud, longitud también mejora ligeramente
# de la variable days_on_site no se puede asumir que sigue alguna distribución en particular


# In[34]:


# A continuación, se realiza una prueba de hipótesis con el fin de analizar si 
# tiene sentido proponer un modelo de regresión 

#  Anderson-Darling test ####### pvalue > 0.05 entonces es normal
anderson_test = stats.anderson(ABT['m2'], dist='norm')
print("\nAnderson-Darling test:")
print("Test statistic:", anderson_test[0])
print("p-values:", anderson_test[1])

# se puede concluir que hay evidencia de que la variable m2 sigue una distribución normal


# In[35]:


# Una vez realizado este análisis la tabla ABT se va a guardar en otro 
# archivo el cual se va a aplicar los modelos
ABT.to_csv('tabla_abt_modelo.csv', index=False)

