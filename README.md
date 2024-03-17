Edgar Joel Pardo 
David Hernando Ávila
Iván Andrés Trujillo Abella


Arquitectura:

Nuestro objetivo fue crear una arquitectura lo más automatizada posible, para el problema en cuestión,  en  el siguiente proyecto se utilizó python 3.x, creándose los módulos:

Preprocessing2
Models2

Las funciones usadas para realizar este ejercicio de analítica se encuentran distribuidas en los tres módulos mencionados anteriormente. Se creó el siguiente repositorio https://github.com/it-ces/Analytics-puj con el objetivo de que los resultados puedan ser rectificados y el trabajo reproducible, basta con que  se abra el cuaderno “first-work-analytics.ipynb” y se le de click al item abrir en colab, dentro de dicho repositorio. 

En el siguiente  link se puede ejecutar el código:
https://colab.research.google.com/github/it-ces/Analytics-puj/blob/main/First-work-Analytics.ipynb 


Metodología:


EDA: 

Para nuestro análisis exploratorio de datos generamos la tabla 1, que compara las variables de la base de datos con el evento de interés, de acuerdo a la naturaleza de la variable, es decir primero se define si la variable es numérica o categórica, posteriormente se utiliza el test de shapiro wilk para determinar la normalidad o no de las variables numéricas de igual forma en la tabla se presenta la cantidad de datos faltantes para saber si se puede presentar un sesgo de selección, se utiliza el t-test para las variables numéricas normales, para las variables no normales se utiliza el kruskal wallis, y por último para las variables categóricas se utiliza el test chi cuadrado.

Modelos

Se realizó un benchmark entre los modelos:

Regresión logística
Máquina de soporte vectorial
Random Forest
Adaboost


Se utilizó cross-validation  con 5- folds para la selección de los hiper parámetros, la métrica de evaluación fue el auc_roc.



