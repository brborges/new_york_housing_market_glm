# In[ ]: Importação dos pacotes necessários
    
import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
import plotly.graph_objs as go # gráfico 3D
from scipy.stats import pearsonr # correlações de Pearson
from sklearn.preprocessing import LabelEncoder # transformação de dados

# In[ ]:
#############################################################################
#         REGRESSÃO COM UMA VARIÁVEL EXPLICATIVA (X) QUALITATIVA            #
#             EXEMPLO 03 - CARREGAMENTO DA BASE DE DADOS                    #
#############################################################################

df_housing = pd.read_csv('data-raw/NY-House-Dataset.csv',delimiter=',',encoding='utf-8')
df_housing

#Características das variáveis do dataset
df_housing.info()

#Estatísticas univariadas
df_housing.describe()



# In[ ]:
#Tabela de frequências da variável 'TYPE'
#Função 'value_counts' do pacote 'pandas' sem e com o argumento 'normalize'
#para gerar, respectivamente, as contagens e os percentuais
contagem = df_housing['TYPE'].value_counts(dropna=False)
percent = df_housing['TYPE'].value_counts(dropna=False, normalize=True)
pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=False)


# In[ ]:
df_housing = df_housing[['TYPE', 'PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT']]
df_housing


# In[ ]: Dummizando a variável 'TYPE'. O código abaixo automaticamente fará: 
# a)o estabelecimento de dummies que representarão cada uma das regiões do dataset; 
# b)removerá a variável original a partir da qual houve a dummização; 
# c)estabelecerá como categoria de referência a primeira categoria, ou seja,
# a categoria 'America_do_sul' por meio do argumento 'drop_first=True'.

df_housing['TYPE'] = df_housing['TYPE'].astype("category")


df_housing_dummies = pd.get_dummies(df_housing, columns=['TYPE'],
                                      drop_first=True, prefix_sep='_')

df_housing_dummies.head(10)


# In[ ]:

dict = {
    'TYPE_Coming Soon': 'TYPE_Coming_Soon',
    'TYPE_Condo for sale': 'TYPE_Condo_for_sale',
    'TYPE_Condop for sale': 'TYPE_Condop_for_sale',
    'TYPE_Contingent': 'TYPE_Contingent',
    'TYPE_For sale': 'TYPE_For_sale',
    'TYPE_Foreclosure': 'TYPE_Foreclosure',
    'TYPE_House for sale': 'TYPE_House_for_sale',
    'TYPE_Land for sale': 'TYPE_Land_for_sale',
    'TYPE_Mobile house for sale': 'TYPE_Mobile_house_for_sale',
    'TYPE_Multi-family home for sale': 'TYPE_Multi_family_home_for_sale',
    'TYPE_Pending': 'TYPE_Pending',
    'TYPE_Townhouse for sale': 'TYPE_Townhouse_for_sale'
}

df_housing_dummies.rename(columns=dict, inplace=True)

# In[ ]: Estimação do modelo de regressão múltipla com n-1 dummies

modelo_housing_dummies = sm.OLS.from_formula("PRICE ~ BEDS + BATH + PROPERTYSQFT + TYPE_Coming_Soon + \
                                             TYPE_Condo_for_sale + TYPE_Condop_for_sale + TYPE_Contingent + \
                                             TYPE_For_sale + TYPE_Foreclosure + TYPE_House_for_sale + \
                                             TYPE_Land_for_sale + TYPE_Mobile_house_for_sale + \
                                             TYPE_Multi_family_home_for_sale + TYPE_Pending + \
                                             TYPE_Townhouse_for_sale",
                                             df_housing_dummies).fit()



#Parâmetros do modelo
modelo_housing_dummies.summary()


# In[ ]: Plotando o modelo_corrupcao_dummies de forma interpolada

#Fitted values do 'modelo_corrupcao_dummies' no dataset 'df_housing_dummies'
df_housing_dummies['fitted'] = modelo_housing_dummies.fittedvalues
df_housing_dummies.head()




# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Wilk (n < 30)
#from scipy.stats import shapiro

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiro_francia' do pacote
#'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.tests import shapiro_francia
shapiro_francia(modelo_housing_dummies.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_housing_dummies.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[ ]: Histograma dos resíduos do modelo OLS linear

plt.figure(figsize=(10,10))
sns.histplot(data=modelo_housing_dummies.resid, kde=True, bins=30, color = 'darkorange')
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()



# In[ ]: Procedimento Stepwise

# Instalação e carregamento da função 'stepwise' do pacote
#'statstests.process'
# Autores do pacote: Helder Prado Santos e Luiz Paulo Fávero
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_housing = stepwise(modelo_housing_dummies, pvalue_limit=0.05)
modelo_step_housing


# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Wilk (n < 30)
#from scipy.stats import shapiro
#shapiro(modelo_step_housing.resid)

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiro_francia' do pacote
#'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.tests import shapiro_francia
shapiro_francia(modelo_step_housing.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_step_housing.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[ ]: Plotando os resíduos do 'modelo_step_housing' e acrescentando
#uma curva normal teórica para comparação entre as distribuições

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_housing.resid, fit=norm, kde=True, bins=20,
             color='goldenrod')
plt.xlabel('Resíduos do Modelo Linear', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Prediction


df_housing_dummies['fitted_after_step'] = modelo_step_housing.fittedvalues
df_housing_dummies.head()

# In[ ]: Plotting


plt.figure(figsize=(15, 10))

sns.scatterplot(x='PROPERTYSQFT', y='fitted_after_step', data=df_housing_dummies)
sns.scatterplot(x='PROPERTYSQFT', y='fitted', data=df_housing_dummies)

plt.show()

# In[ ]: Plotting


plt.figure(figsize=(15, 10))

sns.scatterplot(x='BATH', y='fitted_after_step', data=df_housing_dummies)
sns.scatterplot(x='BATH', y='fitted', data=df_housing_dummies)

plt.show()
