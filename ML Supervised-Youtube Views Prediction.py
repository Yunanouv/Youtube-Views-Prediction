#!/usr/bin/env python
# coding: utf-8

# # Youtube Views Prediction

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# import datetime
# import plotly.express as px
# import missingno as msno
# warnings.filterwarnings('ignore')
# 
# %matplotlib inline

# # Load and Describe Data

# -----

# ## Load Data

# In[2]:


dfraw = pd.read_excel('youtube_statistics.xlsx')
dfraw.sample(5)


# Fitur pada dataset :
# 
# ● `trending_date`: tanggal ketika video trending  
# ● `title`: judul video  
# ● `channel_title`: nama channel  
# ● `category_id`: kategori video dalam label encoding  
# ● `publish_time`: waktu publish video  
# ● `tags`: tag yang digunakan pada video  
# ● `views`: jumlah views video  
# ● `likes`: jumlah likes video  
# ● `dislikes`: jumlah dislikes video  
# ● `comment_count`: jumlah komentar pada video  
# ● `comments_disabled`: apakah status komentar dinonaktifkan pada video  
# ● `ratings_disabled`: apakah rating dinonaktifkan pada video  
# ● `video_error_or_removed`: apakah video error atau sudah dihapus saat ini  
# ● `description`: deskripsi video  
# ● `No_tags`: jumlah tags yang digunakan  
# ● `desc_len`: panjang kata deskripsi video  
# ● `len_title`: panjang kata judul video  
# ● `publish_date`: tanggal publish video  

# ## Data Exploration

# In[3]:


dfraw.info()


# 1. Terdapat 36.791 entries  
# 2. Kolom `trending_date` dan `publish_time` bisa diubah tipe data nya untuk menyesuaikan dengan data.
# 3. Kolom `No_tags` sebaiknya diubah menjadi lower case agar memudahkan penyebutan variabel.

# In[4]:


# Mengubah nama kolom menjadi lower case
dfraw.columns = dfraw.columns.str.lower()


# In[5]:


# Mengecek null value
dfraw.isnull().sum()


# Only one column has null value. Amazing!  
# Karena kolom yang memiliki null value adalah 'description'. Maka, tidak perlu handling null value karena nantinya kolom ini tidak akan digunakan untuk pemodelan.

# In[6]:


# Mengecek duplicated rows
dfraw.duplicated().sum()


# Dataset kita ternyata memiliki banyak sekali duplicated rows.

# In[7]:


# Mengkategorikan fitur sesuai dari tipe data
num = ['category_id','views', 'likes', 'dislikes', 'comment_count', 'no_tags', 'desc_len', 'len_title']
cat = ['trending_date', 'channel_title','publish_time', 'tags', 'comments_disabled', 'ratings_disabled', 'video_error_or_removed']

# Membatasi angka setelah koma hingga hanya 3 angka
pd.set_option('display.float_format', lambda x: '%.3f' % x)

dfraw[num].describe()


# Notes :  
# 
# 1. Fitur numerical tidak ada null value.  
# 2. Nilai min-max nya terlampau ekstrem, kecuali pada `no_tags` dan `len_title` yang mungkin masih masuk akal.  
# 3. Nilai mean-median pada fitur `views`, `likes`, `dislikes`, dan `comment_count` sangat jauh. Hal ini mungkin disebabkan ada beberapa video yang trending.  
# 4. Perlu dilakukan pengecekan fitur pada nilai-nilai ekstrem

# In[8]:


dfraw[cat].describe()


# 1. Pada kolom `tags` tidak terdapat value yang paling sering muncul. Hal ini mungkin disebabkan oleh beragam jenis penulisan tag.   
# 2. Kita bisa melihat bahwa tidak banyak video yang komen nya dibatasi atau rating dibatasi atau video yg error maupun dihapus. Terlihat dari hampir semua value bernilai False.  

# ------

# # Exploratory Data Analysis

# -------

# ## Univariate Analysis

# ### Unique Value From Features

# Pada tahap awal inisiasi, kami mengecek beberapa fitur untuk melihat unique value nya sehingga bisa memberikan gambaran awal terkait fitur tersebut.

# In[9]:


dfraw['category_id'].unique()


# `category_id` dimulai dari angka 1-43. Yang unik nya adalah kategori ini tidak berurutan.

# In[10]:


dfraw['title'].sample(5)


# `title` dari video-video youtube ini sepertinya berasal dari youtuber di India, terlihat banyak tulisan Hindi. Dari sampel diatas juga terlihat bahwa banyak ketidakseragaman, mulai dari bahasa yang berbeda, lower/upper case, panjang judul, dll.

# In[11]:


dfraw['trending_date'].sample(5)


# In[12]:


dfraw['channel_title'].sample(5)


# In[13]:


dfraw['tags'].sample(5)


# Pada kolom `channel_title` dan `tags` juga terlihat beragam macam value nya.

# ### Outliers

# In[14]:


# Melihat penampilan outlier

plt.figure(figsize=(10, 5))
for i in range(0, len(num)):
    plt.subplot(1, len(num), i+1)
    sns.boxplot(dfraw[num[i]], color='red', orient='v')
    plt.title(num[i])
    plt.tight_layout()


# Hampir semua kolom pada fitur numerik memiliki outliers yang ekstrem.

# ### Distribusi Data

# In[15]:


# Melihat grafik distribusi
sub1 = ['no_tags', 'desc_len', 'len_title']

plt.figure(figsize=(12, 4))
for i in range(0, len(sub1)):
    plt.subplot(1, 3, i+1)
    sns.distplot(dfraw[sub1[i]], color='blue')
    plt.tight_layout()


# Kolom `no_tags` memiliki distribusi data yang paling baik diantara ketiganya diatas.  
# Kolom `desc_len` termasuk right-skewed sedangkan kolom `len_title` kebalikannya yaitu left-skewed.

# In[16]:


# Melihat grafik distribusi
sub2 = ['views', 'comment_count']

plt.figure(figsize=(16, 4))
for i in range(0, len(sub2)):
    plt.subplot(1, 2, i+1)
    sns.distplot(dfraw[sub2[i]], color='blue')
    plt.tight_layout()


# In[17]:


# Melihat grafik distribusi
sub3 = ['likes', 'dislikes']

plt.figure(figsize=(16, 4))
for i in range(0, len(sub3)):
    plt.subplot(1, 2, i+1)
    sns.distplot(dfraw[sub3[i]], color='blue')
    plt.tight_layout()


# Grafik dari fitur 'views', 'comment_count', 'likes', dan 'dislikes' menunjukkan distribusi data yang sangat tidak wajar yang terlihat dari grafik yang runcing sekali.

# In[18]:


from scipy.stats import kurtosis
print(f"Kurtosis Co-efficient Likes: {round(dfraw.likes.kurtosis(), 3)}")
print(f"Kurtosis Co-efficient Dislikes: {round(dfraw.dislikes.kurtosis(), 3)}")
print(f"Kurtosis Co-efficient Views: {round(dfraw.views.kurtosis(), 3)}")
print(f"Kurtosis Co-efficient Comment Count: {round(dfraw.comment_count.kurtosis(), 3)}")


# Kurtosis leptokurtic. Tingkat kurtosis nya sangat tinggi. Hal ini mungkin disebabkan oleh outliers yang ekstrim namun wajar dalam dunia trending youtube (bukan input error).

# ### Check Ekstrem Value

# Jika dilihat dari grafik, yang menunjukkan outliers adalah jumlah 'views' mulai dari 4 jutaan keatas.

# In[19]:


# Mengecek baris dengan jumlah views tertinggi
extreme= dfraw.sort_values(['views'], ascending = False).reset_index(drop='index')
extreme.head()


# Video "YouTube Rewind: The Shape of 2017" yang dipublikasikan oleh saluran "YouTube Spotlight" menjadi video dengan view terbanyak, dengan jumlah views sebanyak 125,432,237.
# 
# Faktor yang mungkin menjelaskan mengapa video ini memiliki jumlah views yang sangat tinggi: Video ini adalah bagian dari seri tahunan "YouTube Rewind" yang dirilis oleh saluran resmi YouTube. Video ini biasanya merangkum dan memperingati konten populer dan peristiwa yang terjadi selama tahun tersebut di platform YouTube. Karena ini adalah video resmi dari YouTube sendiri, banyak pengguna yang tertarik untuk melihat bagaimana YouTube merangkum peristiwa tahun tersebut.

# In[20]:


# Mengecek baris dengan jumlah views terendah
extreme= dfraw.sort_values(['views'], ascending = True).reset_index(drop='index')
extreme.head()


# Sama seperti nilai esktrem dengan banyak views, hal serupa juga terjadi pada nilai dengan jumlah `views` yang sedikit, dimana hal tersebut mempengaruhi juga jumlah `likes` dan `dislikes` nya.

# ## Multivariate Analysis

# ### Korelasi Numerik

# In[21]:


# Plot korelasi heatmap
corr_matrix = dfraw.corr()
plt.figure(figsize=(7, 4))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# Berdasarkan grafik heatmap diatas, ada beberapa kesimpulan yang bisa kita tarik diantaranya:
# 
# * Fitur`likes` memiliki nilai korelasi yang tinggi (0.85).
# * Selain `likes` ada `dislikes` dan `comment_count` yang memiliki nilai tinggi namun tidak masuk kedalam kategori redundan.  
# * `dislikes` memiliki nilai korelasi 0.54 (positif korelasi) dimana semakin banyak suatu konten mendapat dislike maka bisa nilai dari views juga akan meningkat.  
# * `comment_count` memiliki angka korelasi 0.67 dimana termasuk kedalam korelasi positif, sehingga semakin banyak suatu konten mendapat comments maka semakin tinggi pula angka views.  
# * `comments_disabled` dan `ratings_disabled` juga menunjukkan korelasi yang lumayan baik.

# ### Korelasi Fitur Dengan Label

# In[22]:


# Pairplot untuk melihat korelasi dengan label : views
plt.figure(figsize=(24,8))
fig = sns.pairplot(data=dfraw, x_vars=['category_id','likes','dislikes','comment_count','no_tags','desc_len','len_title'], y_vars=['views'], height=6, aspect=0.75);
fig.tight_layout();


# Terlihat bahwa semakin tinggi views, maka semakin tinggi pula jumlah likes, dislikes, dan comment nya. Sedangkan untuk banyaknya tag, panjang deskripsi, dan panjang judul video kebanyakan jumlah nya merata namun cenderung hanya untuk views sekitar 0-2 juta.

# ### Korelasi Label Dengan Category

# In[23]:


# Visualisasi Views Dengan Category_id

sns.catplot(data=dfraw, x='category_id', y='views', hue='likes')


# Dari graph ini bisa kita simpulkan bahwa kategori konten yang memiliki jumlah views tinggi cenderung memiliki jumlah likes yang tinggi dan hanya beberapa konten saja yang dapat mendapatkan jumlah views yang terlampau tinggi dibanding dengan konten - konten lain.

# ### Korelasi Comment dan Category Terhadap Views

# In[24]:


sns.catplot(data=dfraw, x='category_id', y='comment_count', hue='views')


# Dari grafik ini dapat disimpulkan bahwa kategori konten mempengaruhi jumlah comment yang ada dan juga total views, sehingga hanya beberapa konten saja yang mendapatkan jumlah comments dan views yang melebihi rata-rata comments atau views.

# ### Korelasi Label Dengan Comment Disabled

# In[25]:


sns.catplot(data=dfraw, x='comments_disabled', y='views', hue='likes')


# Dari grafik ini bisa disimpulkan bahwa konten dengan comment disable memiliki jumlah views yang rendah dan jumlah likes yang rendah pula.

# -------

# # Data Preparation

# -----

# ## Data Cleaning

# ### Drop Columns

# **Duplicated Rows**

# Pada tahap Data Exploration sebelumnya, dataset tercatat memiliki 4229 data duplikat, sehingga kita harus menghapusnya.

# In[26]:


# Menghapus duplicated rows
dfraw = dfraw.drop_duplicates(keep='first')

# Pengecekan kembali apakah masih terdapat duplikat atau tidak
print(f'Jumlah duplikat: {dfraw.duplicated().sum()}')


# **Irrelevant Features**

# Beberapa fitur terlihat kurang relevan dan unique value yang dimiliki terlalu luas seperti yang sudah kita analisis sebelumnya. Sehingga diasumsikan bahwa fitur tersebut tidak memiliki pola tertentu. Fitur tersebut adalah `title`, `channel_title`, `publish_time`, `tags`, `description` yang akan dihapus.

# In[27]:


dfraw = dfraw.drop(['title', 'channel_title', 'publish_time', 'tags', 'description'], axis=1)
dfraw.head(3)


# ### Missing Values

# Sesuai yang telah disebutkan saat proses Data Exploration, data hilang (Null) hanya ditemukan pada kolom `description` meskipun hal tersebut masih dapat dikatakan masuk akal karena terkadang terdapat pemilik vidio yang tidak menuliskan apapun pada kolom deskripsi vidionya. Namun, karena kolom tersebut termasuk fitur yang irrelevant dan telah dihapus, maka tidak perlu menangani missing value.

# ### Handling Outliers

# In[28]:


df_cleansing = dfraw.copy()


# In[29]:


def outlier_del(df, column, mode):
    q1 = df.iloc[:,column].quantile(0.25)
    q3 = df.iloc[:,column].quantile(0.75)
    iqr = q3-q1
    lower_tail = q1 - (1.5 * iqr)
    upper_tail = q3 + (1.5 * iqr)
    nama_kolom = df.columns[column]
    jumlah_outliers = df[(df.iloc[:,column] <= lower_tail)|(df.iloc[:,column] >= upper_tail)].iloc[:,column].count()
    total_row = df.iloc[:,column].count()
    persentase_outliers = round(((jumlah_outliers/total_row)*100),2)
    if mode == 'summary':
        return print('Jumlah Outliers pada kolom ', nama_kolom, ' :', jumlah_outliers, ' dan persentase outliers:', persentase_outliers, '%')
    elif mode == 'df':
        return df[(df.iloc[:,column] >= lower_tail)&(df.iloc[:,column] <= upper_tail)]
    else :
        return print('periksa mode yang diinputkan')


# In[30]:


column = [2, 3, 4, 5, 9, 10, 11]

for i in range(0, len(column)):
    outlier_del(df_cleansing, column[i], 'summary')


# seperti yang juga telah dinyatakan saat Data Exploration bahwa terdapat banyak sekali outlier, selanjutnya akan dilakukan penghapusan outlier pada kolom-kolom yang memiliki outlier dibawah 5% yaitu kolom `no_tags` dan `desc_len`.

# In[31]:


# Penghapusan Outlier
df_cleansing = df_cleansing[df_cleansing.index.isin(outlier_del(dfraw, 9, 'df').reset_index()['index'])]
df_cleansing = df_cleansing[df_cleansing.index.isin(outlier_del(dfraw, 10, 'df').reset_index()['index'])]
df_cleansing.shape


# In[32]:


# Membuat Plot Untuk Membandingkan dengan kondisi sebelum penghapusan outlier
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.distplot(dfraw.no_tags)
plt.title('No_tags - Outliers')
plt.subplot(1, 2, 2)
sns.distplot(df_cleansing.no_tags)
plt.title('No_tags - No Outliers')
plt.show()


# In[33]:


plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.distplot(dfraw.desc_len)
plt.title('Desc_len - Outliers')
plt.subplot(1, 2, 2)
sns.distplot(df_cleansing.desc_len)
plt.title('Desc_len - Outliers')
plt.show()


# ## Data Preparation

# ### Feature Engineering

# Feature engineering yang dilakukan adalah menambahkan fitur baru yaitu is_weekend, yang mengecek apakah sebuah video yang dipublish di weekend memberikan dampak yang signifikan.

# In[34]:


df_cleansing['publish_date'] = df_cleansing['publish_date'].astype(str)
df_cleansing['publish_date'].head()


# In[35]:


timeget = []
for i in df_cleansing['publish_date'].str.split():
    time_split = i[:5]
    date_string = ' '.join(time_split)
    timeget.append(date_string)


# In[36]:


df_cleansing['publish_date'] =  pd.to_datetime(timeget)
df_cleansing['day_of_week'] = df_cleansing['publish_date'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[37]:


def is_weekend(dmap):
    if dmap.weekday() < 5:
        return False
    else:
        return True
df_cleansing['is_weekend'] = df_cleansing['publish_date'].apply(is_weekend)

df_cleansing.head()


# ### Log Transformation

# Dikarenakan persebaran data yang kita miliki sangat tidak normal, maka kita akan melakukan transformasi fitur. Jika outliers pada fitur `no_tags` dan `desc_len` dihapus, maka untuk outliers pada fitur-fitur yang lain akan ditangani dengan menggunakan Log Transformation. Kita bisa menggunakan Log Transformation sehingga distribusi data mendekati Gaussian.

# In[38]:


# Log Transformation for Views

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Plotting
g = sns.distplot(df_cleansing['views'],kde=True, ax=ax[0,0])
ax[0,0].set_title('Views - Before')
ax[0,0].set_xlabel('')

g = sns.boxplot(df_cleansing['views'],color='red',orient='v', ax=ax[0,1])
ax[0,1].set_title('Views - Outliers')
ax[0,1].set_xlabel('')

g = sns.distplot(np.log1p(df_cleansing['views']+1),kde=True, ax=ax[1,0])
ax[1,0].set_title('Views - Log Transformation')
ax[1,0].set_xlabel('')

g = sns.boxplot(np.log1p(df_cleansing['views']+1),color='red',orient='v', ax=ax[1,1])
ax[1,1].set_title('View - Log transformation')
ax[1,1].set_xlabel('')

plt.show()


# In[39]:


# Log Transformation for Likes

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Plotting
g = sns.distplot(df_cleansing['likes'],kde=True, ax=ax[0,0])
ax[0,0].set_title('Likes - Before')
ax[0,0].set_xlabel('')

g = sns.boxplot(df_cleansing['likes'],color='red',orient='v', ax=ax[0,1])
ax[0,1].set_title('Likes - Outliers')
ax[0,1].set_xlabel('')

g = sns.distplot(np.log1p(df_cleansing['likes']+1),kde=True, ax=ax[1,0])
ax[1,0].set_title('Likes - Log Transformation')
ax[1,0].set_xlabel('')

g = sns.boxplot(np.log1p(df_cleansing['likes']+1),color='red',orient='v', ax=ax[1,1])
ax[1,1].set_title('Likes - Log transformation')
ax[1,1].set_xlabel('')

plt.show()


# In[40]:


# Log Transformation for Dislikes

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Plotting
g = sns.distplot(df_cleansing['dislikes'],kde=True, ax=ax[0,0])
ax[0,0].set_title('Dislikes - Before')
ax[0,0].set_xlabel('')

g = sns.boxplot(df_cleansing['dislikes'],color='red',orient='v', ax=ax[0,1])
ax[0,1].set_title('Dislikes - Outliers')
ax[0,1].set_xlabel('')

g = sns.distplot(np.log1p(df_cleansing['dislikes']+1),kde=True, ax=ax[1,0])
ax[1,0].set_title('Dislikes - Log Transformation')
ax[1,0].set_xlabel('')

g = sns.boxplot(np.log1p(df_cleansing['views']+1),color='red',orient='v', ax=ax[1,1])
ax[1,1].set_title('Dislikes - Log transformation')
ax[1,1].set_xlabel('')

plt.show()


# In[41]:


# Log Transformation for Comment Count

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Plotting
g = sns.distplot(df_cleansing['comment_count'],kde=True, ax=ax[0,0])
ax[0,0].set_title('Comment Count - Before')
ax[0,0].set_xlabel('')

g = sns.boxplot(df_cleansing['comment_count'],color='red',orient='v', ax=ax[0,1])
ax[0,1].set_title('Comment Count - Outliers')
ax[0,1].set_xlabel('')

g = sns.distplot(np.log1p(df_cleansing['comment_count']+1),kde=True, ax=ax[1,0])
ax[1,0].set_title('Comment Count - Log Transformation')
ax[1,0].set_xlabel('')

g = sns.boxplot(np.log1p(df_cleansing['comment_count']+1),color='red',orient='v', ax=ax[1,1])
ax[1,1].set_title('Comment Count - Log transformation')
ax[1,1].set_xlabel('')

plt.show()


# In[42]:


# Log Transformation for No Tags

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

# Plotting
g = sns.distplot(df_cleansing['no_tags'],kde=True, ax=ax1)
ax1.set_title('No_tags - Before')

g = sns.distplot(np.log1p(df_cleansing['no_tags']+1),kde=True, ax=ax2)
ax2.set_title('No_tags - Log Transformation')

plt.show()


# In[43]:


# Log Transformation for Desc Len

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

# Plotting
g = sns.distplot(df_cleansing['desc_len'],kde=True, ax=ax1)
ax1.set_title('Desc_len - Before')

g = sns.distplot(np.log1p(df_cleansing['desc_len']+1),kde=True, ax=ax2)
ax2.set_title('Desc_len - Log Transformation')

plt.show()


# In[44]:


s_features = ['category_id','views','likes','dislikes','comment_count','no_tags','desc_len','len_title']

df = df_cleansing.copy()
for var in s_features:
    df['log_'+var]= (df_cleansing[var]+1).apply(np.log)


# In[45]:


df = df.drop(['day_of_week'], axis=1)
df.columns


# Setelah data terdistribusi secara lebih normal, maka selanjutny adalah melakukan train dan test split data.

# ## Train & Test Split

# In[46]:


X = df[['log_category_id','log_likes','log_dislikes','log_comment_count','log_no_tags','log_desc_len','log_len_title','is_weekend','comments_disabled', 'ratings_disabled','video_error_or_removed']]
y = df[['log_views']]


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)


# In[48]:


# Membuat data train dan data test
data_train = pd.concat([X_train, y_train], axis=1)
data_test = pd.concat([X_test, y_test], axis=1)


# Untuk mencegah terjadinya data leak, maka normalisasi dilakukan setelah train test split.

# ## Pre-Processing

# In[49]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
std_train = scaler.fit_transform(data_train)
std_test = scaler.transform(data_test)

# Mengembalikan nilai ke dataframe
std_train = pd.DataFrame(std_train, columns=data_train.columns)
std_test = pd.DataFrame(std_test, columns=data_test.columns)


# In[50]:


Xtrain_std = std_train.drop('log_views', axis=1)
ytrain_std = std_train.log_views
Xtest_std = std_test.drop('log_views', axis=1)
ytest_std = std_test.log_views


# -------

# # Modelling and Evaluation

# ------

# ## Linear Regression

# In[51]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(Xtrain_std, ytrain_std)


# In[52]:


y_pred = regressor.predict(Xtest_std)
y_pred


# Seperti semua fungsi NumPy yang dirancang untuk melakukan perhitungan ini dengan array dan struktur array-like.
# 
# e adalah konstanta matematika yang disebut bilangan Euler, kira-kira 2,718281. Nilai ini memiliki hubungan matematis yang erat dengan pi dan the slope of the curve e^x sama dengan nilainya di setiap titik.
# 
# Maka, fungsi np.exp berguna saat kita perlu menghitung e^x untuk matriks angka yang besar atau mengembalikan fungsi log ke base e (Euler's number).

# In[53]:


# np.exp function transformation
## Mengembalikan fungsi log ke base e
np.exp(y_pred).round()


# In[54]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_regression(model, y_pred, Xtrain_std, Xtest_std, ytrain_std,  ytest_std):
    print("MAE: " , mean_absolute_error(ytest_std, y_pred))
    print("RMSE: " , mean_squared_error(ytest_std, y_pred, squared=False))
    print('R2 score: ', r2_score(ytest_std, y_pred))
    print('Coefficients: ', regressor.coef_)
    print('Intercept: ', regressor.intercept_)

eval_regression(regressor, y_pred, Xtrain_std, Xtest_std, ytrain_std, ytest_std)


# -----

# ## Decision Tree Regression

# In[55]:


from sklearn.tree import DecisionTreeRegressor

dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(Xtrain_std, ytrain_std)


# In[56]:


y_pred = dt_regressor.predict(Xtest_std)
y_pred


# In[57]:


# Evaluasi Decision Tree Regression
eval_regression(dt_regressor, y_pred, Xtrain_std, Xtest_std, ytrain_std, ytest_std)


# ---

# ## Random Forest Regression

# ### Tuning Hyperparameter

# In[58]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

hyperparameters = dict(
                       n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 10)], # Jumlah subtree
                       criterion = ['squared_error','absolute_error'],
                       max_depth = [int(x) for x in np.linspace(10, 110, num = 5)],  # Maximum kedalaman tree
                       n_jobs = [-1], # Core untuk parallel computation. -1 untuk menggunakan semua core
                      )

# Init
rf_reg = RandomForestRegressor(random_state=42)
rf_tuned = RandomizedSearchCV(rf_reg, hyperparameters, cv=5, random_state=42, scoring='r2')
rf_tuned.fit(Xtrain_std, ytrain_std)


# In[60]:


def show_best_hyperparameter(model, hyperparameters):
    for key, value in hyperparameters.items() :
        print('Best '+key+':', model.get_params()[key])

show_best_hyperparameter(rf_tuned.best_estimator_, hyperparameters)


# In[61]:


from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=85, random_state=42, n_jobs=-1)
rf_regressor.fit(Xtrain_std, ytrain_std)


# In[63]:


y_pred = rf_regressor.predict(Xtest_std)
y_pred


# In[64]:


# Evaluasi Random Forest Regression
eval_regression(rf_regressor, y_pred, Xtrain_std, Xtest_std, ytrain_std, ytest_std)


# ---

# ## XG Boost

# In[69]:


get_ipython().system('pip install -U hgboost')


# ### Tuning Hyperparameter

# In[70]:


from hgboost import hgboost

# Initialize library.
hgb = hgboost(
    max_eval=1000,      # Search space is based  on the number of evaluations.
    threshold=0.5,     # Classification threshold. In case of two-class model this is 0.5.
    cv=5,              # k-folds cross-validation.
    test_size=0.2,     # Percentage split for the testset.
    val_size=0.2,      # Percentage split for the validationset.
    top_cv_evals=10,   # Number of top best performing models that is evaluated.
#     is_unbalance=True, # Control the balance of positive and negative weights, useful for unbalanced classes.
    random_state=42, # Fix the random state to create reproducible results.
    n_jobs=-1,         # The number of CPU jobs to run in parallel. -1 means using all processors.
    gpu=False,         # Compute using GPU in case of True.
    verbose=3,         # Print progress to screen.
)


# In[74]:


xgb_reg_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6, 7 ],
        'eta' : [0.1,0.01, 0.05],
        'n_estimators' : [100, 200, 400,600]
}

results  = hgb.xgboost_reg(X, np.array(y), eval_metric='rmse')


# In[75]:


# Results are stored in the object and can be found in:
print(hgb.results)

# Results are also returned by the model:
print(results.keys())

# The params contains the parameters to create the best performing model.
print(results['params'])


# In[76]:


import xgboost as xgb

xgb_regressor = xgb.XGBRegressor(gamma=0, learning_rate=0.1, max_depth=27, min_child_weight=3, n_estimators=140, reg_lambda=5.0, subsample=0.93)
xgb_regressor.fit(Xtrain_std, ytrain_std)


# In[77]:


y_pred = xgb_regressor.predict(Xtest_std)
y_pred


# In[78]:


# Evaluasi Gradient Boosting Regression
eval_regression(xgb_regressor, y_pred, Xtrain_std, Xtest_std, ytrain_std, ytest_std)


# In[79]:


import shap
explainer = shap.TreeExplainer(xgb_regressor)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# ----------

# # Kesimpulan

# 1. Dari hasil pemodelan yang telah dilakukan terhadap dataset Statistik Youtube untuk memprediksi jumlah views, kita bisa menyimpulkan bahwa model terbaik yaitu XG Boost yang memberikan R2 Score : 0.91 serta nilai MAE dan RMSE yang kecil yaitu 0.02 dan 0.04. Artinya model menunjukkan performa yang baik dalam memprediksi target.  
# 
# 2. Model terbaik kedua adalah model Random Forest dengan R2 Score : 0.89 serta nilai MAE dan RMSE yang tidak jauh berbeda dari model XG Boost, yaitu : 0.03 dan 0.04.  
# 
# 3. Rekomendasi bisnis yang bisa diberikan untuk meningkatkan jumlah views adalah  :  
#     * Bisa memanfaatkan waktu publish, yaitu di hari weekend dimana hal tersebut memberikan pengaruh dalam peningkatan jumlah views.  Jika hal ini dilakukan, kemungkinan besar kita akan memperoleh hasil yang signifikan yaitu :  
#         a. Jumlah views bertambah  
#         b. Jika jumlah views bertambah, otomatis jumlah likes, dislikes, maupun comment akan meningkat.  
#         c. Jika hal tersebut semakin banyak, maka algoritma Youtube akan mendeteksi video sebagai video trending.
#     * Tidak membatas komen dan rating. Dari hasil analisa data terlihat bahwa jika video membatasi komen dan rating, maka minat seseorang untuk menonton video tersebut juga akan berkurang, sehingga jumlah interaksi dengan video juga akan berkurang.
#     * Kategori konten juga mempengaruhi jumlah views. Posting lah konten-konten yang sering disukai oleh penonton, lalu sesuaikan tag (baik judul tag maupun jumlah tag) dengan proporsi yang bagus.
