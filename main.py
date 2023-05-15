import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('heart.csv')
data = dataset.loc[:,['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']]

sc = MinMaxScaler(feature_range=(0, 1))
data = sc.fit_transform(data)
# print(data_normalized)

# 3. Clustering dengan k-Means k=2
# clustering = KMeans(n_clusters=2, init='random', n_init=1)
# clusters = clustering.fit_predict(data)
# print("Hasil Clustering:\n", clusters)

# 4. Clustering dengan single,average,complete
# clustering=AgglomerativeClustering(n_clusters=2,linkage='average')
# # clustering=AgglomerativeClustering(n_clusters=2,linkage='single')
# # clustering=AgglomerativeClustering(n_clusters=2,linkage='complete')
# clusters= clustering.fit_predict(data)
# print('\nHasil clustering:\n', clusters)


#5.

cal_val = []
for i in range(10):
    clustering = KMeans(n_clusters=3, init='random', n_init=1)
    clusters = clustering.fit_predict(data)
    print(f'\nHasil clustering {i}:\n', clusters)
    print('\nSSE = :\n', clustering.inertia_)
    cal_val.append(clustering.inertia_)
print("Nilai Terkecil: ",min(cal_val))
print("Pada Index ke: ",pd.Series(cal_val).idxmin())

