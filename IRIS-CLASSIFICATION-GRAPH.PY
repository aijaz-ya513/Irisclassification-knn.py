#load datasets required models
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#load dataset
iris=datasets.load_iris()

features=iris.data
labels=iris.target

target_names=iris.target_names
feature_names=iris.feature_names



#train classifier
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(features, labels)

#new_sample
new_sample=([[5.1,3.5,1.4,0.2]])
pred=clf.predict(new_sample)
print ("predicted class: ",target_names[pred[0]])

x_index=0
y_index=1
#plotting
colors=['red','blue','green']
for i in range(3):
  plt.scatter(features[labels == i,x_index],features[labels == i,y_index],label=target_names[i],c=colors[i])
  plt.xlabel(feature_names[x_index])
  plt.ylabel(feature_names[y_index])
  plt.title("iris flowers classification (knn)")
  plt.tight_layout()
  plt.legend()
  plt.show()






