import pickle
import tensorflow

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import numpy as np

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))

# print(feature_list)

# shape se bhi check krlo its 44k cross 2048 dimension

# print(np.array(feature_list).shape)

# Every image ka 2048 features hai

filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))

# include top false krke upper layer apne hisaaab ki banayi hai,
# weights imagenet ke dataset pe jo model tha wohi use kiya h, uss hisaab ke weights hai
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Import test image

img = image.load_img( 'samplePhoto/che.jpg' , target_size = (224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)


# naye image ke features ko database ke features se distance calculate krna hai and 5 best recommendation dene hai

neighbors = NearestNeighbors(n_neighbors= 5, algorithm='brute',metric='euclidean')    # here cosine distance can also be used

neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print(indices)  # 2d list hai

for file in indices[0]:
    print(filenames[file])

# to display images we use opencv module
# When using new image there no use of slicing ---- indices[0][1:6]
# and n_neighbors will be kept 5 only


