import json
import pandas as pd
import pickle
import urllib.request
from PIL import Image
import requests

# Opening JSON file
f = open('captions_train2014.json', )

# returns JSON object as
# a dictionary
data = json.load(f)
df_image_detail=pd.DataFrame(data['images']) # contains the image_url

#This library uses HuggingFace’s transformers model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
train_df=pd.read_csv("train_df.csv")
def predict_image(text):
    distances, indices = loaded_model.kneighbors(model.encode([text]))
    result = df_image_detail[df_image_detail['id'].isin(train_df.iloc[indices[0]]['image_id'])]

    image_arr=[]
    for i in range(2):

        image_arr.append(Image.open(requests.get(result.iloc[i]["coco_url"], stream=True).raw))

    return image_arr[0],image_arr[1]

