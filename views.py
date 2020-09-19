from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
from keras.model import load_model
from keras.preprocessing import image
import json
import tenserflow as tf
from tensorflow import Graph, Session

img_height, img_width = 224,224
with open('filepath.json') as f:
    lableInfo = f.read()

lableInfo = json.load(lableInfo)
# model = load_model('./models/modelfile.h5')
model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()

    with tf_Session.as_default():
        model = load_model('./models/modelfile.h5')


# Create your views here.
def index(request):
    return render(request,'index.html')

def prediction(request):
    fileobj = request.FILES['image']
    fs = FileSystemStorage()
    fs.save(fileobj.name, fileobj)
    image = fileobj.name.split('.')[:1]
    testimage = '.'+ fs.url(fileobj)

    img = image.load_image(testimage, target_size=(img_height,img_height))
    x = img_to_array(img)
    x = x//225
    x = x.reshape(1, img_height, img_width,3)
    with model_graph.as_default():
        with tf_Session.as_default():
            pred = model.predict(x)
     
    import numpy as np
    predictedlable = lableInfo[str(np.argmax(pred[0]))]

    context = {'file': image, 'filedisplay': fs.url(fileobj), 'predictedlable': predictedlable[1]}
    return render(request,'index.html', context)


def viewdatabase(request):
    import os
    listofimg = os.listdie('./static/media/')
    listofimgpath = ['.static/media/'+i for i in listofimg]
    context = {'listofimg': listofimgpath}
    return render(request, 'viewDB.html', context)