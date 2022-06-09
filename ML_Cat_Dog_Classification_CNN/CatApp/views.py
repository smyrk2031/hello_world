from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import DocumentForm
from .models import Document
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from tensorflow import Graph
import json

img_height, img_width=64,64
with open('./models/catDog.json','r') as f:
    labelInfo=f.read()

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session(target='', graph=None, config=None)
    with tf_session.as_default():
        model=load_model('./models/catvDog.h5')

labelInfo=json.loads(labelInfo)

# Create your views here.
def index(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = DocumentForm()
        obj = Document.objects.all()
 
    return render(request, 'ai_image/model_form_upload.html', {
        'form': form,
        'obj': obj
    })
    
def indexPage(request):
    return render(request, 'index.html')

def get_filepaths(request):
    print("取得したフォルダ内のファイルを全て取得する")


def predection(request):
    print("Prediction実行")
    print(request)
    print(request.POST)
    fileObj = request.FILES['filePath']
    fs=FileSystemStorage()
    print(fs)
    filepathName= fs.save(fileObj.name,fileObj)
    print(filepathName)
    filepathName=fs.url(filepathName)
    print(filepathName)

    testimage = '.' + filepathName
    print("testimageは：")
    print(testimage)
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = x / 255
    x = x.reshape(1, img_height, img_width, 3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi = model.predict(x)
            print("Value of the prediction -- ", predi)
            if(predi[0]>0.5):
                predictedLabel = 'Dog'
            else:
                predictedLabel = 'Cat'

    context={'filepathName' : filepathName,'predictedLabel':predictedLabel}
    return  render(request,'index.html',context)