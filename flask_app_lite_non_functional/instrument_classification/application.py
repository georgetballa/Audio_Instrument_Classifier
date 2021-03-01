from flask import Flask, render_template, url_for, request
import numpy as np
from sklearn.datasets import load_iris
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import io
import librosa
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class_dict = {'bass': 0,
 'brass': 1,
 'flute': 2,
 'guitar': 3,
 'keyboard': 4,
 'mallet': 5,
 'organ': 6,
 'reed': 7,
 'string': 8,
 'synth_lead': 9,
 'vocal': 10}


application = app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home(title=None):
    title="Home"
    return render_template("home.html", title=title)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/eda')
def eda():
    df = pd.read_csv('./validation_prediction.csv')
    filename_list = list(df.filename)
    return render_template("eda.html", filename_list=filename_list)

@app.route('/graphs', methods=['POST'])
def graphs():
    filename = str(request.form['files'])
    filename2 = str(request.form['files2'])
    
    actual = filename.split('/')
    actual = actual[0]

    # if col1 == col2:
    #     return f'Why do you want to graph column {col1+1} by itself?!'
    # else:
    #     fig = Figure()
    #     ax = fig.subplots()
    #     ax.scatter([random.random() for i in range(100)], [random.random() for i in range(100)])
    #     ax.set_title('A Very Random Scatterplot')
    #     pngImage = BytesIO()
    #     FigureCanvas(fig).print_png(pngImage)
    #     pngImageB64String = "data:image/png;base64,"
    #     pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    #     image = pngImageB64String
    return render_template('graphs.html', 
    url=f'./static/val_set/{filename}', 
    wavurl = f'./static/val/{filename[:-3].replace(".", "")}.wav', 
    filename=filename,
    url2=f'./static/val_set/{filename2}',
    wavurl2 = f'./static/val/{filename2[:-3].replace(".", "")}.wav',
    filename2=filename2)

@app.route('/predict')
def predict():
    df = pd.read_csv('./validation_prediction.csv')
    filename_list = list(df.filename)
    return render_template('predict.html', filename_list=filename_list)

@app.route('/contact')
def contact():
    
    return render_template('contact.html')

@app.route('/upload')
def upload():
    
    return render_template('upload.html')

@app.route('/results', methods=['POST'])

def results():
    model = load_model('./instrument_classifier_model')
    
    filename_img = str(request.form['predfiles'])
    filename_img = filename_img.split('/')
    filename_img = filename_img[-2:]
    filename_img = '/'.join(filename_img)
    
        
    filename_wav = f'./static/val/{filename_img[:-3].replace(".", "")}.wav'
    
    
    img = load_img(f'./static/val_set/{filename_img}',target_size = (262,694))
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)
    prediction = model.predict(img)
    
    pred = np.argmax(prediction)
    
    key_list = list(class_dict.keys())
    val_list = list(class_dict.values())
    
    position = val_list.index(pred)
    predicted_name = key_list[position]
    actual = filename_img.split('/')
    actual = actual[0]

    
    


    fig = Figure()
    
    axis = fig.add_subplot(1, 1, 1,)
    fig.set_figheight(14)
    fig.set_figwidth(14)
    axis.bar(key_list,prediction.reshape(-1,))
    
    fig.suptitle('Model Prediction Probability', fontsize = 30)
    plt.setp(axis.xaxis.get_majorticklabels(), rotation=45, fontsize = 20)
    plt.setp(axis.yaxis.get_majorticklabels(),  fontsize = 20)
    plt.tight_layout()
    
    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(output.getvalue()).decode('utf8')

    

    

    return render_template('results.html', prediction=predicted_name, filename_wav=filename_wav, actual=actual, pimage=f'./static/images/{predicted_name}.png',
    aimage=f'./static/images/{actual}.png', resp=pngImageB64String)

@app.route('/results2', methods=['POST'])
def results2():

    model = load_model('./instrument_classifier_model')
    
    
    wav = request.form['myFile']
    y, sr = librosa.load(wav, mono=True, duration=3)

    # filename_img = filename_img.split('/')
    # filename_img = filename_img[-2:]
    # filename_img = '/'.join(filename_img)
    
        
    # filename_wav = f'./static/val/{filename_img[:-3].replace(".", "")}.wav'
    
    
    # img = load_img(f'./static/val_set/{filename_img}',target_size = (262,694))
    # img = img_to_array(img)
    # img = img.reshape((1,) + img.shape)
    # prediction = model.predict(img)
    
    # pred = np.argmax(prediction)
    
    # key_list = list(class_dict.keys())
    # val_list = list(class_dict.values())
    
    # position = val_list.index(pred)
    # predicted_name = key_list[position]
    # actual = filename_img.split('/')
    # actual = actual[0]

    
    


    # fig = Figure()
    
    # axis = fig.add_subplot(1, 1, 1,)
    # fig.set_figheight(14)
    # fig.set_figwidth(14)
    # axis.bar(key_list,prediction.reshape(-1,))
    
    # fig.suptitle('Model Prediction Probability', fontsize = 30)
    # plt.setp(axis.xaxis.get_majorticklabels(), rotation=45, fontsize = 20)
    # plt.setp(axis.yaxis.get_majorticklabels(),  fontsize = 20)
    # plt.tight_layout()
    
    # output = BytesIO()
    # FigureCanvas(fig).print_png(output)
    # pngImageB64String = "data:image/png;base64,"
    # pngImageB64String += base64.b64encode(output.getvalue()).decode('utf8')

    

    

    # return render_template('results2.html', prediction=predicted_name, filename_wav=filename_wav, actual=actual, pimage=f'./static/images/{predicted_name}.png',
    # aimage=f'./static/images/{actual}.png', resp=pngImageB64String)
    return render_template('results2.html', wav=wav)

if __name__=="__main__":
    # model = load_model('./instrument_classifier_model')
    app.run(debug=False)