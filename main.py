#CNN

from flask import Flask, render_template, request
from io import BytesIO
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from base64 import b64encode
from skimage.transform import resize

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

# code which helps initialize our server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret key'

bootstrap = Bootstrap(app)

#saved_model = load_model("models/model1.h5")
saved_model = load_model("model1.h5")
saved_model._make_predict_function()

class UploadForm(FlaskForm):
    photo = FileField('Upload an image',validators=[FileAllowed(['jpg', 'png', 'jpeg'], u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Predict')


port = int(os.environ.get("PORT", 5000))
@app.route('/', methods=['GET','POST'])
def predict():
    form = UploadForm()
    if form.validate_on_submit():
   	 print(form.photo.data)
   	 image_stream = form.photo.data.stream
   	 original_img = Image.open(image_stream)
	 #convert pil.jpegimageplugin.jpegimagefile to numpy array
   	 img = np.array(original_img)
   	 #img_my2 = Image.fromarray(img, 'RGB')
   	 #img_my2.save('my2.png')
   	 #img_my2.show()

   	 print('sasadasfasdf',type(img))
   	 print('sasadasfasdf',img.shape)
   	 img = resize(img, (96, 96, 3), anti_aliasing=True)
   	 img = img
   	 print('sasadasfasdf',img.shape)

   	 img = np.expand_dims(img, axis=0)
   	 prediction = saved_model.predict(img)
   	 print('****************************************************************************')
   	 print(type(img))
   	 print(img.shape)
   	 print(img[0].shape)
   	 print('valor para comparar:',np.amax(img))
   	 print(type(prediction))
   	 print(prediction.shape)
   	 print(prediction[0])
   	 print(prediction[0][0])
   	 print(prediction[0][1])
   	 print('****************************************************************************')

   	 img2 = img[0]
   	 img3 = Image.fromarray(img2, 'RGB')
   	 img3.save('my.png')


   	 if (prediction[0][0] < prediction[0][1]):
   		 result = "SHIP"
   		 prediccion_0 = prediction[0][0]
   		 prediccion_1 = prediction[0][1]
   	 else:
   		 result = "NO SHIP"

   	 byteIO = BytesIO()
   	 original_img.save(byteIO, format=original_img.format)
   	 byteArr = byteIO.getvalue()
   	 encoded = b64encode(byteArr)

   	 return render_template('result.html', result=result, encoded_photo=encoded.decode('ascii'))
    return render_template('index.html', form=form)

if __name__ == '__main__':
#    app.run(debug=True)
    app.run(debug=True,host='0.0.0.0',port=port)