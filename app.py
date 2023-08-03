from flask import *
from flask_api import status
from PIL import Image
import os
import shutil
import zipfile
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)  

### swagger specific ###
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Watermark-Python-Flask-REST-API"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
### end swagger specific ###
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    
    return ('.' in filename and filename.split(
        '.')[-1].lower() in ALLOWED_EXTENSIONS)

@app.route('/water', methods = ['POST'])  
def upload():  
    obj = {"status": "KO", "host": request.remote_addr}

    if 'files[]' not in request.files:
        obj['message'] = "No file present"
        return make_response(jsonify(obj), status.HTTP_400_BAD_REQUEST)
    
    ONLY_BG_REMOVAL = request.form.get('ONLY_BG_REMOVAL')
    ONLY_BG_REMOVAL = ONLY_BG_REMOVAL.capitalize()

    try:
        path = 'Final_test_images/test_images/'
        for file_name in os.listdir(path):
            # construct full file path
            file = path + file_name
            if os.path.isfile(file):
                print('Deleting file:', file)
                os.remove(file)

        dir_path = 'Final_test_images/final_output/'
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
            print("Deleted '%s' directory successfully" % dir_path)

        dir_path2 = 'Final_test_images/pass_to_segment/'
        if os.path.exists(dir_path2):
            shutil.rmtree(dir_path2, ignore_errors=True)
            print("Deleted '%s' directory successfully" % dir_path2)
    except:
        print("except block")
    
    files = request.files.getlist('files[]')
    for file in files:
        if file and allowed_file(file.filename):
            # filename = str(uuid.uuid4()) + secure_filename(file.filename)
            # file_names.append(filename)
            compressed_image = Image.open(file)
            compressed_image = compressed_image.convert('RGB')
            f = 'Final_test_images/test_images/'+file.filename
            #f = 'Final_test_images/new/'+file.filename
            compressed_image.save(f,optimizer=True,quality=90)
        
        else:
            obj['message'] = "file type not allowed"
            return make_response(jsonify(obj), status.HTTP_400_BAD_REQUEST)
    if ONLY_BG_REMOVAL == None:
        ONLY_BG_REMOVAL = False
    os.system("/bin/bash INFERENCE.sh {}".format(ONLY_BG_REMOVAL))

    for root,dirs, files in os.walk('Final_test_images/final_output/'):
        for file in files:
            filename = 'Final_test_images/final_output/'+file
    return send_file(filename, mimetype='image/gif')

    # zipf = zipfile.ZipFile('Name.zip','w', zipfile.ZIP_DEFLATED)
    # for root,dirs, files in os.walk('Final_test_images/final_output/'):
    #     for file in files:
    #         zipf.write('Final_test_images/final_output/'+file)
    # zipf.close()
    # return send_file('Name.zip',
    #         mimetype = 'zip',
    #         attachment_filename= 'Name.zip',
    #         as_attachment = True)

    #return make_response(jsonify(obj), status.HTTP_200_OK) 

@app.route('/download_all')
def download_all():
    zipf = zipfile.ZipFile('Name.zip','w', zipfile.ZIP_DEFLATED)
    for root,dirs, files in os.walk('Final_test_images/final_output/'):
        for file in files:
            zipf.write('Final_test_images/final_output/'+file)
    zipf.close()
    return send_file('Name.zip',
            mimetype = 'zip',
            attachment_filename= 'Name.zip',
            as_attachment = True)

@app.route('/hello')
def hello():

    obj = {"status": "KO", "host": request.remote_addr}
    return make_response(jsonify(obj), status.HTTP_200_OK)

if __name__ == '__main__':  
    app.run(debug = True, port=6000)  