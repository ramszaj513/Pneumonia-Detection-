import dropbox

MODEL_FILE = "pneumonia_detection_modelv2.h5"

# Replace these with your app's credentials
ACCESS_TOKEN = ''

# Initialize Dropbox client
dbx = dropbox.Dropbox(ACCESS_TOKEN)

# Specify the path to the file you want to download from Dropbox
dropbox_file_path = '/path/to/your/file.txt'

# Specify the local path where you want to save the downloaded file
local_file_path = MODEL_FILE

# Download the file from Dropbox
with open(local_file_path, "wb") as f:
    metadata, res = dbx.files_download(dropbox_file_path)
    f.write(res.content)
    
https://www.dropbox.com/scl/fi/0uyffcmomojlpjovyzz3z/pneumonia_detection_modelv2.h5?rlkey=lsxi1pd9qqma9aavr8xj5asd0&st=45pl4d7z&dl=0
