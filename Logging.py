import os, shutil
from datetime import datetime
from time import time, strftime

class Logging():
    def __init__(self, log_path):
        self.filename = log_path
        
    def record(self, str_log):
        now = datetime.now()
        filename = self.filename
        print(str_log)
        with open(filename, 'a') as f:
            f.write("%s %s\r\n" % (now.strftime('%Y-%m-%d-%H:%M:%S'), str_log))
            f.flush()

def now():
    return str(strftime('%Y-%m-%d %H:%M:%S'))

def check_dir(file_path):
    import os
    save_path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()
