import datetime
import shutil
import copy
import os

def create_data_dir(data_dir,file_path,config_path,file_name,comment=None):
  dir=copy.copy(data_dir)
  if comment==None:
    now = datetime.datetime.now()
    data_dir=data_dir+"/exp_data/"+file_name+"/"+now.strftime("%Y%m%d_%H%M%S")
  else:
    data_dir=data_dir+"/exp_data/"+file_name+"/"+comment  
  os.makedirs(data_dir)
  os.chdir(data_dir)
  shutil.copytree(dir+"/creature",data_dir+"/creature")
  shutil.copy(config_path,data_dir+"/creature")
  shutil.copy(file_path,data_dir+"/evaluation.py")
  shutil.move(data_dir+"/creature/replay.py",data_dir)

  return data_dir

  