import os
from pathlib import Path
from shutil import copyfile

#path to files
#local_path = Path("C:/Users/Robin.De.Bie/Documents/Scripts/foto")
local_path = Path("D:/home/foto")
#backup_path = Path("C:/Users/Robin.De.Bie/Documents/Scripts/backup/foto")
backup_path = Path("D:/home/backup/foto")

#Foto opslaan in backup directory
for i in os.listdir(local_path):
    new_backup_path = backup_path / i
    for subdir, dirs, files in os.walk(local_path / i):
        for file in files:
            foto = os.path.join(subdir, file)
            copyfile(foto, new_backup_path / file)