import shutil
import os



os.mkdir("val") 
os.mkdir("train") 
class_vec = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# move class folder to "train" folder
for base in class_vec:
    shutil.move(base, "train/")

# create class folder in "val"
os.chdir("val/")
for base in class_vec:
     os.mkdir(base) 


# select files to "val" folder
for base in class_vec:
    for i in range(1000):
        try:
            if (i+1) % 5 == 0:
                filename = base + str(i) + ".jpg"
                shutil.move("train/" + base + "/" + filename, "val/" + base + "/")
                print("Moved " + filename)
        except:
            pass
    

