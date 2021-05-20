import os
import glob
from PIL import Image

files = glob.glob('../Images/onion/*.jpg')

i = 0
for f in files:
    img = Image.open(f)
    img_resize = img.resize((64, 64))
    title, ext = os.path.splitext(f)
    #print(title, ext)
    #print(f)
    #i = i + 1
    #new_path = './64tomato' + '/tomato' + str(i) + '.jpg'
    print(title, ext)
    img_resize.save(title + ext)