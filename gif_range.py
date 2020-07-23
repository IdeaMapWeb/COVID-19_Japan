from PIL import Image, ImageDraw

s=30

images = []
path = './fig/rc=400p=0.5mr=10/'
for i in range(0,s,1):
    st_name = 'fig'+str(i) + '_'
    im = Image.open(path+'{}.png'.format(st_name)) 
    im =im.resize(size=(1200, 600), resample=Image.NEAREST)
    images.append(im)
    
images[0].save(path+'rc=400p=0.5mr=10_600_1.gif', save_all=True, append_images=images[1:s], duration=100*10, loop=0)
