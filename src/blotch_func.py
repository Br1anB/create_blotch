from PIL import Image
import PIL.ImageDraw as ImageDraw
import numpy as np 
import random 

def generateBlotch(image, maxNumBlotches, blotchProb, maxBlotchSize):
    padded_image = np.pad(image, [(maxBlotchSize//2, maxBlotchSize//2), (maxBlotchSize//2, maxBlotchSize//2), (0, 0)])
    padded_shape = padded_image.shape
    canvas = np.ones_like(padded_image, dtype=np.uint8)*255
    canvas = canvas.astype(np.uint8)
    canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas)
    shapes = ['arc', 'ellipse', 'chord', 'circle', 'pieslice', 'polygon', 'rectangle', 'rounded_rectangle']
    for _ in range(maxNumBlotches):
        _blotchPresent = random.random()
        if _blotchPresent < blotchProb:
            _shape = random.choice(shapes)
            if _shape == 'arc':
                x, y = random.randint(0, padded_shape[0]-maxBlotchSize//2), random.randint(0, padded_shape[1]-maxBlotchSize//2)
                x_end, y_end = random.randint(x, x+maxBlotchSize-1), random.randint(y, y+(maxBlotchSize)-1)
                start = random.randint(0, 360)
                end = random.randint(0, 360)
                width = maxBlotchSize//4
                draw.arc((x, y, x_end, y_end), start, end, fill='black', width=width)
            if _shape == 'ellipse':
                x, y = random.randint(0, padded_shape[0]-maxBlotchSize//2), random.randint(0, padded_shape[1]-maxBlotchSize//2)
                x_end, y_end = random.randint(x, x+maxBlotchSize-1), random.randint(y, y+(maxBlotchSize)-1)
                draw.ellipse((x, y, x_end, y_end), fill='black', outline='black')
            if _shape == 'chord':
                x, y = random.randint(0, padded_shape[0]-maxBlotchSize//2), random.randint(0, padded_shape[1]-maxBlotchSize//2)
                x_end, y_end = random.randint(x, x+maxBlotchSize-1), random.randint(y, y+(maxBlotchSize)-1)
                start = random.randint(0, 360)
                end = random.randint(0, 360)
                width = maxBlotchSize//4
                draw.chord((x, y, x_end, y_end), start, end, fill='black', width=width)
            if _shape == 'circle':
                x, y = random.randint(maxBlotchSize//2, padded_shape[0]-maxBlotchSize//2), random.randint(maxBlotchSize//2, padded_shape[1]-maxBlotchSize//2)
                radius = random.randint(1, maxBlotchSize//2)
                draw.circle((x, y), radius, 'black', 'black', width=1)
            if _shape == 'pieslice':
                x, y = random.randint(0, padded_shape[0]-maxBlotchSize//2), random.randint(0, padded_shape[1]-maxBlotchSize//2)
                x_end, y_end = random.randint(x, x+maxBlotchSize-1), random.randint(y, y+(maxBlotchSize)-1)
                start = random.randint(0, 360)
                end = random.randint(0, 360)
                width = maxBlotchSize//4
                draw.pieslice((x, y, x_end, y_end), start, end, fill='black', width=width)
            if _shape == 'polygon':
                x, y = random.randint(maxBlotchSize//2, padded_shape[0]-maxBlotchSize//2), random.randint(maxBlotchSize//2, padded_shape[1]-maxBlotchSize//2)
                radius = random.randint(1, maxBlotchSize//2)
                n_sides = random.randint(3, 10)
                rotation = random.randint(0, 360)
                draw.regular_polygon((x, y, radius), n_sides, rotation, 'black', 'black', 1)
            if _shape == 'rectangle':
                x, y = random.randint(0, padded_shape[0]-maxBlotchSize//2), random.randint(0, padded_shape[1]-maxBlotchSize//2)
                x_end, y_end = random.randint(x, x+maxBlotchSize-1), random.randint(y, y+(maxBlotchSize)-1)
                draw.rectangle((x, y, x_end, y_end), 'black', 'black', 1)
            if _shape == 'rounded_rectangle':
                x, y = random.randint(0, padded_shape[0]-maxBlotchSize//2), random.randint(0, padded_shape[1]-maxBlotchSize//2)
                x_end, y_end = random.randint(x, x+maxBlotchSize-1), random.randint(y, y+(maxBlotchSize)-1)
                radius = random.randint(1, maxBlotchSize//2)
                draw.rounded_rectangle((x, y, x_end, y_end), radius, 'black', 'black')
    canvas = np.array(canvas)
    canvas = canvas[maxBlotchSize//2:-maxBlotchSize//2, maxBlotchSize//2:-maxBlotchSize//2]
    canvas = np.where(canvas > 0.5, 1, 0)
    canvas = canvas.astype(np.uint8)
    return canvas