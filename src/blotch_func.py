from PIL import Image
import PIL.ImageDraw as ImageDraw
import numpy as np 
import random 

def generateBlotch_new(image, maxNumBlotches, blotchProb, maxBlotchSize):
    # Pad the image to allow for drawing shapes at the edges
    padded_image = np.pad(image, [(maxBlotchSize//2, maxBlotchSize//2), (maxBlotchSize//2, maxBlotchSize//2), (0, 0)])
    padded_shape = padded_image.shape
    canvas = np.ones_like(padded_image, dtype=np.uint8) * 255  # White background
    canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas)

    shapes = ['arc', 'ellipse', 'chord', 'pieslice', 'polygon', 'rectangle', 'rounded_rectangle']

    for _ in range(maxNumBlotches):
        _blotchPresent = random.random()
        if _blotchPresent < blotchProb:
            _shape = random.choice(shapes)
            x = random.randint(maxBlotchSize//2, padded_shape[1] - maxBlotchSize//2)
            y = random.randint(maxBlotchSize//2, padded_shape[0] - maxBlotchSize//2)

            if _shape == 'arc':
                x_end, y_end = random.randint(x, x + maxBlotchSize - 1), random.randint(y, y + maxBlotchSize - 1)
                start = random.randint(0, 360)
                end = random.randint(0, 360)
                width = maxBlotchSize // 4
                draw.arc((x, y, x_end, y_end), start, end, fill='black', width=width)
            elif _shape == 'ellipse':
                x_end, y_end = random.randint(x, x + maxBlotchSize - 1), random.randint(y, y + maxBlotchSize - 1)
                draw.ellipse((x, y, x_end, y_end), fill='black', outline='black')
            elif _shape == 'chord':
                x_end, y_end = random.randint(x, x + maxBlotchSize - 1), random.randint(y, y + maxBlotchSize - 1)
                start = random.randint(0, 360)
                end = random.randint(0, 360)
                width = maxBlotchSize // 4
                draw.chord((x, y, x_end, y_end), start, end, fill='black', width=width)
            elif _shape == 'pieslice':
                x_end, y_end = random.randint(x, x + maxBlotchSize - 1), random.randint(y, y + maxBlotchSize - 1)
                start = random.randint(0, 360)
                end = random.randint(0, 360)
                width = maxBlotchSize // 4
                draw.pieslice((x, y, x_end, y_end), start, end, fill='black', outline='black')
            elif _shape == 'polygon':
                radius = random.randint(1, maxBlotchSize // 2)
                n_sides = random.randint(3, 10)
                rotation = random.randint(0, 360)
                # Generate polygon vertices
                points = [(x + radius * np.cos(2 * np.pi * i / n_sides + rotation), 
                           y + radius * np.sin(2 * np.pi * i / n_sides + rotation)) for i in range(n_sides)]
                draw.polygon(points, fill='black', outline='black')
            elif _shape == 'rectangle':
                x_end, y_end = random.randint(x, x + maxBlotchSize - 1), random.randint(y, y + maxBlotchSize - 1)
                draw.rectangle((x, y, x_end, y_end), fill='black', outline='black')
            elif _shape == 'rounded_rectangle':
                x_end, y_end = random.randint(x, x + maxBlotchSize - 1), random.randint(y, y + maxBlotchSize - 1)
                radius = random.randint(1, maxBlotchSize // 2)
                draw.rounded_rectangle((x, y, x_end, y_end), radius, fill='black', outline='black')

    # Convert canvas back to a NumPy array
    canvas = np.array(canvas)
    canvas = canvas[maxBlotchSize // 2:-maxBlotchSize // 2, maxBlotchSize // 2:-maxBlotchSize // 2]
    canvas = np.where(canvas > 0.5, 1, 0)
    canvas = canvas.astype(np.uint8)
    
    return canvas
