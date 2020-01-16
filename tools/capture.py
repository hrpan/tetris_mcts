from PIL import Image, ImageDraw
import numpy as np
import time

scale = 20

size = (11 * scale, 23 * scale)

img = Image.new('L', size)

draw = ImageDraw.Draw(img)

f = open('./board_output', 'rb')

count = 0
while True:

    s = f.read()

    if len(s) == 220:
        data = np.fromstring(s, dtype=np.int8).reshape(22, 10)

    f.seek(0)

    draw.rectangle([scale // 2, scale // 2, size[0] - scale // 2, size[1] - scale // 2], fill=255)

    for i in range(22):
        for j in range(10):
            x0 = scale // 2 + j * scale
            x1 = x0 + scale
            y0 = scale // 2 + i * scale
            y1 = y0 + scale

            if data[i][j] == 1:
                c = 255
            elif data[i][j] == 0:
                c = 0
            else:
                c = 122
            draw.rectangle([x0, y0, x1, y1], fill=c)
    for i in range(21):
        x0 = scale // 2
        x1 = x0 + 10 * scale
        y0 = scale // 2 + (i + 1) * scale
        y1 = scale // 2 + (i + 1) * scale
        draw.line([x0, y0, x1, y1], fill=0)
    for i in range(9):
        x0 = scale // 2 + (i + 1) * scale
        x1 = scale // 2 + (i + 1) * scale
        y0 = scale // 2
        y1 = scale // 2 + 22 * scale
        draw.line([x0, y0, x1, y1], fill=0)


    img.save('./pngs/{}.png'.format(count), compression_level=9)
    count += 1
    time.sleep(0.5)
