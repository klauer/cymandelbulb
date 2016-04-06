from cymandelbulb import mandelbrot_image
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

print('calling image...')
im = mandelbrot_image(0, image_width=1000, image_height=1000,
                      scale=(4., 4., 50.0), n=8)

print('image is', im, im.shape)
plt.imshow(im / 255.)
plt.show()
