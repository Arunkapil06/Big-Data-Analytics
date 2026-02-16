from PIL import Image, ImageDraw

# Create a 32x32 black image with a white 'D' for 'Data'
img = Image.new('RGB', (32, 32), color='black')
d = ImageDraw.Draw(img)
d.text((8, 4), "D", fill='white', size=24)

# Save as ICO file
img.save('static/favicon.ico', format='ICO') 