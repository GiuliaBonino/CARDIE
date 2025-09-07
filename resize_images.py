from PIL import Image


path = "output_collage_HDRPLUS.jpg"
# Load the original image
image = Image.open(path)

# Get original dimensions
width, height = image.size

# Calculate new dimensions (half the size)
new_size = (width // 2, height // 2)

# Resize the image
smaller_image = image.resize(new_size, Image.Resampling.LANCZOS)


print(path.split(".")[0] + "_resized.png")

smaller_image.save(path.split(".")[0] + "_resized.png")
