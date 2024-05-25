import cv2

path=r"C:\Users\grupo_gepar\Documents\lucho\Osteo\archivePrep\merged\1\9013798R.png"
#r"C:\Users\grupo_gepar\Documents\lucho\Osteo\Osteoarthritis_Assignment_Merged\Normal\9003126R.png"
# Load the image using OpenCV's imread function
image = cv2.imread(path)

# Print the shape of the image using NumPy's shape attribute
print(image.shape)
