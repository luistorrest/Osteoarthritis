import cv2
import os
paths = {
    # '2clases':{
    #     'source':r"C:/Users/grupo_gepar/Documents/lucho/Osteo/Dataset",
    #     'output':r"C:/Users/grupo_gepar/Documents/lucho/Osteo/DatasetPrep",
    # },
    # '5clases':{
    #     'source':r"C:/Users/grupo_gepar/Documents/lucho/Osteo/archive",
    #     'output':r"C:/Users/grupo_gepar/Documents/lucho/Osteo/archivePrep",
    # },
    # '5clasesSAM':{
    #     'source':r"C:/Users/grupo_gepar/Documents/lucho/Osteo/archive",
    #     'output':r"C:/Users/grupo_gepar/Documents/lucho/Osteo/archiveSAMPrep",
    # },
    'kaggle-prep':{
        'source':r"C:\Users\grupo_gepar\Documents\lucho\Osteo\Osteoarthritis_Assignment_Merged",
        'output':r"C:\Users\grupo_gepar\Documents\lucho\Osteo\Osteoarthritis_Assignment_Merged_Prep",
    },

}

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
model_path=r"C:\Users\grupo_gepar\Downloads\sam_vit_h_4b8939.pth"
model_type="vit_h"
#path_image=r"C:\Users\grupo_gepar\Documents\lucho\Osteo\archivePrep\test\0\9392060L.png"
#path_image=r"C:\Users\grupo_gepar\Documents\lucho\Osteo\DatasetPrep\train\train\Osteoarthritis\9645958L.png"
sam = sam_model_registry[model_type](checkpoint=model_path)
mask_generator = SamAutomaticMaskGenerator(sam)
import numpy as np
show=False
i=0
import glob
import os
for dataset in paths.keys():
    source_path = paths[dataset]['source']
    output_path = paths[dataset]['output']
    a=os.makedirs(output_path,exist_ok=True)


    image_list = glob.glob(os.path.join(source_path,"**","*.png"),recursive=True)
    output_list = [x.replace(source_path,output_path) for x in image_list]

    for path,path_out in zip(image_list,output_list):

        image = cv2.imread(path, cv2.IMREAD_COLOR)

        final_image=image
        # Check the number of channels in the image
        # if len(image.shape) == 2:
        #     # The image is grayscale
        #     print('The image has 1 channel.')
        # else:
        #     # The image is color
        #     print('The image has 3 channels.')

        # Perform histogram equalization
        if len(image.shape) == 2:
            # The image is grayscale, so perform histogram equalization directly
            equalized_image = cv2.equalizeHist(image)
        else:
            # The image is color, so convert it to grayscale first
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)


        # Add Gaussian blur
        kernel_size = (7, 7)  # The size of the Gaussian kernel
        blurred_image = cv2.GaussianBlur(equalized_image, kernel_size, 0)

        # # Apply Gaussian blur to the grayscale image
        # blurred = cv2.GaussianBlur(equalized_image, (5, 5), 0)

        # Apply edge detection using Canny
        #edge_image = cv2.Canny(blurred_image 50, 150)

        # # (Optional) Apply dilation and erosion to the edge map
        # dilated = cv2.dilate(edges, None, iterations=2)
        # eroded = cv2.erode(dilated, None, iterations=1)
        # # # Display the original and equalized images


        blurred_image =cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
        # SAM
        if False:
            masks = mask_generator.generate(blurred_image)

            final_image=blurred_image

            hueso_masks = []
            for i, (mask) in enumerate(zip(masks)):
                mask_array=mask[0]["segmentation"]
                
                mask_array.shape
                middle_x=mask_array.shape[0]//2
                middle_y=mask_array.shape[1]//2

                hueso_up=np.sum(mask_array[:middle_y,middle_x])/mask_array[:middle_y,middle_x].shape
                hueso_down=np.sum(mask_array[middle_y:,middle_x])/mask_array[middle_y:,middle_x].shape

                hueso= hueso_up > 0.5 or hueso_down > 0.5
                if hueso:
                    hueso_masks.append(mask_array)


            if len(hueso_masks) != 2:
                pass
                print("mierda")
                # TODO: Further logic to distinguish huesos
                os.makedirs(os.path.dirname(path_out),exist_ok=True)
                ext=os.path.splitext(path_out)[-1]

                cv2.imwrite(path_out.replace(ext,"err."+ext), final_image)

                continue

            full_mask = np.logical_or(hueso_masks[0],hueso_masks[1])

            equalized_image[full_mask] = 255
            equalized_image[np.logical_not(full_mask)]=0

        final_image = blurred_image
        os.makedirs(os.path.dirname(path_out),exist_ok=True)
        cv2.imwrite(path_out, final_image)

        if show:
            cv2.imshow('Original Image', image)
            cv2.imshow('Preprocessed Image', final_image)
            cv2.waitKey()
        print(i)
        i+=1
        # # Wait for a key press and close the windows
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()