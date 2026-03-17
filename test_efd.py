from eliptic_fourier_descriptors.efd import process_image

process_image("test_images/Gemini_Generated_Image_9a2yya9a2yya9a2y.png", efd_orders=(5, 10,20, 40, 80), min_contour_area=500)

