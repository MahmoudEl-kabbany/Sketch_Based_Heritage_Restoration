from eliptic_fourier_descriptors.efd import process_image

IMAGE_PATH = "test_images/Gemini_Generated_Image_9a2yya9a2yya9a2y.png"
EFD_ORDERS = (5, 10, 20, 40, 80)
MIN_CONTOUR_AREA = 500
USE_SKELETON = False  # Set True to skeletonize before contour extraction.


process_image(
	IMAGE_PATH,
	efd_orders=EFD_ORDERS,
	min_contour_area=MIN_CONTOUR_AREA,
	use_skeleton=USE_SKELETON,
)

