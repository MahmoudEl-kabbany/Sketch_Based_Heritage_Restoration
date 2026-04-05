from eliptic_fourier_descriptors.efd import process_image

IMAGE_PATH = "test_images/bolt.png"
EFD_ORDERS = (5, 10, 20, 40)
MIN_CONTOUR_AREA = 500
USE_SKELETON = False  # Set True to skeletonize before contour extraction.
CONTOUR_RETRIEVAL = "external"  # "tree" | "external"


process_image(
	IMAGE_PATH,
	order=max(EFD_ORDERS),
	min_contour_area=MIN_CONTOUR_AREA,
	use_skeleton=USE_SKELETON,
	contour_retrieval=CONTOUR_RETRIEVAL,
)

