from bezier_curves.bezier import fit_from_image_skeleton, fit_from_image
paths, adj = fit_from_image_skeleton('test_images/damaged_oval.png')
print('damaged_oval skeleton paths:', len(paths))

paths2 = fit_from_image('test_images/damaged_oval.png')
print('damaged_oval contour paths:', len(paths2))

paths3, adj3 = fit_from_image_skeleton('test_images/restoration_test.png')
print('restoration_test skeleton paths:', len(paths3))
