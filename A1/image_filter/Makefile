icc: image_blurring.c icc image_blurring_parallel.c stb_image.h stb_image_write.h
	icc image_blurring.c -o image_blurring -qopenmp -lm
	icc image_blurring_parallel.c -o image_blurring_parallel -qopenmp -lm

gcc: image_blurring.c stb_image.h stb_image_write.h
	gcc -o image_blurring image_blurring.c -fopenmp -lm
