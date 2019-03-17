#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "omp.h"

double ** getGaussian(int width, int height, double sigma)
{
    double sum=0.0;
    double ** kernel = (double **) malloc (height * sizeof(double *));
    int i,j;

    for(i = 0; i < height; i++) {
	kernel[i] = (double *) malloc (width * sizeof(double));
    }
    for (i=0 ; i<height ; i++) {
        for (j=0 ; j<width ; j++) {
            kernel[i][j] = exp(-(i*i+j*j)/(2*sigma*sigma))/(2*M_PI*sigma*sigma);
            sum += kernel[i][j];
        }
    }

    for (i=0 ; i<height ; i++) {
        for (j=0 ; j<width ; j++) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

double *** loadImage(const char *filename, int * width, int * height)
{
    int bpp;
    uint8_t* rgb_image = stbi_load(filename, width, height, &bpp, 3);
    double *** imageMatrix = (double ***) malloc (3 * sizeof(double **));

    int h,w;
    imageMatrix[0] = (double **) malloc (*height * sizeof(double *));
    imageMatrix[1] = (double **) malloc (*height * sizeof(double *));
    imageMatrix[2] = (double **) malloc (*height * sizeof(double *));
    for (h=0 ; h < *height ; h++) {
	imageMatrix[0][h] = (double *) malloc ((*width * 3) * sizeof(double)); 
	imageMatrix[1][h] = (double *) malloc ((*width * 3) * sizeof(double));
	imageMatrix[2][h] = (double *) malloc ((*width * 3) * sizeof(double));
    }

    for (h=0 ; h < *height ; h++) {
        for (w=0 ; w < *width ; w++) {
            imageMatrix[0][h][w] = rgb_image[h*(*width * 3)+w*3];
            imageMatrix[1][h][w] = rgb_image[h*(*width * 3)+w*3+1];
            imageMatrix[2][h][w] = rgb_image[h*(*width * 3)+w*3+2];
        }
    }
    return imageMatrix;
}

void saveImage(double *** imageMatrix, const char *filename, int width, int height)
{
    uint8_t* rgb_image;
    rgb_image = malloc(width*height*3);
   
    int h,w;
    for (h=0 ; h < height ; h++) {
        for (w=0 ; w < width ; w++) {
	    rgb_image[h*(width*3)+w*3] = imageMatrix[0][h][w];
            rgb_image[h*(width*3)+w*3+1] = imageMatrix[1][h][w];
            rgb_image[h*(width*3)+w*3+2] = imageMatrix[2][h][w];
        }
    }
    stbi_write_png(filename, width, height, 3, rgb_image, width*3);
    stbi_image_free(rgb_image);
}

double *** applyFilter(double *** image, double ** filter, int width, int height, int filterWidth, int filterHeight){
    int newImageHeight = height-filterHeight+1;
    int newImageWidth = width-filterWidth+1;
    int d,i,j,h,w;

    double *** newImage = (double ***) malloc (3 * sizeof(double **));

    newImage[0] = (double **) malloc (height * sizeof(double *));
    newImage[1] = (double **) malloc (height * sizeof(double *));
    newImage[2] = (double **) malloc (height * sizeof(double *));
    for (h=0 ; h < height ; h++) {
	newImage[0][h] = (double *) malloc ((width * 3) * sizeof(double)); 
	newImage[1][h] = (double *) malloc ((width * 3) * sizeof(double));
	newImage[2][h] = (double *) malloc ((width * 3) * sizeof(double));
    }
    
    for (d=0 ; d<3 ; d++) {
        for (i=0 ; i<newImageHeight ; i++) {
            for (j=0 ; j<newImageWidth ; j++) {
                for (h=i ; h<i+filterHeight ; h++) {
                    for (w=j ; w<j+filterWidth ; w++) {
                        newImage[d][i][j] += filter[h-i][w-j]*image[d][h][w];
                    }
                }
            }
        }
    }

    return newImage;
}

void averageRGB(double *** image, int width, int height) {
	double sum[3] = { 0.0 };
	int i, j, k;
	for (i=0 ; i<3 ; i++) {
        	for (j=0 ; j<height ; j++) {
            		for (k=0 ; k<width ; k++) {
                        	sum[i] += image[i][j][k];
                	}
            	}
    	}
	int size = width * height;
	printf("R: %0.6lf, G: %0.6lf, B: %0.6lf\n", sum[0]/size, sum[1]/size, sum[2]/size);
}

int main(int argc, char const *argv[]) {
    int width, height, filter_width = 21, filter_height = 21;
    if (argc < 2){
	    printf("Please enter the name of the PNG file.\n");
	    exit(0);
    }
    double time1 = omp_get_wtime();
    double *** input_image = loadImage(argv[1], &width, &height);

    double ** filter = getGaussian(filter_width, filter_height, 10.0);
 
    double *** output_image = applyFilter(input_image, filter, width, height, filter_width, filter_height); 
    char output_filename[50];
    sprintf(output_filename, "blurred_%s", argv[1]);
    saveImage(output_image, output_filename, width, height);
    printf("Elapsed time: %0.2lf\n", omp_get_wtime() - time1);

    //for debugging purpose
    averageRGB(output_image, width, height);
    return 0;
}
