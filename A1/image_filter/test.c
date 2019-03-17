#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "omp.h"
#include <stdio.h> 
#include <string.h>
#include <stdlib.h>
#include <math.h>

int changeA(int a) {
    return a + 1;
}

int find(int answer[5], int found) {
    int row, col;
    
        printf("%d\n", found);
        int num;
        int a = 5; 
        #pragma omp task shared(a, found) firstprivate(num,row) if(found == 0)
        {
            printf("%d\n", found);
            a = changeA(a); 
            for (row = 0; row < 5; row++) {                
            #pragma omp critical
            found = 1;
            if(answer[row] == 1)
                find(answer, found);
            } 
          
        }
                  printf("Main %d \n", a);
         
        
    
    return 0; 
} 


int main() { 
    //#pragma omp parallel
    {
        double time1 = omp_get_wtime();
        int height = 3;
        int width = 2;
        double sigma = 3;
        double sum=0.0;

        int found = 0 ;
        int matrix[] = {0,0,0,1,0};
        #pragma omp parallel shared(found)
        { 
        #pragma omp single
        {
            printf("answer %d\n", find(matrix, found));
        }  
        }
        printf("Elapsed time: %0.20lf\n", omp_get_wtime() - time1); 
 
    }
}

