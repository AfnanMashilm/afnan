#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <png.h>

#define MAX_ITER 2000
#define X_MIN -2
#define X_MAX 2
#define Y_MIN -2
#define Y_MAX 2
#define WID 800
#define HEI 800

typedef struct {
    double real;
    double imag;
} complex;

int mandelbrot(complex c) {
    complex z = {0, 0};
    int i;
    for (i = 0; i < MAX_ITER; i++) {
        double r = z.real;
        double im = z.imag;
        z.real = r * r - im * im + c.real;
        z.imag = 2 * r * im + c.imag;
        if (z.real * z.real + z.imag * z.imag > 4) {
            return i;
        }
    }
    return MAX_ITER;
}

void write_png_file(char* file_name, int* buffer) {
    FILE *fp = fopen(file_name, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, WID, HEI, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    png_bytep row = (png_bytep) malloc(3 * WID * sizeof(png_byte));
    int y, x;
    for (y = 0; y < HEI; y++) {
        for (x = 0; x < WID; x++) {
            int color = buffer[y * WID + x];
            row[x * 3] = color;
            row[x * 3 + 1] = color;
            row[x * 3 + 2] = color;
        }
        png_write_row(png_ptr, row);
    }
    png_write_end(png_ptr, NULL);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    fclose(fp);
    free(row);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = HEI / size;
    int start_row = rank * chunk_size;
    int end_row = (rank + 1) * chunk_size;

    if (rank == size - 1) {
        end_row = HEI;
    }

    int buffer[WID * chunk_size];
    int row, col;
    for (row = start_row; row < end_row; row++) {
        for (col = 0; col < WID; col++) {
            double x = X_MIN + (col * (X_MAX - X_MIN)) / (double) WID;
            double y = Y_MIN + (row * (Y_MAX - Y_MIN)) / (double) HEI;
            complex c = {x, y};
            int iteration = mandelbrot(c);
            int color = (iteration == MAX_ITER) ? 0 : 255 * sqrt((double) iteration / MAX_ITER);
        buffer[(row - start_row) * WID + col] = color;
    }
}

if (rank == 0) {
    int* global_buffer = (int*) malloc(WID * HEI * sizeof(int));
    MPI_Gather(buffer, WID * chunk_size, MPI_INT, global_buffer, WID * chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    write_png_file("image.png", global_buffer);
    free(global_buffer);
} else {
    MPI_Gather(buffer, WID * chunk_size, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
}

MPI_Finalize();
return 0;

}
