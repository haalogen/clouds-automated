#include "findobject.h"

#include <algorithm>
#include <cmath>
#include <QDebug>


FindObject::FindObject(QImage *image, QImage *object, int chi_cnt) : image(image), object(object), fft_image_output_cached(NULL)
{
    img_width = image->width();
    img_height = image->height();
    img_size = img_width * img_height; // number of pixels on image
    obj_width = object->width();
    obj_height = object->height();

    fft_object_input = fftw_alloc_real(img_size);
    fft_object_output = fftw_alloc_complex(img_size);
    fft_multiplied = fftw_alloc_complex(img_size);
    // r2c -- real to complex
    fft_object_plan = fftw_plan_dft_r2c_2d( img_height, img_width, fft_object_input, fft_object_output, 0);
    ifft_result = fftw_alloc_real(img_size);
    ifft_plan = fftw_plan_dft_c2r_2d( img_height, img_width, fft_multiplied, ifft_result, 0);

    chi_count = chi_cnt;

    qDebug() << "object->rect(): " << object->rect();
}


FindObject::~FindObject()
{
    fftw_free(fft_object_input);
    fftw_free(fft_object_output);
    fftw_free(fft_multiplied);
    fftw_destroy_plan(fft_object_plan);
    fftw_free(ifft_result);
    fftw_destroy_plan(ifft_plan);

}



double *FindObject::calc_convolution(const std::vector<bool> &chi, bool sqrImage) 
{
    fftw_complex *fft_image_output = NULL; 

    // In order not to calculate Fourier for image every time, 
    // we cache it. Now we calculate Fourier for image only once, 
    // and then we can just use it
    if (sqrImage || fft_image_output_cached == NULL) { 
        double *fft_image_input = fftw_alloc_real(img_size); /
        fft_image_output = fftw_alloc_complex(img_size);
        fftw_plan fft_image_plan = fftw_plan_dft_r2c_2d( img_height, img_width, fft_image_input, fft_image_output, FFTW_ESTIMATE);


        std::fill(fft_image_input, fft_image_input+img_size, 0);
        for (int y = 0; y < img_height; y++)
            for (int x = 0; x < img_width; x++) {
                int index = y*img_width + x;
                fft_image_input[index] = qGray(image->pixel(x, y)) / (256.0 / chi_count);
            }

        if (sqrImage) {
            for (int i = 0; i < img_size; i++) // raises fft_image_input to the power 2
                fft_image_input[i] = fft_image_input[i] * fft_image_input[i];
        }
        else if (!sqrImage) { // cache output of Fourier for image 
            fft_image_output_cached = fft_image_output;
        }

        // Fourier for image
        fftw_execute(fft_image_plan);
        fftw_free(fft_image_input);
        fftw_destroy_plan(fft_image_plan);
    }

    // get output of Fourier for image from cache
    if (!sqrImage) fft_image_output = fft_image_output_cached;


    std::fill(fft_object_input, fft_object_input+img_size, 0); 
    // Create a mask of object
    for (int y = 0; y < img_height; y++)
        for (int x = 0; x < img_width; x++) {
            int index = y*img_width + x;
            int obj_index = y*img_width + x;

            fft_object_input[index] = chi[ obj_index ];
        }

    // Fourier for object
    fftw_execute(fft_object_plan);

    // Multiply Fourier images
    for (int i = 0; i < img_size; i++) { 
        fft_multiplied[i][0] = (fft_image_output[i][0] * fft_object_output[i][0]) - (fft_image_output[i][1] * fft_object_output[i][1]);
        fft_multiplied[i][1] = (fft_image_output[i][0] * fft_object_output[i][1]) + (fft_image_output[i][1] * fft_object_output[i][0]);
    }

    // Inverse Fourier
    fftw_execute(ifft_plan); 

    if (sqrImage) {
        fftw_free(fft_image_output);
        fftw_free(fft_image_output_cached);
     }


    double *result = fftw_alloc_real(img_size); 
    for (int i = 0; i < img_size; i++) {
        result[i] = ifft_result[i] / img_size;
        if (result[i] < 10e-5) {
            result[i] = 0;
        }
    }

    return result;
}


QRect FindObject::find()
{
    std::vector< std::vector<bool> > chi( chi_count );
    for (int i = 0; i < chi_count; i++) {
        chi[i] = std::vector<bool>(img_size);
    }

    // Rotate object in \chi        // Convert to GrayScale //Stanislav
    for (int h = 0; h < obj_height; h++) {
        for (int w = 0; w < obj_width; w++) {
            int color = qGray(object->pixel(w,h)) / (256.0 / chi_count); // grayscale == (r * 11 + g * 16 + b * 5) / 32
            // Reflects chi horisontally & vertically and 
            // places object indicator function in left top corner of matrix chi, other -- zeros  
            int index = (obj_height - h - 1)*img_width + (obj_width - w - 1); 
            chi[ color ][ index ] = true; // filling chi-masks
        }
    }


    std::vector<long> chi_elements(chi_count); // Let's count how many pixels have color == i
    // chi_elements[i] -- number of grayscale object pixels that have color == i
    for (int i = 0; i < chi_count; i++) {
        chi_elements[i] = count(chi[i].begin(), chi[i].end(), 1);
    }

    std::vector<double*> convolution(chi_count);
    for (int i = 0; i < chi_count; i++) { // in this loop TONS of memory allocated ! (8.2 Mb per iteration)
        convolution[i] = (chi_elements[i] > 0) ? calc_convolution(chi[i], false) : NULL;
    }

    std::vector<bool> f_chi(img_size);
    for (int h = 0; h < obj_height; h++)
        for (int w = 0; w < obj_width; w++) {
            int index = h * img_width + w;
            f_chi[index] = true;
        }

    double *f = calc_convolution(f_chi, true);

    int conv_offset_x = obj_width - 1;
    int conv_offset_y = obj_height - 1;

    // ||f- Ph(f)||^2 -- chislitel'
    std::vector<double> numerator(img_size);
    std::fill(numerator.begin(), numerator.end(), 66*10e100); // default value
    for (int h = 0; h <= (img_height - obj_height); h++)
        for (int w = 0; w <= (img_width - obj_width); w++) {
            int index = img_width * h + w;
            int index_k = (img_width) * (h + conv_offset_y) + (w + conv_offset_x); //?
            double sum = 0;
            for (int k = 0; k < chi_count; k++) {
                if (chi_elements[k] > 0) {
                    sum += (convolution[k][index_k] * convolution[k][index_k]) / chi_elements[k];
                }
            }
            numerator[index] = fabs(f[index_k] - sum);
        }



    // ||Po(f) - Pn(f)||^2  -- znamenatel'
    std::vector<double> denominator(img_size);
    for (int h = 0; h <= (img_height - obj_height); h++)
        for (int w = 0; w <= (img_width - obj_width); w++) {
            int index = img_width * h + w;
            int index_k = (img_width) * (h + conv_offset_y) + (w + conv_offset_x); //???
            double sum = 0;
            double sum_i = 0;
            double sum_k = 0;
            for (int k = 0; k < chi_count; k++) {
                if (chi_elements[k] > 0) {
                    sum_i += convolution[k][index_k];
                    sum_k += chi_elements[k];
                    sum += (convolution[k][index_k] * convolution[k][index_k] * 1.0) / chi_elements[k];
                }
            }
            denominator[index] = sum - (sum_i*sum_i*1.0 / sum_k);
            denominator[index] = fabs(denominator[index]);
        }


    for (int i = 0; i < chi_count; i++) {
            if (convolution[i] != NULL) {
                fftw_free(convolution[i]);
            }
        }


    std::vector<double> result(img_size);
    for (int i = 0; i < img_size; i++)
        result[i] = numerator[i] / denominator[i];


    int index = std::min_element(result.begin(), result.end()) - result.begin(); // index of top-left corner with minimal tau

    fftw_free(f);

    qDebug() << "object->rect(): " << object->rect();

    // it is really how you can calculate x,y
    return QRect(index % image->width(), index / image->width(), object->width(), object->height()); 
}
