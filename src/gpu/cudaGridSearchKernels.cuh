/************************************************************************
 Sample CUDA MEX kernel code written by Fang Liu (leoliuf@gmail.com).
 ************************************************************************/

#ifndef _ADD_KERNEL_GPU_H_
#define _ADD_KERNEL_GPU_H_

#include <stdio.h>
#include <mex.h> 
#include <cstdint>
#include <cuda/std/limits>
#define mexPrintf printf

__global__ void
gpuAddKernel(double *d_A, double *d_B, double *d_C, mwSignedIndex Am, mwSignedIndex An) {
    /* index */
    unsigned int tid = blockIdx.x * blockDim.y + threadIdx.y; /* thread id in matrix*/
    /* strip */
    unsigned int strip = gridDim.x * blockDim.y;


    //    int i = threadIdx.x;
    //    int j = threadIdx.y;
    //    int index = j * An + i;
    //    if (index < Am * An) {
    //        printf("filled position (%d,%d)\n", j, i);
    //        d_C[index] = d_A[index] + d_B[index];
    //    }
    while (1) {
        if (tid < Am * An) {
            //printf("filled position (%d)\n", tid);
            d_C[tid] = d_A[tid] + d_B[tid];
        } else {
            break;
        }
        tid += strip;
    }
}

class GridSearchableXY {
public:

    virtual CUDAFUNCTION float evalXY(int x, int y) = 0;
    virtual CUDAFUNCTION void evalXY(int* x, int* y, float* result) = 0;
};

template<typename pixType>
class ImageXY {
    int _width;
    int _height;
    pixType* _values;
    float _mean;
    float _variance;

public:

    ImageXY() : _width(0), _height(0), _values(nullptr),
    _mean(cuda::std::numeric_limits<float>::infinity()),
    _variance(cuda::std::numeric_limits<float>::infinity()) {

    }

    ImageXY(int width, int height) : ImageXY() {
        _width = width;
        _height = height;
        _values = new pixType[_width * _height];
    }

    ImageXY(int height, int width, const pixType* values) : ImageXY(width, height) {
        // do memcpy instead?
        for (int x = 0; x < _width; x++) {
            for (int y = 0; y < _height; y++) {
                _values[toIndex(x, y)] = values[toIndex(x, y)];
            }
        }
    }

    ImageXY(const ImageXY<pixType>& img) : ImageXY(img.width(), img.height(), img.values()) {
    }

    ~ImageXY() {
        delete[] _values;
    }

    void setWidth(int width) {
        _width = width;
    }

    void setHeight(int height) {
        _height = height;
    }

    CUDAFUNCTION
    int width() const {
        return _width;
    }

    CUDAFUNCTION
    int height() const {
        return _height;
    }

    // non-const to allow overwriting this value when copying the object to 
    // a GPU device

    void setValues(pixType* values) {
        _values = values;
    }

    CUDAFUNCTION
    pixType* values() const {
        //const pixType* values() const {
        return _values;
    }

    CUDAFUNCTION
    pixType& at(int x, int y) const {
        return _values[toIndex(x, y)];
    }

    template<typename T>
    CUDAFUNCTION
    int toIndex(T x, T y) const {
        // column-major ordering of _data
        return ((int) x)*_height + (int) y;
    }

    CUDAFUNCTION
    bool inImage(float x, float y) const {
        if (x >= 0 && y >= 0 && x < _width && y < _height)
            return true;
        return false;
    }

    CUDAFUNCTION
    float valueAt(float x, float y) const {
        return valueAt_nearest_neighbor(x, y);
    }

    CUDAFUNCTION
    float valueAt_nearest_neighbor(float x, float y) const {
        return (inImage(x, y)) ? _values[toIndex(x, y)] : cuda::std::numeric_limits<float>::infinity();
    }

    CUDAFUNCTION
    float mean() {
        // compute mean only one time
        if (_mean == cuda::std::numeric_limits<float>::infinity()) {
            _mean = 0;
            for (pixType* val = &_values[0]; val - &_values[0] < _width * _height; val++) {
                _mean += *val;
            }
            _mean /= (_width * _height);
        }
        return _mean;
    }

    CUDAFUNCTION
    float variance() {
        // compute mean only one time
        if (_mean == cuda::std::numeric_limits<float>::infinity()) {
            mean();
        }
        // compute variance only one time
        if (_variance == cuda::std::numeric_limits<float>::infinity()) {
            _variance = 0;
            for (pixType* val = &_values[0]; val - &_values[0] < _width * _height; val++) {
                _variance += (*val - _mean)*(*val - _mean);
            }
            _variance /= (_width * _height - 1);
        }
        return _variance;
    }

    void print() const {
        printf("[ ");
        for (int x = 0; x < _width; x++) {
            for (int y = 0; y < _height; y++) {
                printf("%0.1f ", (float) valueAt(x, y));
            }
            printf("\n");
        }
        printf("]\n");
    }
};

template<typename pixType>
class ImageErrorFunctionalXY : public GridSearchableXY {
protected:
    ImageXY<pixType> img_moved;
    ImageXY<pixType> img_fixed;

public:

    ImageErrorFunctionalXY(const ImageXY<pixType>& _img_moved, const ImageXY<pixType>& _img_fixed) :
    img_moved(_img_moved),
    img_fixed(_img_fixed) {
    }

    virtual CUDAFUNCTION float evalXY(int x, int y) = 0;
    virtual CUDAFUNCTION void evalXY(int* x, int* y, float* result) = 0;
    virtual CUDAFUNCTION void evalXY(int* x, int* y, float* result, ImageXY<pixType>* img_moved, ImageXY<pixType>* img_fixed) = 0;

    const ImageXY<pixType>& imageMoved() {
        return img_moved;
    }

    const ImageXY<pixType>& imageFixed() {
        return img_fixed;
    }
};

#define CALL_GRIDSEARCH_FN_REF(object, ptrToMember)  ((object).*(ptrToMember))
#define CALL_GRIDSEARCH_FN_PTR(object, ptrToMember)  ((object)->*(ptrToMember))

template<typename pixType>
class NegativeCrossCorrelationSearchXY : public ImageErrorFunctionalXY<pixType> {
public:
    typedef float(NegativeCrossCorrelationSearchXY<pixType>::*gridsearchFunc)(int x, int y);
    gridsearchFunc evalXY_func;

    NegativeCrossCorrelationSearchXY(const ImageXY<pixType>& _img_moved, const ImageXY<pixType>& _img_fixed) :
    ImageErrorFunctionalXY<pixType>(_img_moved, _img_fixed),
    // this can point to different implementations of error functions
    //evalXY_func(&SumOfAbsoluteDifferencesSearchXY::gridsearch_evalXY_func_1)
    evalXY_func(&NegativeCrossCorrelationSearchXY<pixType>::gridsearch_evalXY_func_2) {
        // invoke as follows for object references
        // CorrelationSearch cs1;
        // CALL_GRIDSEARCH_FN_REF(cs1, cs1.evalXY)(0,0);
        // or equivalently
        // cs1.evalXY();
        // invoke as follows for object pointers
        // CorrelationSearch* cs2 = new CorrelationSearch();
        // CALL_GRIDSEARCH_FN_PTR(cs2, cs2.evalXY)(0,0);
        // or equivalently
        // cs2->evalXY();
    }

    CUDAFUNCTION
    float evalXY(int x, int y) {
        return CALL_GRIDSEARCH_FN_PTR(this, evalXY_func)(x, y);
    }

    CUDAFUNCTION
    void evalXY(int* x, int* y, float* result) {
        //return CALL_GRIDSEARCH_FN_PTR(this, evalXY_func2)(x, y);
    }

    CUDAFUNCTION
    void evalXY(int* x, int* y, float* result, ImageXY<pixType>* img_moved, ImageXY<pixType>* img_fixed) {
        
    }
    
    CUDAFUNCTION
    float gridsearch_evalXY_func_1(int tx, int ty) {
        return negativeCrossCorrelation(tx, ty, this->img_moved, this->img_fixed);
        //return 0.0f;
    }

    CUDAFUNCTION
    float negativeCrossCorrelation(int tx, int ty, ImageXY<pixType>& image_fixed, ImageXY<pixType>& image_moved) {
        float cross_correlation = 0;
        float mean_fixed = image_fixed.mean();
        float variance_fixed = image_fixed.variance();
        float mean_moved = image_moved.mean();
        float variance_moved = image_moved.variance();
        for (int x = 0; x < image_fixed.width(); x++) {
            for (int y = 0; y < image_fixed.height(); y++) {
                float image_moved_value = image_moved.valueAt(x + tx, y + ty);
                if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    cross_correlation += (image_moved_value - mean_moved)*(image_fixed.valueAt(x, y) - mean_fixed);
                }
            }
        }
        float std_dev_moved_and_fixed = sqrtf(variance_moved * variance_fixed);
        cross_correlation /= std_dev_moved_and_fixed;
        return -cross_correlation;
    }

    CUDAFUNCTION
    float gridsearch_evalXY_func_2(int x, int y) {
        return averageNegativeCrossCorrelation(x, y, this->img_moved, this->img_fixed);
        //return 0.0f;
    }

    CUDAFUNCTION
    float averageNegativeCrossCorrelation(int tx, int ty, ImageXY<pixType>& image_fixed, ImageXY<pixType>& image_moved) {
        int num_errors = 0;
        float cross_correlation = 0;
        float mean_fixed = image_fixed.mean();
        float variance_fixed = image_fixed.variance();
        float mean_moved = image_moved.mean();
        float variance_moved = image_moved.variance();
        for (int x = 0; x < image_fixed.width(); x++) {
            for (int y = 0; y < image_fixed.height(); y++) {
                float image_moved_value = image_moved.valueAt(x + tx, y + ty);
                if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    cross_correlation += (image_moved_value - mean_moved)*(image_fixed.valueAt(x, y) - mean_fixed);
                    num_errors++;
                }
            }
        }
        float std_dev_moved_and_fixed = sqrtf(variance_moved * variance_fixed);
        cross_correlation /= std_dev_moved_and_fixed;
        return -cross_correlation / num_errors;
    }
};

template<typename pixType>
class SumOfAbsoluteDifferencesSearchXY : public ImageErrorFunctionalXY<pixType> {
public:
    typedef float(SumOfAbsoluteDifferencesSearchXY<pixType>::*gridsearchFunc)(int x, int y);
    gridsearchFunc evalXY_func;
    typedef void(SumOfAbsoluteDifferencesSearchXY<pixType>::*gridsearchFunc2)(int* x, int* y, float* result);
    gridsearchFunc2 evalXY_func2;

    SumOfAbsoluteDifferencesSearchXY(const ImageXY<pixType>& _img_moved, const ImageXY<pixType>& _img_fixed) :
    ImageErrorFunctionalXY<pixType>(_img_moved, _img_fixed),
    // this can point to different implementations of error functions
    //evalXY_func(&SumOfAbsoluteDifferencesSearchXY::gridsearch_evalXY_func_1)
    evalXY_func(&SumOfAbsoluteDifferencesSearchXY::gridsearch_evalXY_func_2),
    evalXY_func2(&SumOfAbsoluteDifferencesSearchXY::gridsearch_evalXY_func_2a) {
        // invoke as follows for object references
        // CorrelationSearch cs1;
        // CALL_GRIDSEARCH_FN_REF(cs1, cs1.evalXY)(0,0);
        // or equivalently
        // cs1.evalXY();
        // invoke as follows for object pointers
        // CorrelationSearch* cs2 = new CorrelationSearch();
        // CALL_GRIDSEARCH_FN_PTR(cs2, cs2.evalXY)(0,0);
        // or equivalently
        // cs2->evalXY();
    }

    CUDAFUNCTION
    float evalXY(int x, int y) {
        return CALL_GRIDSEARCH_FN_PTR(this, evalXY_func)(x, y);
    }

    CUDAFUNCTION
    void evalXY(int* x, int* y, float* result) {
        CALL_GRIDSEARCH_FN_PTR(this, evalXY_func2)(x, y, result);
    }

    CUDAFUNCTION
    void evalXY(int* tx, int* ty, float* result, ImageXY<pixType>* img_moved, ImageXY<pixType>* img_fixed) {
        int num_errors = 0;
        float sum_of_absolute_differences = 0;
        for (int x = 0; x < img_fixed->width(); x++) {
            for (int y = 0; y < img_fixed->height(); y++) {
                float image_moved_value = img_moved->valueAt(x + *tx, y + *ty);
                if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    sum_of_absolute_differences += std::abs(image_moved_value - img_fixed->valueAt(x, y));
                    num_errors++;
                }
            }
        }
        *result = sum_of_absolute_differences / num_errors;        
    }

    CUDAFUNCTION
    float gridsearch_evalXY_func_1(int tx, int ty) {
        return sumOfAbsoluteDifferences(tx, ty, this->img_moved, this->img_fixed);
        //return 0.0f;
    }

    CUDAFUNCTION
    float sumOfAbsoluteDifferences(int tx, int ty, ImageXY<pixType>& image_fixed, ImageXY<pixType>& image_moved) {
        float sum_of_absolute_differences = 0;
        for (int x = 0; x < image_fixed.width(); x++) {
            for (int y = 0; y < image_fixed.height(); y++) {
                float image_moved_value = image_moved.valueAt(x + tx, y + ty);
                if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    sum_of_absolute_differences += std::abs(image_moved_value - image_fixed.valueAt(x, y));
                }
            }
        }
        return sum_of_absolute_differences;
    }

    CUDAFUNCTION
    float gridsearch_evalXY_func_2(int x, int y) {
        return averageSumOfAbsoluteDifferences(x, y, this->img_moved, this->img_fixed);
        //return 0.0f;
    }

    CUDAFUNCTION
    void gridsearch_evalXY_func_2a(int* tx, int* ty, float* result) {
        *result = sumOfAbsoluteDifferences(*tx, *ty, this->img_moved, this->img_fixed);
        //return 0.0f;
    }

    CUDAFUNCTION
    float averageSumOfAbsoluteDifferences(int tx, int ty, ImageXY<pixType>& image_fixed, ImageXY<pixType>& image_moved) {
        int num_errors = 0;
        float sum_of_absolute_differences = 0;
        for (int x = 0; x < image_fixed.width(); x++) {
            for (int y = 0; y < image_fixed.height(); y++) {
                float image_moved_value = image_moved.valueAt(x + tx, y + ty);
                if (image_moved_value != cuda::std::numeric_limits<float>::infinity()) {
                    sum_of_absolute_differences += std::abs(image_moved_value - image_fixed.valueAt(x, y));
                    num_errors++;
                }
            }
        }
        return sum_of_absolute_differences / num_errors;
    }
};

class GridPointXY {
    int _x, _y;
    float _value;

public:

    GridPointXY(int x, int y, float value) : _value(value), _x(x), _y(y) {
    }

    int& x() {
        return _x;
    }

    int& y() {
        return _y;
    }

    float& value() {
        return _value;
    }
};

template<typename pixType>
class ImageTranslationGridSearchXY {
    std::vector<float> _gridvalues;

    ImageXY<float>* _gridvalues_2;
    ImageErrorFunctionalXY<pixType>* _imageErrorFuncXY;
    GridPointXY _minValue;
    GridPointXY _maxValue;

public:

    ImageTranslationGridSearchXY(ImageErrorFunctionalXY<pixType>* imageErrorFuncXY) :
    _gridvalues_2(nullptr),
    _imageErrorFuncXY(imageErrorFuncXY),
    _minValue(cuda::std::numeric_limits<int>::infinity(), cuda::std::numeric_limits<int>::infinity(), cuda::std::numeric_limits<float>::infinity()),
    _maxValue(-cuda::std::numeric_limits<int>::infinity(), -cuda::std::numeric_limits<int>::infinity(), -cuda::std::numeric_limits<float>::infinity()) {
    }

    ~ImageTranslationGridSearchXY() {
        delete _gridvalues_2;
    }

    void run() {
        const ImageXY<pixType>& image_moved = _imageErrorFuncXY->imageMoved();
        const ImageXY<pixType>& image_fixed = _imageErrorFuncXY->imageFixed();
        bool computeSwappedSolution = (image_moved.width() * image_moved.height() > image_fixed.width() * image_fixed.height());
        //int min_size = (!computeSwappedSolution) ? image_moved._width() * image_moved._height() : image_fixed._width() * image_fixed._height();
        int fixed_width = (!computeSwappedSolution) ? image_fixed.width() : image_moved.width();
        int moved_width = (!computeSwappedSolution) ? image_moved.width() : image_fixed.width();
        int fixed_height = (!computeSwappedSolution) ? image_fixed.height() : image_moved.height();
        int moved_height = (!computeSwappedSolution) ? image_moved.height() : image_fixed.height();
        int tx_min = -(moved_width / 2);
        int tx_max = fixed_width - (moved_width / 2);
        int ty_min = -(moved_height / 2);
        int ty_max = fixed_height - (moved_height / 2);
        _gridvalues.resize((tx_max - tx_min + 1)*(ty_max - ty_min + 1), 0);
        _gridvalues_2 = new ImageXY<float>(tx_max - tx_min + 1, ty_max - ty_min + 1);
        for (int x = tx_min; x <= tx_max; x++) {
            for (int y = ty_min; y <= ty_max; y++) {
                float value = _imageErrorFuncXY->evalXY(x, y);
                _gridvalues.push_back(value);
                _gridvalues_2->at(x - tx_min, y - ty_min) = value;
                if (value < _minValue.value()) {
                    _minValue.x() = x;
                    _minValue.y() = y;
                    _minValue.value() = value;
                }
                if (value > _maxValue.value()) {
                    _maxValue.x() = x;
                    _maxValue.y() = y;
                    _maxValue.value() = value;
                }
            }
        }
        if (computeSwappedSolution) {
            // change the (x,y) locations of the _minValue and _maxValue to be
            // the inverse transformation
        }
    }

    const ImageXY<float>& getImageXY() {
        return *_gridvalues_2;
    }

    const float* values() {
        return _gridvalues_2->values();
    }
};

__device__
void f1(int* x, int* y, ImageErrorFunctionalXY<uint8_t>* error_func, float* result) {
    if ((!threadIdx.x) && (!blockIdx.x)) printf("error(%d,%d) = %f\n", *x, *y, 1.0f);
    *result = 5.4321f;
}

__device__
void f2(double* vol, int* a) {
    if ((!threadIdx.x) && (!blockIdx.x)) printf("value = %f\n", *vol);
    *vol += 5.4321f;
}

template <typename... Types>
__global__ void setup_kernel(void (**my_callback)(Types*...)) {
    *my_callback = f1;
}

__global__ void gridsearch_kernel(SumOfAbsoluteDifferencesSearchXY<uint8_t>* d_gsobj,
        ImageXY<uint8_t>* d_imageMoved, ImageXY<uint8_t>* d_imageFixed) {
    // convert grid, block and thread indices into a global search space index
    // convert the global search space index into a collection of arguments
    // for the grid search function call
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    //float val0 = 1.2345f;
    //  // does not use gpu (0% gpu utilization)
    //  for ( int i = 0; i < 1000000; i++ ) {
    //g.f(&val0, &val1);
    //d_gsobj->*callback(&x, &y, nullptr, &val0);
    float result;
    d_gsobj->evalXY(&x, &y, &result, d_imageMoved, d_imageFixed);
    //  }
    //val0 = 0.0f;

    // uses gpu (99% gpu utilization)
    //  for ( int i = 0; i < 10000000; i++ ) {
    //f1(&x, &y, nullptr, &val0);
    //  }
    printf("result = %f\n", result);
    if ((!threadIdx.x)&&(!blockIdx.x)) printf("in-kernel func_d()   address = %p\n", f2);
}

// host function

__device__
void f3(int* x, int* y, ImageErrorFunctionalXY<uint8_t>* error_func, float* result) {
    if ((!threadIdx.x) && (!blockIdx.x)) printf("error(%d,%d) = %f\n", *x, *y, 1.0f);
    *result = 5.4321f;
}

__global__ void setup_gs_kernel(SumOfAbsoluteDifferencesSearchXY<uint8_t>::gridsearchFunc2* my_callback, SumOfAbsoluteDifferencesSearchXY<uint8_t>* d_gsobj) {
    *my_callback = d_gsobj->evalXY_func2;
}

template <typename... Types>
void host_func(void (*callback)(Types*...)) {
    // get user kernel number of arguments.
    constexpr int I = sizeof...(Types);
    printf("size of Args = %d\n", I);

    printf("callback() address = %p\n", callback);
    //printf("func_d()   address = %p\n", func_d);

    dim3 nblocks = 100;
    int nthread = 100;
    unsigned long long *d_callback, h_callback;
    cudaMalloc(&d_callback, sizeof (unsigned long long));
    setup_kernel << <1, 1 >> >((void (**)(Types*...))d_callback);
    cudaMemcpy(&h_callback, d_callback, sizeof (unsigned long long), cudaMemcpyDeviceToHost);
    printf("host reference to in-kernel func_d()   address = %p\n", (void *) h_callback);
    //gridsearch_kernel<Types...> << <nblocks, nthread>>>((void (*)(Types*...))h_callback);
    cudaDeviceSynchronize();
}


#endif