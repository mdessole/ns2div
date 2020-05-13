#include <thrust/system_error.h>
#include<thrust/device_vector.h>
#include<thrust/reduce.h>
#include<thrust/sort.h>
#include<iostream>
typedef double                    type_fd;
typedef unsigned long long int    type_uint;




using std::cout;
using std::endl;
using std::cerr;

extern "C" int sort_by_key(type_uint* dev_key_ptr, type_fd* dev_val_ptr, int n) {

  try{
    //cout << "iniziato sort thrust" << endl;
    thrust::device_ptr<type_uint> key_ptr(dev_key_ptr);
    thrust::device_ptr<type_fd> val_ptr(dev_val_ptr);
    thrust::sort_by_key(key_ptr, key_ptr + n, val_ptr);
  }catch(std::bad_alloc &e){
      std::cerr << "Ran out of memory while sorting" << std::endl;
      exit(-1);
  }catch(thrust::system_error &e){
    // output an error message and exit
    cerr << "Errore in thrust sort " << e.what() << endl;
    exit(-1);
  }
  return 0;
}

extern "C" int reduce_by_key(type_uint* dev_key_ptr,     type_fd* dev_val_ptr,
			     type_uint* out_dev_key_ptr, type_fd* out_dev_val_ptr, int n) {
  // arrays must be sorted by key
  try{
    thrust::device_ptr<type_uint> key_ptr(dev_key_ptr);
    thrust::device_ptr<type_fd> val_ptr(dev_val_ptr);
    thrust::device_ptr<type_uint> out_key_ptr(out_dev_key_ptr);
    thrust::device_ptr<type_fd> out_val_ptr(out_dev_val_ptr);
    thrust::reduce_by_key(key_ptr, key_ptr + n, val_ptr, out_key_ptr, out_val_ptr);
  }catch(thrust::system_error &e)
    {
      // output an error message and exit
      cerr << "Errore in thrust reduce " << e.what() << endl;
      exit(-1);
    }
  return 0;
}


extern "C" int reduction(type_fd *x, type_fd *y, int n) {
  // arrays must be sorted by key
  type_fd tot = 0.0;
  try{
    thrust::device_ptr<type_fd> x_ptr(x);
    thrust::device_ptr<type_fd> y_ptr(y);
    tot = thrust::reduce(x_ptr, x_ptr + n);
    thrust::fill( y_ptr, y_ptr + 1, tot);
    //std::cout << "tot  = " << tot << std::endl;
  }catch(thrust::system_error &e)
    {
      // output an error message and exit
      cerr << "Errore in thrust reduction " << e.what() << endl;
      exit(-1);
    }
  return 0;
}

extern "C" int reduction_int(int *x, int *y, int n) {
  // arrays must be sorted by key
  type_fd tot = 0.0;
  try{
    thrust::device_ptr<int> x_ptr(x);
    thrust::device_ptr<int> y_ptr(y);
    tot = thrust::reduce(x_ptr, x_ptr + n);
    thrust::fill( y_ptr, y_ptr + 1, tot);
    //std::cout << "tot  = " << tot << std::endl;
  }catch(thrust::system_error &e)
    {
      // output an error message and exit
      cerr << "Errore in thrust reduction " << e.what() << endl;
      exit(-1);
    }
  return 0;
}

extern "C" int inclusive_scan(int *x, int n) {
  // arrays must be sorted by key

  try{
    thrust::device_ptr<int> x_ptr(x);
    thrust::inclusive_scan(x_ptr, x_ptr + n, x_ptr);
  }catch(thrust::system_error &e)
    {
      // output an error message and exit
      cerr << "Errore in thrust inclusive_scan " << e.what() << endl;
      exit(-1);
    }
  return 0;
}
