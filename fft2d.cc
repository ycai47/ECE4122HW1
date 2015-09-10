// Distributed two-dimensional Discrete FFT transform
// Yushan Cai
// ECE4412 Project 1

//mpirun -np 16 ./fft2d
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>
#include "Complex.h"
#include "InputImage.h"

using namespace std;

//transpose function for 1d array
void Transpose(Complex* src, int& height, int& width)
{
  Complex* temp = new Complex[height*width];
  memcpy(temp, src, sizeof(Complex) * height * width);
  int r = 0, c = 0, index = 0;
  for(int j = 0; j < height * width; j++)
  {
    r = j / width;
    c = j % width;
    index = c * height + r;
    src[index] = temp[j];
  }
  delete [] temp;
  return;
}

// 1-d DFT using the double summation equation
// h is the time-domain input data, w is the width (N), and H is the output array
void Transform1D(Complex* h, int w, Complex* H)
{
  Complex W_nk;
  for (int n = 0; n < w; ++n)
  {
    H[n].real = 0;
    H[n].imag = 0; //initialize H[n]
    for (int k = 0; k < w; ++k)
    {
      W_nk.real = cos(double(2 * M_PI * n * k / w));
      W_nk.imag = - sin(double(2 * M_PI * n * k / w));
      H[n] = H[n] + W_nk * h[k];
    }
    if (fabs(H[n].real) < 1E-10) H[n].real = 0;
    if (fabs(H[n].imag) < 1E-10) H[n].imag = 0; //adjust double precision
  }
}

//test 1-d DFT
void Test1D(InputImage * image)
{
  int width = image->GetWidth();
  int height = image->GetHeight();
  Complex * h = image->GetImageData();
  Complex * H = new Complex[width * height];
  for (int i = 0; i < height; ++i)
  {
    Transform1D(&h[i * width], width, &H[i * width]);
  }
  image->SaveImageData("result_1d.txt", H, width, height);
  string fn("after1d.txt");
  InputImage * data = new InputImage(fn.c_str());
  Complex * result = data->GetImageData();
  for (int i = 0; i < height*width; ++i)
  {
    if(fabs(H[i].real - result[i].real) < 0.01 && fabs(H[i].imag - result[i].imag) < 0.01) continue;
    else
    {
      printf("value on row %d, col %d is not correct.\n", i / width, i % width);
      break;
    }
  }
  delete [] H;
  return;
}

void Transform2D(const char* inputFN) 
{
  // Use the InputImage object to read in the Tower.txt file and
  // find the width/height of the input image.
    InputImage * image = new InputImage(inputFN); // helper object for reading the image
    //Test1D(image);
    int width = image->GetWidth();
    int height = image->GetHeight();
  // find how many CPUs in total, and which one this process is
    int rank, nCPUs, rc;
    MPI_Status status;
    MPI_Request request;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nCPUs); 
  // Obtain a pointer to the Complex 1d array of input data
    Complex * data = image->GetImageData();
    const int row_per_proc = height / nCPUs;
    int start_row = rank * row_per_proc;
    Complex * result_1d = new Complex[row_per_proc * width];
    if (result_1d == NULL)
    {
      printf("Error: memory could not be allocated on rank %d", rank);
    }
  // Individual 1D transforms on the rows assigned to current CPU
    for (int i = 0; i < row_per_proc; ++i)
    {
      Transform1D(&data[(start_row + i) * width], width, &result_1d[i * width]);
    }
    // printf("first 1d transform for rank %d done\n", rank);
  // Send the resultant transformed values to CPU 0
    rc = MPI_Isend(result_1d, row_per_proc * width * sizeof(Complex), MPI_BYTE, 0,
              0, MPI_COMM_WORLD, &request);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank << "first send to CPU 0 failed, rc " << rc << endl;
      MPI_Finalize();
      return;
    }
  //CPU 0 receives all results from first 1d transform and perform transpose and then send back to proper CPUs
    if (rank == 0)
    {
      for (int i = 0; i < nCPUs; ++i)
      {
        rc = MPI_Recv(&data[i * row_per_proc * width], row_per_proc * width * sizeof(Complex), MPI_BYTE, i,
                  0, MPI_COMM_WORLD, &status);
      }
      Transpose(data, height, width);
      for (int i = 0; i < nCPUs; ++i)
      {
        rc = MPI_Isend(&data[i * row_per_proc * width], row_per_proc * width * sizeof(Complex), MPI_BYTE, i,
                  0, MPI_COMM_WORLD, &request);
      }
    }
  // Receive rows from CPU 0
    Complex * mess_recv = new Complex[row_per_proc * width];
    rc = MPI_Recv(mess_recv, row_per_proc * width * sizeof(Complex), MPI_BYTE, 0,
              0, MPI_COMM_WORLD, &status);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank << " recv failed, rc " << rc << endl;
      MPI_Finalize();
      return;
    }
  // When all rows received, do the 1D transforms on the rows
    Complex * result = new Complex[row_per_proc * width];
    for (int i = 0; i < row_per_proc; ++i)
    {
      Transform1D(&mess_recv[i * width], width, &result[i * width]);
      // printf("second 1d transform for rank %d done\n", rank);
    }
  // Send final answers to CPU 0
    rc = MPI_Isend(result, row_per_proc * width * sizeof(Complex), MPI_BYTE, 0,
              0, MPI_COMM_WORLD, &request);
    if (rc != MPI_SUCCESS)
    {
      cout << "Rank " << rank << "final send to CPU 0 failed, rc " << rc << endl;
      MPI_Finalize();
      return;
    }
  // If CPU 0, collect all values from other processors and transpose back and print out with SaveImageData().
    if(rank == 0)
    {
      Complex * result_2d = new Complex[width * height];
      for (int i = 0; i < nCPUs; ++i)
      {
        rc = MPI_Recv(&result_2d[i * row_per_proc * width], row_per_proc * width * sizeof(Complex), MPI_BYTE, i,
                  0, MPI_COMM_WORLD, &status);
        if (rc != MPI_SUCCESS)
        {
          cout << "Rank " << i << " recv failed, rc " << rc << endl;
          MPI_Finalize();
          return;
        }
      }
      Transpose(result_2d, height, width);
      string fn("MyAfter2d.txt");
      image->SaveImageData(fn.c_str(), result_2d, width, height);
      delete [] result_2d;
    }
    //clean up
    delete [] result_1d;
    delete [] mess_recv;
    delete [] result;
    return;
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization
  int rc = MPI_Init(NULL, NULL);
  if (rc != MPI_SUCCESS)
  {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  Transform2D(fn.c_str()); // Perform the transform.
  // Finalize MPI
  MPI_Finalize();
}