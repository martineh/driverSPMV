#include "utils.hpp"
#include "colors.hpp"

#ifdef LIB_PETSC
  #include "petscmat.h"
  #include "mmloader.h"
#else
  #include "ginkgo/ginkgo.hpp"
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <dirent.h>

using ValueType = double;  // FP64
using IndexType = int;

void spmv_baseline (const IndexType* row_ptrs, const IndexType* col_idxs, const ValueType* values,
    IndexType num_rows, const ValueType* b, ValueType* c) {
  for (IndexType row = 0; row < num_rows; ++row) {
    c[row] = 0.0;
    for (IndexType k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
      c[row] += values[k] * b[col_idxs[k]];
    }
  }
}

double gemm_validation(size_t m, const ValueType *Vref, const ValueType *V) {
  double error = 0.0;
  double nrm   = 0.0;
  double tmp;

  for ( size_t i = 0; i < m; i++ ) {
    tmp = (double) Vref[i]*Vref[i];
    nrm += tmp*tmp;
    tmp = (double) dabs(Vref[i]-V[i]);
    error += tmp*tmp;
  }

  if ( nrm!=0.0 ) error = sqrt(error) / sqrt(nrm);
  else            error = sqrt(error);

  return error;
}



int main(int argc, char* argv[]) {
  char   matrix_name[512];
  double tmin  = 0.0;
  int    nreps = 0;
  double time  = 0.0;
  char   test;
  double t1, t2, flops, GFLOPS, err;
  char   *directory, *logs;
  DIR    *dir;
  size_t mnz;
  struct dirent *entry;
  double err_limit = 1.0e-10;
  int    nrows, ncols;
  FILE   *fd_logs;

  tmin      = atof(argv[1]); 
  test      = argv[2][0];
  directory = argv[3];
  logs      = argv[4];

  dir = opendir(directory);
  if (!dir) {
      perror("ERR: Failed to open the directory\n");
      exit(-1);
  }

  fd_logs = fopen(logs, "w");
  fprintf(fd_logs, "#Matrix_name;MNZ;M;K;Time;GFlops\n");

  printf("\n");
  printf("=====================================================\n");
  printf("|             %sSPMV DRIVER CONFIGURATION%s             |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf("=====================================================\n");
  printf("| Minimum Time (s) : %s%-30.2f%s |\n", COLOR_BOLDCYAN, tmin, COLOR_RESET);
  printf("| Test             : %s%-30c%s |\n", COLOR_BOLDCYAN, test, COLOR_RESET); 
  printf("| Matrix Path      : %s%-30s%s |\n", COLOR_BOLDCYAN, directory, COLOR_RESET); 
  printf("| Output Log       : %s%-30s%s |\n", COLOR_BOLDCYAN, logs, COLOR_RESET); 
  #ifdef LIB_PETSC
  printf("| Library selected : %s%-30s%s |\n", COLOR_BOLDCYAN, "PETSc", COLOR_RESET); 
  #else
  printf("| Library selected : %s%-30s%s |\n", COLOR_BOLDCYAN, "Ginkgo", COLOR_RESET); 
  #endif
  printf("=====================================================\n\n");

  printf("=============================================================================================================\n");
  printf("|                                               %sSPMV DRIVER%s                                                 |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf("=============================================================================================================\n");
  printf("|  MATRIX NAME                   NNZ          M          K    |    TIME        GFLOPS      ERR    |   TEST  |\n");
  printf("+------------------------------------------------------------------------------------------------------------\n");

  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_type != DT_REG) continue; //Only regular files

    sprintf(matrix_name, "%s/%s", directory, entry->d_name);

    //----------------------------------------------------------------------------
    //SPMV GINKGO
    //----------------------------------------------------------------------------
    /*
    */
    //----------------------------------------------------------------------------
    //----------------------------------------------------------------------------
    
      
    //----------------------------------------------------------------------------
    //SPMV PETSc
    //----------------------------------------------------------------------------
    #ifdef LIB_PETSC
    PetscInitialize(&argc, &argv, NULL, NULL);
    Mat A;
    MatInfo info;
    PetscViewer viewer;

    PetscCall(MatCreateFromMTX(&A, matrix_name, PETSC_TRUE));

    Vec b, c;
    MatGetSize(A, &nrows, &ncols);
    MatGetInfo(A, MAT_GLOBAL_SUM, &info);
    mnz = (PetscInt)info.nz_allocated; // Number of nonzeros

    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, ncols);
    VecSetFromOptions(b);
    VecDuplicate(b, &c);

    PetscScalar *b_set;
    VecGetArray(b, &b_set);
    generate_vector_double(ncols, b_set);
    VecSet(c, 0.0);
    #else
    auto ref_exec = gko::ReferenceExecutor::create();
    // For GPU: auto exec = gko::CudaExecutor::create(0, ref_exec);

    auto A = gko::read<gko::matrix::Csr<ValueType, IndexType>>(
        std::ifstream(matrix_name), ref_exec
    );

    auto b = gko::matrix::Dense<ValueType>::create(
        ref_exec, gko::dim<2>{A->get_size()[1], 1}
    );
    auto c_ginkgo = gko::matrix::Dense<ValueType>::create(
        ref_exec, gko::dim<2>{A->get_size()[0], 1}
    );

    generate_vector_double(b->get_size()[0], b->get_values());
    std::fill_n(c_ginkgo->get_values(), c_ginkgo->get_size()[0], 0.0);
    mnz = A->get_num_stored_elements();
    nrows = A->get_size()[0]; 
    ncols = A->get_size()[1];
    #endif  

    time  = 0.0; 
    nreps = 0; 
    t1 = dclock();
    while (time <= tmin) {
      #ifdef LIB_PETSC
      MatMult(A, b, c); //SPMV Petsc
      #else
      A->apply(b, c_ginkgo); //SPMV Ginkgo
      #endif
      nreps++;
      t2 = dclock();
      time = (t2 > t1 ? t2 - t1: 0.0);
    }
    //----------------------------------------------------------------------------
    //----------------------------------------------------------------------------


    time   = time / nreps;
    flops  = mnz * 2.0;
    GFLOPS = flops / (1.0e+9 * time);
 
    if (test == 'T') {
      ValueType *c_baseline = (ValueType *)calloc(ncols, sizeof(ValueType));
      #ifdef LIB_PETSC
      const PetscInt *row_ptrs, *col_idx;
      const PetscScalar *values;
      PetscInt m;
      PetscBool done;
      MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &m, &row_ptrs, &col_idx, &done);
      MatSeqAIJGetArrayRead(A, &values);
      const PetscScalar *b_array, *c_array;
      VecGetArrayRead(b, &b_array); 
      VecGetArrayRead(c, &c_array); 
      spmv_baseline(row_ptrs, col_idx, values, ncols, b_array, c_baseline);
      err = gemm_validation(ncols, c_baseline, c_array);
      #else
      spmv_baseline (A->get_const_row_ptrs(), A->get_const_col_idxs(),
                     A->get_const_values(), A->get_size()[0],
                     b->get_const_values(), c_baseline);
      err = gemm_validation(A->get_size()[0], c_baseline, c_ginkgo->get_const_values());
      #endif
      free(c_baseline);	      
    } else err = 0.0; 

      printf("| %s%-25s%s %10zu  %10d %10d |  %8.2e   %s%8.2f%s     %8.2e | ", COLOR_BOLDYELLOW, matrix_name, COLOR_RESET, mnz, nrows, ncols, time, COLOR_BOLDCYAN, GFLOPS, COLOR_RESET, err);
      fprintf(fd_logs, "%s;%zu;%d;%d;%.2f;%.2e\n", matrix_name, mnz, nrows, ncols, time, GFLOPS);

      if (test == 'T')
        if (err < err_limit) printf("   %sOK%s   |\n", COLOR_BOLDGREEN,  COLOR_RESET);
        else                 printf("   %sERR%s  |\n", COLOR_BOLDRED,    COLOR_RESET);
      else                   printf("   %s--%s   |\n", COLOR_BOLDYELLOW, COLOR_RESET);


      #ifdef LIB_PETSC
      MatDestroy(&A);
      VecDestroy(&b);
      VecDestroy(&c);
      #else
      A.reset();
      b.reset();
      c_ginkgo.reset();
      #endif
    }

    printf("+------------------------------------------------------------------------------------------------------------\n\n");

    #ifdef LIB_PETSC
    PetscFinalize();
    #endif

    fclose(fd_logs);

    return 0;
}
