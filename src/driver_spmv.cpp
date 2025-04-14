#include "utils.hpp"
#include "colors.hpp"

#include "petscmat.h"
#include "mmloader.h"
#include "ginkgo/ginkgo.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <dirent.h>

#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

#define MAX_LINE 1024

using ValueType = double;  // FP64
using IndexType = int;

void spmv_baseline (const IndexType* row_ptrs, const IndexType* col_idxs, const ValueType* values,
  IndexType num_rows, const ValueType* b, ValueType* c) {
  for (IndexType row = 0; row < num_rows; ++row) {
    ValueType sum = 0.0;
    for (IndexType k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
      sum += values[k] * b[col_idxs[k]];
      //if (row < 2) printf("c[%d] = (%.5f += %.5f * %.5f) [%d - %d]\n", row, sum, values[k], b[col_idxs[k]], row_ptrs[row], row_ptrs[row + 1]);
    }
    c[row] = sum;
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



struct CSRMatrix {
    vector<double> values;     // Valores no nulos
    vector<int> columns;       // Índices de columna
    vector<int> row_ptr;       // Punteros de fila
    int rows, cols, nnz;       // Filas, columnas, elementos no nulos

    // Constructor
    CSRMatrix(int r, int c, int n) : rows(r), cols(c), nnz(n) {
        row_ptr.resize(rows + 1, 0);
   }
       // Liberar memoria
    void clear() {
        values.clear();
        columns.clear();
        row_ptr.clear();
    }
};


int main(int argc, char* argv[]) {
  char   matrix_name[MAX_LINE/2];
  char   matrix_path[MAX_LINE];
  double tmin  = 0.0;
  int    nreps = 0;
  double time  = 0.0;
  double time_ginkgo, time_petsc, GFLOPS_ginkgo, GFLOPS_petsc;
  double basetime  = 0.0;
  int    nonzerorowcnt = 0;
  char   test;
  double t1, t2, flops, baseGFLOPS, GFLOPS, err;
  char   *directory, *logs, *prefix;
  DIR    *dir;
  size_t nnz;
  struct dirent *entry;
  double err_limit = 1.0e-10;
  int    nrows, ncols;
  FILE   *fd_logs;
 
  tmin = atof(argv[1]); 
  test = argv[2][0];
  std::ifstream matrix_list(argv[3]);
  if (!matrix_list) {
    std::cerr << "Error opening file\n";
    return EXIT_FAILURE;
   }
  logs   = argv[4];
  prefix = argv[5];

  fd_logs = fopen(logs, "w");
  fprintf(fd_logs, "#Matrix_name;MNZ;M;K;Time_Ginkgo;GFLOPS_Ginkgo;Time_Petsc;GFLOPS_Petsc\n");

  printf("\n");
  printf("=====================================================\n");
  printf("|             %sSPMV DRIVER CONFIGURATION%s             |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf("=====================================================\n");
  printf("| Minimum Time (s) : %s%-30.2f%s |\n", COLOR_BOLDCYAN, tmin, COLOR_RESET);
  printf("| Test             : %s%-30s%s |\n", COLOR_BOLDCYAN, "GINKGO REFERENCE", COLOR_RESET); 
  printf("| Matrix List      : %s%-30s%s |\n", COLOR_BOLDCYAN, argv[3], COLOR_RESET); 
  printf("| Output Log       : %s%-30s%s |\n", COLOR_BOLDCYAN, logs, COLOR_RESET); 
  printf("=====================================================\n\n");

  printf("==================================================================================================================================\n");
  printf("|                                         %sSPMV DRIVER FOR PETSC AND GINKGO EVALUATION%s                                            |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf("==================================================================================================================================\n");
  printf("|                       MATRIX INFORMATION                    |         GINKGO        |          PETSc        |    VALIDATION    |\n");
  printf("+-------------------------------------------------------------+-----------------------+-----------------------+------------------+\n");
  printf("|  MATRIX NAME                   NNZ          M          K    |    TIME      GFLOPS   |   TIME       GFLOPS   |   ERR      TEST  |\n");
  printf("+-------------------------------------------------------------+-----------------------+-----------------------+------------------+\n");

  while (matrix_list.getline(matrix_name, MAX_LINE)) {
    if (matrix_name[0] == '%') continue;

    sprintf(matrix_path, "%s/%s", prefix, matrix_name);
    nonzerorowcnt = 0;

    //----------------------------------------------------------------------------
    //Reading CSR with Ginkgo
    //----------------------------------------------------------------------------
    auto ref_exec = gko::ReferenceExecutor::create();
    // For GPU: auto exec = gko::CudaExecutor::create(0, ref_exec);

    auto A_ginkgo = gko::read<gko::matrix::Csr<ValueType, IndexType>>(
        std::ifstream(matrix_path), ref_exec);

    auto b_ginkgo = gko::matrix::Dense<ValueType>::create(
        ref_exec, gko::dim<2>{A_ginkgo->get_size()[1], 1});

    auto c_ginkgo = gko::matrix::Dense<ValueType>::create(
        ref_exec, gko::dim<2>{A_ginkgo->get_size()[0], 1});

    generate_vector_double(b_ginkgo->get_size()[0], b_ginkgo->get_values());

    std::fill_n(c_ginkgo->get_values(), c_ginkgo->get_size()[0], 0.0);

    nrows = A_ginkgo->get_size()[0]; 
    ncols = A_ginkgo->get_size()[1];
    nnz   = A_ginkgo->get_num_stored_elements();
    
    const auto* row_ptrs = A_ginkgo->get_const_row_ptrs();
    const auto* col_idxs = A_ginkgo->get_const_col_idxs();
    const auto* values   = A_ginkgo->get_const_values();
    //----------------------------------------------------------------------------
    //----------------------------------------------------------------------------

    //SPMV PETSc
    PetscInitialize(&argc, &argv, NULL, NULL);
    MatInfo info;
    Mat A_petsc;
    Vec b_petsc, c_petsc;
    PetscScalar *b_set;

    //With PETSC_TRUE, Petsc only calculates one of the two symetric parts of the matrix
    PetscCall(MatCreateFromMTX(&A_petsc, matrix_path, PETSC_FALSE));

    //MatCreate(PETSC_COMM_SELF, &A_petsc);
    //MatSetSizes(A_petsc, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols);
    //MatSetType(A_petsc, MATSEQAIJ);
    //MatSeqAIJSetPreallocation(A_petsc, 0, nnz_per_row.data());
    //for (PetscInt i = 0; i < nrows; ++i) {
        //PetscInt ncols = row_ptrs[i+1] - row_ptrs[i];
        //if (ncols > 0)
          //MatSetValues(A_petsc, 1, &i, ncols,
                        //const_cast<PetscInt*>(col_idxs + row_ptrs[i]),
                        //const_cast<PetscScalar*>(values + row_ptrs[i]),
                        //INSERT_VALUES);
    //}
    //MatAssemblyBegin(A_petsc, MAT_FINAL_ASSEMBLY);
    //MatAssemblyEnd(A_petsc, MAT_FINAL_ASSEMBLY);

    VecCreate(PETSC_COMM_WORLD, &b_petsc);
    VecSetSizes(b_petsc, PETSC_DECIDE, ncols);
    VecSetFromOptions(b_petsc);
    VecDuplicate(b_petsc, &c_petsc);
    VecSet(c_petsc, 0.0);

    VecGetArray(b_petsc, &b_set);
    for (int i = 0; i < ncols; i++) b_set[i] = b_ginkgo->get_values()[i];

    std::vector<PetscInt> nnz_per_row(nrows);
    for (PetscInt i = 0; i < nrows; ++i) {
        nnz_per_row[i] = row_ptrs[i+1] - row_ptrs[i];
    }


    flops = 2.0 * nnz;

    time = 0.0; 
    nreps = 0; 
    t1    = dclock();
    while (time <= tmin) {
      A_ginkgo->apply(b_ginkgo, c_ginkgo); //SPMV Ginkgo
      nreps++;
      t2 = dclock();
      time = (t2 > t1 ? t2 - t1: 0.0);
    }
    time_ginkgo   = time / nreps;
    GFLOPS_ginkgo = flops / (1.0e+9 * time_ginkgo);


    time  = 0.0; 
    nreps = 0; 
    t1    = dclock();
    while (time <= tmin) {
      MatMult(A_petsc, b_petsc, c_petsc); //SPMV Petsc
      nreps++;
      t2 = dclock();
      time = (t2 > t1 ? t2 - t1: 0.0);
    }
    time_petsc   = time / nreps;
    GFLOPS_petsc = flops / (1.0e+9 * time_petsc);
    
    
    const PetscScalar *c_array_petsc;
    VecGetArrayRead(c_petsc, &c_array_petsc); 
    const ValueType *c_array_ginkgo = c_ginkgo->get_const_values();
    err = gemm_validation(ncols, c_array_ginkgo, c_array_petsc);

    std::filesystem::path pathObj(matrix_name);
    std::string pname = pathObj.filename().string();

    printf("| %s%-25s%s %10zu  %10d %10d |  %s%8.2e %8.2f%s    |  %s%8.2e %8.2f%s    | %8.2e ", COLOR_BOLDYELLOW, pname.c_str(), COLOR_RESET, nnz, nrows, ncols, COLOR_BOLDCYAN, time_ginkgo, GFLOPS_ginkgo, COLOR_RESET, COLOR_BOLDMAGENTA, time_petsc, GFLOPS_petsc, COLOR_RESET, err);

    if (err < err_limit) printf("   %sOK%s   |\n", COLOR_BOLDGREEN,  COLOR_RESET);
    else                 printf("   %sERR%s  |\n", COLOR_BOLDRED,    COLOR_RESET);

    fprintf(fd_logs, "%s;%zu;%d;%d;%.2e;%.2e;%.2f;%.2f\n", pname.c_str(), nnz, nrows, ncols, time_ginkgo, GFLOPS_ginkgo, time_petsc, GFLOPS_petsc);

    MatDestroy(&A_petsc);
    VecDestroy(&b_petsc);
    VecDestroy(&c_petsc);
      
    A_ginkgo.reset();
    b_ginkgo.reset();
    c_ginkgo.reset();
      
      
  }

  printf("+-------------------------------------------------------------+-----------------------+-----------------------+------------------+\n");

  PetscFinalize();
  fclose(fd_logs);

  return 0;
}
