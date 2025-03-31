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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
void ReadMTXToCSR (char* filename, ValueType** values, IndexType** col_idx,
                   IndexType** row_ptr, int* num_rows, int* num_cols,    
                   int* num_nonzeros) {
  printf("start read\n");
  FILE* file = fopen(filename, "r");
  if (!file) {
      fprintf(stderr, "Error: No se pudo abrir %s\n", filename);
      exit(EXIT_FAILURE);
  }

  char line[1024];
  int symmetric = 0;
  int pattern = 0;

  // Leer cabecera (comentarios)
  while (fgets(line, sizeof(line), file)) {
      if (line[0] != '%') break; // Fin de comentarios
      //if (strstr(line, "symmetric")) symmetric = 1;
      //if (strstr(line, "pattern")) pattern = 1;
  }

  // Leer dimensiones: filas, columnas, no ceros
  sscanf(line, "%d %d %d", num_rows, num_cols, num_nonzeros);

  // Reservar memoria temporal para las entradas (máximo 2*nnz si es simétrica)
  int max_entries = symmetric ? 2 * (*num_nonzeros) : *num_nonzeros;
  int* temp_rows = (int*)malloc(max_entries * sizeof(int));
  int* temp_cols = (int*)malloc(max_entries * sizeof(int));
  double* temp_vals = (double*)malloc(max_entries * sizeof(double));
  int entry_count = 0;

  // Leer todas las entradas
  while (fgets(line, sizeof(line), file)) {
    if (line[0] == '\n') continue; // Saltar líneas vacías
    int row, col;
    double val = 1.0; // Valor por defecto para matrices "pattern"
    if (pattern) {
      sscanf(line, "%d %d", &row, &col);
    } else {
        sscanf(line, "%d %d %lf", &row, &col, &val);
    }
    row--; col--; // Convertir a base-0

    // Almacenar entrada
    temp_rows[entry_count] = row;
    temp_cols[entry_count] = col;
    temp_vals[entry_count] = val;
    entry_count++;

    // Si es simétrica y no está en la diagonal, añadir entrada transpuesta
    if (symmetric && row != col) {
      temp_rows[entry_count] = col;
      temp_cols[entry_count] = row;
      temp_vals[entry_count] = val;
      entry_count++;
    }
  }
  fclose(file);

  printf("read done\n");
  // Actualizar número de no ceros (por si hay simetría)
  *num_nonzeros = entry_count;

  // Ordenar entradas por fila y luego por columna (para CSR)
  for (int i = 0; i < entry_count; i++) {
    for (int j = i + 1; j < entry_count; j++) {
      if (temp_rows[i] > temp_rows[j] || 
         (temp_rows[i] == temp_rows[j] && temp_cols[i] > temp_cols[j])) {
        // Intercambiar entradas
        int tmp_row = temp_rows[i];
        int tmp_col = temp_cols[i];
        double tmp_val = temp_vals[i];
        temp_rows[i] = temp_rows[j];
        temp_cols[i] = temp_cols[j];
        temp_vals[i] = temp_vals[j];
        temp_rows[j] = tmp_row;
        temp_cols[j] = tmp_col;
        temp_vals[j] = tmp_val;
      }
    }
  }
  printf("order done\n");

  // Construir row_ptr (contar elementos por fila)
  *row_ptr = (int*)calloc(*num_rows + 1, sizeof(int));
  for (int i = 0; i < entry_count; i++) {
    (*row_ptr)[temp_rows[i] + 1]++;
  }

  // Acumular para obtener los punteros de fila
  for (int i = 1; i <= *num_rows; i++) {
    (*row_ptr)[i] += (*row_ptr)[i - 1];
  }

  // Construir col_idx y values
  *col_idx = (int*)malloc(entry_count * sizeof(int));
  *values = (double*)malloc(entry_count * sizeof(double));
  int* row_counter = (int*)calloc(*num_rows, sizeof(int));

  for (int i = 0; i < entry_count; i++) {
    int row = temp_rows[i];
    int pos = (*row_ptr)[row] + row_counter[row];
    (*col_idx)[pos] = temp_cols[i];
    (*values)[pos] = temp_vals[i];
    row_counter[row]++;
  }

  // Liberar memoria temporal
  free(temp_rows);
  free(temp_cols);
  free(temp_vals);
  free(row_counter);
  
  printf("finish read\n");
}
*/

void ReadMTXToCSR(
    const char* filename,
    double** values,
    int** col_ind,
    int** row_ptr,
    int* num_rows,
    int* num_cols,
    int* num_nonzeros
) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error al abrir %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[1024];
    int symmetric = 0;

    // Leer cabecera
    while (fgets(line, sizeof(line), file)) {
        if (line[0] != '%') break;
        if (strstr(line, "symmetric")) symmetric = 1;
    }

    // Leer dimensiones
    sscanf(line, "%d %d %d", num_rows, num_cols, num_nonzeros);

    // Reservar memoria (sobrestimada si hay simetría)
    int nnz = symmetric ? 2 * (*num_nonzeros) : *num_nonzeros;
    *values = (double*)malloc(nnz * sizeof(double));
    *col_ind = (int*)malloc(nnz * sizeof(int));
    *row_ptr = (int*)calloc(*num_rows + 1, sizeof(int));

    // Contar elementos por fila y llenar valores/columnas
    int entry_count = 0;
    while (fgets(line, sizeof(line), file)) {
        int row, col;
        double val;
        sscanf(line, "%d %d %lf", &row, &col, &val);
        row--; col--;  // Base-1 → Base-0

        // Almacenar entrada directamente (asumiendo que el archivo está ordenado)
        (*values)[entry_count] = val;
        (*col_ind)[entry_count] = col;
        (*row_ptr)[row + 1]++;  // Contar elementos por fila
        entry_count++;

        // Si es simétrica, añadir entrada transpuesta
        if (symmetric && row != col) {
            (*values)[entry_count] = val;
            (*col_ind)[entry_count] = row;
            (*row_ptr)[col + 1]++;
            entry_count++;
        }
    }
    fclose(file);

    // Calcular row_ptr acumulativo
    for (int i = 1; i <= *num_rows; i++) {
        (*row_ptr)[i] += (*row_ptr)[i - 1];
    }
}


void freeCSR(ValueType *values, IndexType *col_idx, IndexType *row_ptr ) {
  if (values  != NULL) free(values);
  if (col_idx != NULL) free(col_idx);
  if (row_ptr != NULL) free(row_ptr);
}

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



int main(int argc, char* argv[]) {
  char   matrix_name[512];
  double tmin  = 0.0;
  int    nreps = 0;
  double time  = 0.0;
  double basetime  = 0.0;
  char   test;
  double t1, t2, flops, baseGFLOPS, GFLOPS, err;
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

  printf("==================================================================================================================================\n");
  printf("|                                                        %sSPMV DRIVER%s                                                             |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf("==================================================================================================================================\n");
  printf("|                       MATRIX INFORMATION                    |               TARGET              |      REFERENCE      |        |\n");
  printf("+-------------------------------------------------------------+-----------------------------------+---------------------+  TEST  |\n");
  printf("|  MATRIX NAME                   NNZ          M          K    |    TIME        GFLOPS      ERR    |  TIME       GFLOPS  |        |\n");
  printf("+-------------------------------------------------------------+-----------------------------------+---------------------+---------\n");

  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_type != DT_REG) continue; //Only regular files

    sprintf(matrix_name, "%s/%s", directory, entry->d_name);
      
    //----------------------------------------------------------------------------
    //SPMV PETSc
    //----------------------------------------------------------------------------
    #ifdef LIB_PETSC
    PetscInitialize(&argc, &argv, NULL, NULL);
    Mat A;
    MatInfo info;
    PetscViewer viewer;

    PetscCall(MatCreateFromMTX(&A, matrix_name, PETSC_FALSE));

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
      ValueType *base_values;
      IndexType *base_row_ptr, *base_col_idx;
      int base_nrows, base_ncols, base_nnz;
      ReadMTXToCSR (matrix_name, &base_values, &base_col_idx, &base_row_ptr, &base_nrows, &base_ncols, &base_nnz);
      
      ValueType *base_c = (ValueType *)calloc(base_ncols, sizeof(ValueType));

      #ifdef LIB_PETSC
      const PetscInt *row_ptrs, *col_idx;
      const PetscScalar *values;
      PetscInt m;
      PetscBool done;
      MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &m, &row_ptrs, &col_idx, &done);
      MatSeqAIJGetArrayRead(A, &values);
      
      const PetscScalar *c_array, *b_array;
      VecGetArrayRead(b, &b_array); 
      VecGetArrayRead(c, &c_array); 

      //for (int x = 0; x < 20; x++) printf("PETSC[%d]  : %d %d %.10f\n", x, row_ptrs[x], col_idx[x], values[x]);
      //for (int x = 0; x < 20; x++) printf("C[%d]      : %.10f\n", x, c_array[x]);

      //for (int x = 0; x < 10; x++) printf("BASELINE: %d %d %.10f\n", base_row_ptr[x],base_col_idx[x],base_values[x]);
      //spmv_baseline(row_ptrs, col_idx, values, ncols, b_array, c_baseline);
      //err = gemm_validation(ncols, c_baseline, c_array);
      
      //-------------------------------------------------------------------------
      //Alignment issue?
      //-------------------------------------------------------------------------
      //Remove this for a normal execution
      //ValueType *values_m   = (ValueType *)malloc( mnz   * sizeof(ValueType));
      //IndexType *row_ptrs_m = (IndexType *)malloc( ncols * sizeof(IndexType));
      //IndexType *col_idx_m  = (IndexType *)malloc( mnz * sizeof(IndexType));
      //for (int i=0; i < mnz; i++)   values_m[i]   = values[i];
      //for (int i=0; i < ncols; i++) row_ptrs_m[i] = row_ptrs[i];
      //for (int i=0; i < mnz; i++) col_idx_m[i]  = col_idx[i];
      //values   = values_m;
      //row_ptrs = row_ptrs_m;
      //col_idx  = col_idx_m;
      //-------------------------------------------------------------------------
      #else
      //const ValueType   *values = A->get_const_values();
      const ValueType  *c_array = c_ginkgo->get_const_values();
      const ValueType  *b_array = b->get_const_values();
      //const IndexType *row_ptrs = A->get_const_row_ptrs(); 
      //const IndexType *col_idx  = A->get_const_col_idxs();
                        //ncols   = A->get_size()[0];
      //spmv_baseline (A->get_const_row_ptrs(), A->get_const_col_idxs(),
                     //A->get_const_values(), A->get_size()[0],
                     //b->get_const_values(), c_baseline);
      //err = gemm_validation(A->get_size()[0], c_baseline, c_ginkgo->get_const_values());

      //for (int x = 0; x < 20; x++) printf("GINKGO[%d] : %d %d %.10f\n", x, A->get_const_row_ptrs()[x], A->get_const_col_idxs()[x], A->get_const_values()[x]);
      //for (int x = 0; x < 20; x++) printf("C[%d]      : %.10f\n", x, c_array[x]);
      #endif
    
      //Evaluate Baseline
      basetime  = 0.0; 
      nreps = 0; 
      t1 = dclock();
      while (basetime <= tmin) {
        spmv_baseline(base_row_ptr, base_col_idx, base_values, base_ncols, b_array, base_c);
        nreps++;
        t2 = dclock();
        basetime = (t2 > t1 ? t2 - t1: 0.0);
      }

      basetime   = basetime / nreps;
      baseGFLOPS = flops / (1.0e+9 * basetime);
      err = gemm_validation(base_ncols, base_c, c_array);
      free(base_c);
        
      //-------------------------------------------------------------------------
      //Alignment issue?
      //-------------------------------------------------------------------------
      //Remove this for a normal execution
      //free(values_m);
      //free(row_ptrs_m);
      //free(col_idx_m);
      //-------------------------------------------------------------------------

    } else err = 0.0; 

      printf("| %s%-25s%s %10zu  %10d %10d |  %8.2e   %s%8.2f%s     %8.2e | %8.2e %s%8.2f%s   |", COLOR_BOLDYELLOW, matrix_name, COLOR_RESET, mnz, nrows, ncols, time, COLOR_BOLDCYAN, GFLOPS, COLOR_RESET, err, basetime, COLOR_BOLDRED, baseGFLOPS, COLOR_RESET);

      if (test == 'T')
        if (err < err_limit) printf("   %sOK%s   |\n", COLOR_BOLDGREEN,  COLOR_RESET);
        else                 printf("   %sERR%s  |\n", COLOR_BOLDRED,    COLOR_RESET);
      else                   printf("   %s--%s   |\n", COLOR_BOLDYELLOW, COLOR_RESET);

      fprintf(fd_logs, "%s;%zu;%d;%d;%.2e;%.2f;%.2e;%.2f\n", matrix_name, mnz, nrows, ncols, time, GFLOPS, basetime, baseGFLOPS);

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

    printf("+-------------------------------------------------------------+-----------------------------------+---------------------+---------\n");

    #ifdef LIB_PETSC
    PetscFinalize();
    #endif

    fclose(fd_logs);

    return 0;
}
