#include "utils.hpp"
#include "colors.hpp"

#ifdef LIB_PETSC
  #include "petscmat.h"
  #include "mmloader.h"
#else
  #include "ginkgo/ginkgo.hpp"
#endif

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


// Función para leer archivo .mtx y construir CSR
CSRMatrix readMTXToCSR(char *filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo: " << filename << endl;
        exit(1);
    }

    string line;
    // Leer encabezado
    while (getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Leer dimensiones
    int rows, cols, nnz;
    istringstream iss(line);
    iss >> rows >> cols >> nnz;
    printf("Read rows=%d, cols=%d, nnz=%d\n", rows, cols, nnz);

    // Crear matriz temporal para almacenar todos los elementos (incluyendo simétricos)
    vector<vector<double>> temp_matrix(rows, vector<double>(cols, 0.0));
    int actual_nnz = 0;

    printf("Reading data\n");
    // Leer datos
    for (int i = 0; i < nnz; ++i) {
        getline(file, line);
        if (line.empty()) continue;

        int row, col;
        double value;
        istringstream iss(line);
        iss >> row >> col >> value;

        // Convertir a índices basados en 0
        row--;
        col--;

        // Almacenar elemento
        if (temp_matrix[row][col] == 0) actual_nnz++;
        temp_matrix[row][col] = value;

        // Si es simétrica, almacenar también el elemento transpuesto
        if (row != col) {
            if (temp_matrix[col][row] == 0) actual_nnz++;
            temp_matrix[col][row] = value;
        }
    }

    file.close();
    printf("Read ok\n");
    // Construir CSR
    CSRMatrix csr(rows, cols, actual_nnz);
    csr.row_ptr[0] = 0;

    int count = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (temp_matrix[i][j] != 0) {
                csr.values.push_back(temp_matrix[i][j]);
                csr.columns.push_back(j);
                count++;
            }
        }
        csr.row_ptr[i+1] = count;
    }

    return csr;
}

// Función para crear matriz PETSc MATSEQAIJ a partir de CSR
Mat createPETScMatrixFromCSR(CSRMatrix& csr) {
    Mat A;
    MatCreate(PETSC_COMM_SELF, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, csr.rows, csr.cols);
    MatSetType(A, MATSEQAIJ);
    
    // Pre-asignación de memoria
    vector<PetscInt> nnz_per_row(csr.rows);
    for (int i = 0; i < csr.rows; ++i) {
        nnz_per_row[i] = csr.row_ptr[i+1] - csr.row_ptr[i];
    }
    MatSeqAIJSetPreallocation(A, 0, nnz_per_row.data());
    
    // Insertar valores
    for (int i = 0; i < csr.rows; ++i) {
        int start = csr.row_ptr[i];
        int end = csr.row_ptr[i+1];
        int ncols = end - start;
        
        if (ncols > 0) {
            MatSetValues(A, 1, &i, ncols, 
                        csr.columns.data() + start, 
                        csr.values.data() + start, 
                        INSERT_VALUES);
        }
    }
    
    // Ensamblar matriz
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    // Liberar memoria de CSR
    csr.clear();
    
    return A;
}


// Función para imprimir la matriz CSR
void printCSR(const CSRMatrix& csr) {
    cout << "Valores: ";
    for (double val : csr.values) cout << val << " ";
    cout << endl;

    cout << "Columnas: ";
    for (int col : csr.columns) cout << col << " ";
    cout << endl;

    cout << "Punteros de fila: ";
    for (int ptr : csr.row_ptr) cout << ptr << " ";
    cout << endl;
}


int main(int argc, char* argv[]) {
  char   matrix_name[MAX_LINE/2];
  char   matrix_path[MAX_LINE];
  double tmin  = 0.0;
  int    nreps = 0;
  double time  = 0.0;
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
  fprintf(fd_logs, "#Matrix_name;MNZ;M;K;Time;GFlops\n");

  printf("\n");
  printf("=====================================================\n");
  printf("|             %sSPMV DRIVER CONFIGURATION%s             |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf("=====================================================\n");
  printf("| Minimum Time (s) : %s%-30.2f%s |\n", COLOR_BOLDCYAN, tmin, COLOR_RESET);
  printf("| Test             : %s%-30c%s |\n", COLOR_BOLDCYAN, test, COLOR_RESET); 
  printf("| Matrix List      : %s%-30s%s |\n", COLOR_BOLDCYAN, argv[3], COLOR_RESET); 
  printf("| Output Log       : %s%-30s%s |\n", COLOR_BOLDCYAN, logs, COLOR_RESET); 
  #ifdef LIB_PETSC
  printf("| Library selected : %s%-30s%s |\n", COLOR_BOLDCYAN, "PETSc", COLOR_RESET); 
  #else
  printf("| Library selected : %s%-30s%s |\n", COLOR_BOLDCYAN, "Ginkgo", COLOR_RESET); 
  #endif
  printf("=====================================================\n\n");

  printf("============================================================================================================\n");
  printf("|                                                   %sSPMV DRIVER%s                                            |\n", COLOR_BOLDYELLOW, COLOR_RESET);
  printf("============================================================================================================\n");
  printf("|                       MATRIX INFORMATION                    |               TARGET              |        |\n");
  printf("+-------------------------------------------------------------+-----------------------------------+  TEST  |\n");
  printf("|  MATRIX NAME                   NNZ          M          K    |    TIME        GFLOPS      ERR    |        |\n");
  printf("+-------------------------------------------------------------+-----------------------------------+--------+\n");

  while (matrix_list.getline(matrix_name, MAX_LINE)) {
    //while ((entry = readdir(dir)) != NULL) {
    //if (entry->d_type != DT_REG) continue; //Only regular files
    if (matrix_name[0] == '%') continue;
    sprintf(matrix_path, "%s/%s", prefix, matrix_name);
    nonzerorowcnt = 0;
    //----------------------------------------------------------------------------
    //SPMV PETSc
    //----------------------------------------------------------------------------
    #ifdef LIB_PETSC
    PetscInitialize(&argc, &argv, NULL, NULL);
    MatInfo info;
    PetscViewer viewer;
    Mat A;

    //CSRMatrix csr = readMTXToCSR(matrix_path);
    //printf("Read csr done\n");
    //Mat A = createPETScMatrixFromCSR(csr);
    PetscCall(MatCreateFromMTX(&A, matrix_path, PETSC_FALSE));

    Vec b, c;
    MatGetSize(A, &nrows, &ncols);
    MatGetInfo(A, MAT_GLOBAL_SUM, &info);


    nnz = (PetscInt)info.nz_allocated; // Number of nonzeros
    nnz = (PetscInt)info.nz_used; // Number of nonzeros

    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, ncols);
    VecSetFromOptions(b);
    VecDuplicate(b, &c);

    PetscScalar *b_set;
    VecGetArray(b, &b_set);
    generate_vector_double(ncols, b_set);
    VecSet(c, 0.0);

    const PetscInt *row_ptrs, *col_idx;
    PetscInt m, rownz; 
    PetscBool done;
    MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &m, &row_ptrs, &col_idx, &done);
    for (int i = 0; i < m; i++)  {
      if (row_ptrs[i+1] > row_ptrs[i]) nonzerorowcnt++;
    }
    nrows = m;
    #else
    auto ref_exec = gko::ReferenceExecutor::create();
    // For GPU: auto exec = gko::CudaExecutor::create(0, ref_exec);

    auto A = gko::read<gko::matrix::Csr<ValueType, IndexType>>(
        std::ifstream(matrix_path), ref_exec
    );

    auto b = gko::matrix::Dense<ValueType>::create(
        ref_exec, gko::dim<2>{A->get_size()[1], 1}
    );
    auto c_ginkgo = gko::matrix::Dense<ValueType>::create(
        ref_exec, gko::dim<2>{A->get_size()[0], 1}
    );

    generate_vector_double(b->get_size()[0], b->get_values());
    std::fill_n(c_ginkgo->get_values(), c_ginkgo->get_size()[0], 0.0);
    nnz = A->get_num_stored_elements();
    nrows = A->get_size()[0]; 
    ncols = A->get_size()[1];
    const auto& row_ptrs = A->get_row_ptrs();
    for (int i = 0; i < nrows; i++)  {
      if (row_ptrs[i+1] > row_ptrs[i]) nonzerorowcnt++;
    }
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
    flops = 2.0 * nnz - nonzerorowcnt;
    GFLOPS = flops / (1.0e+9 * time);

    //const ValueType  *c_array = c_ginkgo->get_const_values(); //Ginkgo
    
    #ifdef LIB_PETSC
    const PetscScalar *c_array;
    VecGetArrayRead(c, &c_array); 
    #else
    const ValueType *c_array = c_ginkgo->get_const_values();	    
    #endif
    for (int i = 0; i < 32; i++) printf("%.8f, \n", c_array[i]);

    /* 
    if (test == 'T') {
      ValueType *base_values;
      IndexType *base_row_ptr, *base_col_idx;
      int base_nrows, base_ncols, base_nnz;
      
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

      #else
      //const ValueType   *values = A->get_const_values();
      const ValueType  *b_array = b->get_const_values();
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

    } else*/ 
    err = 0.0; 
    
    std::filesystem::path pathObj(matrix_name);
    std::string pname = pathObj.filename().string();

      printf("| %s%-25s%s %10zu  %10d %10d |  %8.2e   %s%8.2f%s     %8.2e |", COLOR_BOLDYELLOW, pname.c_str(), COLOR_RESET, nnz, nrows, ncols, time, COLOR_BOLDCYAN, GFLOPS, COLOR_RESET, err);

      if (test == 'T')
        if (err < err_limit) printf("   %sOK%s   |\n", COLOR_BOLDGREEN,  COLOR_RESET);
        else                 printf("   %sERR%s  |\n", COLOR_BOLDRED,    COLOR_RESET);
      else                   printf("   %s--%s   |\n", COLOR_BOLDYELLOW, COLOR_RESET);

      fprintf(fd_logs, "%s;%zu;%d;%d;%.2e;%.2f;%.2e;%.2f\n", pname.c_str(), nnz, nrows, ncols, time, GFLOPS, basetime, baseGFLOPS);

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

    printf("+-------------------------------------------------------------+-----------------------------------+--------+\n");

    #ifdef LIB_PETSC
    PetscFinalize();
    #endif

    fclose(fd_logs);

    return 0;
}
