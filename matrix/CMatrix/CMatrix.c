#include <stdio.h>
#include <stdlib.h>

typedef struct CMatrix CMatrix;
struct CMatrix {
    int row;
    int column;
    double** data;
};

CMatrix* cmat_constructor(int row, int column, double** _data) {
    CMatrix* cmat_this = (CMatrix*)malloc(sizeof(CMatrix));
    if(cmat_this == NULL){
        return NULL;
    }
    cmat_this->row = row;
    cmat_this->column = column;
    cmat_this->data = _data;
    return cmat_this;
}

void cmat_destructor(CMatrix* cmat_this) {
    free(cmat_this->data);
    free(cmat_this);
}

CMatrix* test(CMatrix* mat) {
    // printf("%d, %d, %p\n", t->a, t->b, t->data);
    printf("row: %d, column: %d\n", mat->row, mat->column);
    for(int i=0; i<mat->row; i++){
        for(int j=0; j<mat->column; j++){
            printf("%.2f\t", mat->data[i][j]);
        }
        printf("\n");
    }
    mat->data[0][0] = 0.49;
    return mat;
}