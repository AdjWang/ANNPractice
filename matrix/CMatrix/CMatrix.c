#include <stdio.h>
#include <stdlib.h>

typedef struct CMatrix CMatrix;
struct CMatrix {
    int row;
    int column;
    double** data;
};

#define FOR_EACH(mat, expr)     do{for(int r=0; r<(mat)->row; r++){\
                                    for(int c=0; c<(mat)->column; c++){\
                                        {expr}\
                                    }\
                                }}while(0)

/*  DEPRECATED: ONLY manage memory by python.
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
*/

void print(const CMatrix* mat) {
    // printf("%d, %d, %p\n", t->a, t->b, t->data);
    printf("row: %d, column: %d\n", mat->row, mat->column);
    for(int i=0; i<mat->row; i++){
        for(int j=0; j<mat->column; j++){
            printf("%.2f\t", mat->data[i][j]);
        }
        printf("\n");
    }
}

void cmat_transpose(const CMatrix* mat, CMatrix* out){
    FOR_EACH(mat, {
        out->data[c][r] = mat->data[r][c];
    });
}

void cmat_add(const CMatrix* mat1, const CMatrix* mat2, CMatrix* out){
    FOR_EACH(mat1, {
        out->data[r][c] = mat1->data[r][c] + mat2->data[r][c];
    });
}

void cmat_sub(const CMatrix* mat1, const CMatrix* mat2, CMatrix* out){
    FOR_EACH(mat1, {
        out->data[r][c] = mat1->data[r][c] - mat2->data[r][c];
    });
}

void cmat_mul_mat(const CMatrix* mat1, const CMatrix* mat2, CMatrix* out){
    FOR_EACH(mat1, {
        out->data[r][c] = mat1->data[r][c] * mat2->data[r][c];
    });
}

void cmat_mul_val(const CMatrix* mat, double val, CMatrix* out){
    FOR_EACH(mat, {
        out->data[r][c] = mat->data[r][c] * val;
    });
}

double cmat_sum(const CMatrix* mat){
    double sum = 0.0;
    FOR_EACH(mat, {
        sum += mat->data[r][c];
    });
    return sum;
}

void cmat_dot(const CMatrix* mat1, const CMatrix* mat2, CMatrix* out){
    for(int r1=0; r1<mat1->row; r1++){
        for(int c2=0; c2<mat2->column; c2++){
            double sum = 0.0;
            for(int i=0; i<mat1->column; i++){
                sum += mat1->data[r1][i] * mat2->data[i][c2];
            }
            out->data[r1][c2] = sum;
        }
    }
}
