#include <iostream>
#include <vector>

class Matrix {
private:
    int rows;
    int cols;
    std::vector<std::vector<int>> data;

public:
    // Constructor to initialize matrix with given dimensions
    Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<int>(c, 0)) {}

    // Constructor to initialize matrix with a 2D array
    Matrix(int r, int c, const int** arr) : rows(r), cols(c), data(r, std::vector<int>(c)) {
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                data[i][j] = arr[i][j];
            }
        }
    }

    // Method to set an element in the matrix
    void setElement(int r, int c, int value) {
        if (r >= 0 && r < rows && c >= 0 && c < cols) {
            data[r][c] = value;
        } else {
            std::cerr << "Index out of bounds" << std::endl;
        }
    }

    // Method to get an element from the matrix
    int getElement(int r, int c) const {
        if (r >= 0 && r < rows && c >= 0 && c < cols) {
            return data[r][c];
        } else {
            std::cerr << "Index out of bounds" << std::endl;
            return -1; // Return an invalid value
        }
    }

    // Method to display the matrix
    void display() const {
        for (const auto& row : data) {
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        }
    }

    // Static method to multiply two matrices
    static Matrix multiply(const Matrix& mat1, const Matrix& mat2) {
        if (mat1.cols != mat2.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }

        Matrix result(mat1.rows, mat2.cols);
        for (int i = 0; i < mat1.rows; ++i) {
            for (int j = 0; j < mat2.cols; ++j) {
                for (int k = 0; k < mat1.cols; ++k) {
                    result.data[i][j] += mat1.data[i][k] * mat2.data[k][j];
                }
            }
        }
        return result;
    }

    // Static method to calculate the dot product of two matrices
    static Matrix dotProduct(const Matrix& mat1, const Matrix& mat2) {
        if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for dot product");
        }

        Matrix result(mat1.rows, mat1.cols);
        for (int i = 0; i < mat1.rows; ++i) {
            for (int j = 0; j < mat1.cols; ++j) {
                result.data[i][j] = mat1.data[i][j] * mat2.data[i][j];
            }
        }
        return result;
    }

    // Static method to perform scalar multiplication on a matrix
    static Matrix scalarMultiply(const Matrix& mat, int scalar) {
        Matrix result(mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                result.data[i][j] = mat.data[i][j] * scalar;
            }
        }
        return result;
    }

    // Static method to perform element-wise multiplication of two matrices
    static Matrix elementWiseMultiply(const Matrix& mat1, const Matrix& mat2) {
        if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication");
        }

        Matrix result(mat1.rows, mat1.cols);
        for (int i = 0; i < mat1.rows; ++i) {
            for (int j = 0; j < mat1.cols; ++j) {
                result.data[i][j] = mat1.data[i][j] * mat2.data[i][j];
            }
        }
        return result;
    }

};

int main() {
    // Initialize matrix using dimensions
    Matrix mat1(3, 3);
    mat1.setElement(0, 0, 1);
    mat1.setElement(1, 1, 2);
    mat1.setElement(2, 2, 3);
    mat1.display();

    std::cout << "----" << std::endl;

    // Initialize matrix using a 2D array
    const int rows = 3;
    const int cols = 5;
    const int arr[rows][cols] = {
        {1, 2, 3, 1, 2},
        {4, 5, 6, 1, 2},
        {7, 8, 9, 1, 2}
    };

    // Convert 2D array to array of pointers
    const int* arrPtrs[rows];
    for (int i = 0; i < rows; ++i) {
        arrPtrs[i] = arr[i];
    }

    Matrix mat2(rows, cols, arrPtrs);
    //mat2.display();

    std::cout << "----" << std::endl;

    // Initialize another matrix for multiplication
    const int arr2[cols][2] = {
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
        {9, 10}
    };

    // Convert 2D array to array of pointers
    const int* arrPtrs2[cols];
    for (int i = 0; i < cols; ++i) {
        arrPtrs2[i] = arr2[i];
    }

    Matrix mat3(cols, 2, arrPtrs2);
  //  mat3.display();

    std::cout << "----" << std::endl;

    // Multiply matrices
    Matrix result = Matrix::multiply(mat2, mat3);
   // result.display();

    std::cout << "----" << std::endl;

    // Calculate dot product of two matrices
    Matrix mat4(3, 3);
    mat4.setElement(0, 0, 1);
    mat4.setElement(1, 1, 2);
    mat4.setElement(2, 2, 3);

    mat4.display();
  //  Matrix dotProductResult = Matrix::dotProduct(mat1, mat4);
  //  dotProductResult.display();

    std::cout << "----" << std::endl;

    // Perform scalar multiplication on a matrix
    int scalar = 2;
 //   Matrix scalarMultiplicationResult = Matrix::scalarMultiply(mat1, scalar);
 //   scalarMultiplicationResult.display();

    std::cout << "----" << std::endl;

    std::cout << "----" << std::endl;

    // Perform element-wise multiplication of two matrices
    Matrix elementWiseResult = Matrix::elementWiseMultiply(mat1, mat4);
    elementWiseResult.display();

    std::cout << "----" << std::endl;


    return 0;
}
