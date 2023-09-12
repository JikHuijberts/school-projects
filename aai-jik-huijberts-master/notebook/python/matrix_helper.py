from copy import copy, deepcopy


class MatrixHelper:
    # Multipy two matricies
    def matMul(self, A, B):
        return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

    def idMat(self, size):
        return [[1 if row == column else 0 for column in range(size)] for row in range(size)]

    def invMat(self, A):
        # Step 1 get the determinant of the matrix
        determinant = self.getMatrixDeternminant(A)
        # check of the length is 2 for a quick solution
        if len(A) == 2:
            return
        inverse = []
        for row in range(len(A)):
            inverseRow = []
            for col in range(len(A)):
                minor = self.getMatrixMinor(A, row, col)
                inverseRow.append(((-1) ** (row + col)) * self.getMatrixDeternminant(minor))
            inverse.append(inverseRow)
        inverse = self.transpose(inverse)
        for row in range(len(inverse)):
            for col in range(len(inverse)):
                inverse[row][col] = inverse[row][col] / determinant
        return inverse

    def getMatrixMinor(self, A, i, j):
        return [row[:j] + row[j + 1:] for row in (A[:i] + A[i + 1:])]

    def getMatrixDeternminant(self, A):
        # A quick solution for a 2 length matrix
        if len(A) == 2:
            return A[0][0] * A[1][1] - A[0][1] * A[1][0]

        determinant = 0
        # loop trough all the columns
        for col in range(len(A)):
            determinant += ((-1) ** col) * A[0][col] * self.getMatrixDeternminant(
                self.getMatrixMinor(A, 0, col))
        return determinant

    def transpose(__self__, A):
        return [list(i) for i in zip(*A)]
