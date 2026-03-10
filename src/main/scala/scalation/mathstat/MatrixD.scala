
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Jun 17 19:29:23 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Matrix Data Structure of Doubles
 */

package scalation
package mathstat

import java.util.Arrays.copyOf
import java.io.PrintWriter

import scala.annotation.unused
import scala.collection.immutable.{IndexedSeq => IIndexedSeq, Set => ISet}
import scala.collection.mutable.{ArrayBuffer, IndexedSeq, Set}
import scala.math.{min, round}
import scala.util.control.Breaks.{break, breakable}

/** Top-level type definition for functions mapping from `MatrixD`:
 */
type FunctionM2V = MatrixD => VectorD                          // matrix `MatrixD` to vector `VectorD`
type FunctionM2M = MatrixD => MatrixD                          // matrix `MatrixD` to matrix `MatrixD`

/** Top-level type definition for functions mapping from `VectorD`:
 */
type FunctionV2M  = VectorD => MatrixD                         // matrix `MatrixD` to vector `VectorD`
type FunctionV2MV = VectorD => (MatrixD, VectorD)              // vector `VectorD` to (matrix, vector) (`MatrixD`, `VectorD`)
type FunctionVV2M = (VectorD, VectorD) => MatrixD              // (vector, vector) (`VectorD`, VectorD`) to `MatrixD`


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Matricize a vector function (V2V) by applying it to each row of a matrix.
 *  MatrixD (for i <- x.indices yield f(x(i)))
 *  @param f  the vector function to matricize
 *  @param x  the matrix to apply the function to
 */
def matricize (f: FunctionV2V)(x: MatrixD): MatrixD = MatrixD (x.indices.map { i => f(x(i)) })


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Matrixize a vector function (V2V) to create a matrix function (M2M).
 *  @param f  the vector function to matrixize
 */
def matrixize (f: FunctionV2V): FunctionM2M = (x: MatrixD) => x.mmap (f(_))


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Diagnose matrix x looking for high correlation, high condition number,
 *  lower than expected rank, zero variance columns (there should only be one).
 *  @param x  the data matrix to diagnose
 */
def diagnoseMat (x: MatrixD): Unit =
    banner ("diagnoseMat: Matrix Dimensions")
    println (s"x.dim = ${x.dim}, x.dim2 = ${x.dim2}")

    banner ("Correlation Matrix")
    println (s"x.corr = ${x.corr}")

//  banner ("Matrix Condition Number")
//  println (s"x.conditionNum = ${x.conditionNum}")            // FIX - better ways to calculate

    banner ("Matrix Rank")
    val fac = new Fac_QR_RR (x).factor ()                      // use Rank Revealing QR Factorization
    println (s"fac.rank = ${fac.rank}")

    banner ("Variance of Matrix Columns")
    cfor (0, x.dim2) { j => println (s"x(?, $j).variance = ${x(?, j).variance}") }
end diagnoseMat


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MatrixD` class stores and operates on Numeric Matrices of base type `Double`.
 *  the 'cfor' loop is used for most loops for faster execution.
 *  @param dim   the first (row) dimension of the matrix
 *  @param dim2  the second (column) dimension of the matrix
 *  @param v     the 2D array used to store matrix elements
 */
class MatrixD (val dim:  Int,
               val dim2: Int,
               private [mathstat] var v: Array [Array [Double]] = null):

    private val debug = debugf ("MatrixD", true)               // partial invocation of debug function
    private val flaw  = flawf ("MatrixD")                      // partial invocation of flaw function

    if v == null then                                          // no array => allocate array
        v = Array.ofDim [Double] (dim, dim2)
    else                                                       // existing array => check dimensions
        val v_dim  = v.length
        val v_dim2 = if v_dim > 0 then v(0).length else dim2
        if dim != v_dim || dim2 != v_dim2 then
            flaw ("init", s"dimensions are wrong: dims = ($dim, $dim2) vs. ($v_dim, $v_dim2)")
//          throw new Exception ()
    end if

    /** The row index range
     */
    val indices  = 0 until dim

    /** The column index range
     */
    val indices2 = 0 until dim2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the row and column dimensions of this matrix.
     */
    inline def dims: (Int, Int) = (dim, dim2)

    private val minDim  = math.min (dim, dim2)                 // the minimum dimension
    private val TSZ     = 100                                  // the tile/block size (tunable)
    private val fString = "%g,\t"                              // output format spec

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a deep copy of this matrix (note: clone may not be deep).
     *  Uses Java's native `Arrays.copyOf` for efficient copying of 2D array.
     */
    def copy: MatrixD =
        val a = new Array [Array [Double]] (dim)
        var i = 0
        cfor (i < dim, i += 1) { a(i) = copyOf (v(i), dim2) }
        new MatrixD (dim, dim2, a)
    end copy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a deep copy of this matrix (check which is more efficient).
     *  @see https://stackoverflow.com/questions/1870711/deep-copy-of-2d-array-in-scala
     */
//  def copy: MatrixD = new MatrixD (dim, dim2, v.map (_.clone))

// apply, update and related methods

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the ELEMENT in row i, column j of this matrix.
     *  usage: x(3, 2)
     *  @param i  the row index
     *  @param j  the column index
     */
    def apply (i: Int, j: Int): Double = v(i)(j)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the intersection of the ROWS in range ir and COLUMNS in range jr
     *  of this matrix as a new independent matrix.
     *  usage: x(3 until 6, 2 until 4)
     *  Caveat:  Only verified for "a until b" ranges, not "a to b" ranges
     *  @param ir  the index range of rows to return
     *  @param jr  the index range of columns to return
     */
    def apply (ir: Range, jr: Range): MatrixD =
        val i1 = ir.start; val j1 = jr.start
        val a  = Array.ofDim [Double] (ir.size, jr.size)
        cfor (ir) { i =>
            val v_i = v(i); val a_i = a(i-i1)
            cfor (jr) { j => a_i(j-j1) = v_i(j) }
        } // cfor
        new MatrixD (ir.size, jr.size, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the actual i-th ROW of this matrix.
     *  usage: x(3)
     *  @param i  the row index
     */
    inline def apply (i: Int): VectorD = new VectorD (dim2, v(i))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the ROWS in range ir of this matrix as a new independent matrix.
     *  usage: x(3 until 6)
     *  @param ir  the index range of rows to return
     */
    def apply (ir: Range): MatrixD =
        val i1 = ir.start
        val a  = Array.ofDim [Array [Double]] (ir.size)
        cfor (ir) { i => a(i-i1) = copyOf (v(i), dim2) }
        new MatrixD (ir.size, dim2, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the ROWS in range ir of this matrix for column j as a vector.
     *  usage: x(3 until 6)
     *  @param ir  the index range of rows to return
     *  @param j   the column index
     */
    def apply (ir: Range, j: Int): VectorD =
        val i1 = ir.start
        val a  = Array.ofDim [Double] (ir.size)
        cfor (ir) { i => a(i-i1) = v(i)(j) }
        new VectorD (ir.size, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the COLUMNS in range jr of this matrix for row i as a vector.
     *  usage: x(2, 3 until 6)
     *  @param jr  the index range of columns to return
     *  @param i   the row index
     */
    def apply (i: Int, jr: Range): VectorD =
        val v_i = v(i)
        val j1 = jr.start
        val a  = Array.ofDim [Double] (jr.size)
        cfor (jr) { j => a(j-j1) = v_i(j) }
        new VectorD (jr.size, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the ROWS in index set iset of this matrix as a new independent matrix.
     *  usage: x(Set (3, 5, 7))
     *  @param iset  the index set of rows to return
     */
    def apply (iset: Set [Int]): MatrixD =
        val a = Array.ofDim [Array [Double]] (iset.size)
        var k = 0
        for i <- iset do { a(k) = copyOf (v(i), dim2); k += 1 }
        new MatrixD (iset.size, dim2, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the ROWS in index sequence idx of this matrix as a new independent matrix.
     *  usage: x(Array (3, 5, 7))
     *  @param idx  the index sequence of rows to return
     */
    def apply (idx: IndexedSeq [Int]): MatrixD =
        val a = Array.ofDim [Array [Double]] (idx.size)
        var k = 0
        for i <- idx do { a(k) = copyOf (v(i), dim2); k += 1 }
        new MatrixD (idx.size, dim2, a)
    end apply

    def apply (idx: IIndexedSeq [Int]): MatrixD =
        val a = Array.ofDim [Array [Double]] (idx.size)
        var k = 0
        for i <- idx do { a(k) = copyOf (v(i), dim2); k += 1 }
        new MatrixD (idx.size, dim2, a)
    end apply

    def apply (idx: Array [Int]): MatrixD =
        val a = Array.ofDim [Array [Double]] (idx.size)
        var k = 0
        for i <- idx do { a(k) = copyOf (v(i), dim2); k += 1 }
        new MatrixD (idx.size, dim2, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the j-th COLUMN of this matrix as an independent vector.
     *  usage: x(?, 2)
     *  @param all  use the all rows indicator ?
     *  @param j    the column index
     */
    inline def apply (@unused all: Char, j: Int): VectorD =
        val a = Array.ofDim [Double] (dim)
        cfor (0, dim) { i => a(i) = v(i)(j) }
        new VectorD (dim, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the COLUMNS in range jr of this matrix as a new independent matrix.
     *  usage: x(?, 2 until 4)
     *  @param all  use the all rows indicator ?
     *  @param jr   the index range of columns to return
     */
    def apply (@unused all: Char, jr: Range): MatrixD =
        val j1 = jr.start
        val a  = Array.ofDim [Double] (dim, jr.size)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (jr) { j => a_i(j-j1) = v_i(j) }
        } // cfor
        new MatrixD (dim, jr.size, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the COLUMNS in index set jset of this matrix as a new independent matrix.
     *  usage: x(?, Set (2, 4, 6))
     *  @param all   use the all rows indicator ?
     *  @param jset  the index set of columns to return
     */
    def apply (@unused all: Char, jset: Set [Int]): MatrixD =
        val a = Array.ofDim [Double] (dim, jset.size)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            var l = 0
            for j <- jset do { a_i(l) = v_i(j); l += 1 }
        } // cfor
        new MatrixD (dim, jset.size, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the COLUMNS in index sequence jdx of this matrix as a new independent matrix.
     *  usage: x(?, Array (2, 4, 6))
     *  @param all  use the all rows indicator ?
     *  @param jdx  the index set of columns to return
     */
    def apply (@unused all: Char, jdx: IndexedSeq [Int]): MatrixD =
        val a = Array.ofDim [Double] (dim, jdx.size)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            var l = 0
            for j <- jdx do { a_i(l) = v_i(j); l += 1 }
        } // cfor
        new MatrixD (dim, jdx.size, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the main DIAGONAL of this matrix as a new independent vector.
     *  usage: x(?)
     *  @param diag  use the all diagonal elements indicator ?
     */
    inline def apply (@unused diag: Char): VectorD =
        val a = Array.ofDim [Double] (minDim)
        cfor (0, minDim) { i => a(i) = v(i)(i) }
        new VectorD (minDim, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the last element (at the last row and last column) in the matrix.
     *  usage: x.last
     */
    inline def last: Double = v(dim-1)(dim2-1)
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return all but the i-th ROW of this matrix as a new independent matrix.
     *  usage: x.not(3)
     *  @param i  the row index to exclude
     */
    def not (i: Int): MatrixD =
        if i == 0 then          apply(i+1 until dim)
        else if i == dim-1 then apply(0 until i)
        else apply(0 until i) ++ apply(i+1 until dim)
    end not

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return all but the ROWS in index sequence idx of this matrix as a new
     *  independent matrix.
     *  usage: x.not(Array (3, 5, 7))
     *  @param idx  the index sequence of rows to exclude
     */
    def not (idx: IndexedSeq [Int]): MatrixD =
        val a = Array.ofDim [Array [Double]] (dim - idx.size)
        var k = 0
        cfor (0, dim) { i =>
            if ! (idx contains i) then { a(k) = copyOf (v(i), dim2); k += 1 }
        } // cfor
        new MatrixD (a.length, dim2, a)
    end not

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return all but the j-th COLUMN of this matrix as a new independent matrix..
     *  usage: x.not(?, 2)
     *  @param all  use the all rows indicator ?
     *  @param j    the column index to exclude
     */
    def not (@unused all: Char, j: Int): MatrixD =
        if j == 0 then           apply(?, j+1 until dim2)
        else if j == dim2-1 then apply(?, 0 until j)
        else apply(?, 0 until j) ++^ apply(?, j+1 until dim2)
    end not

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get column col from the matrix, returning it as a vector.
     *  @param col   the column to extract from the matrix
     *  @param from  the position to start extracting from
     */
    def col (col: Int, from: Int = 0): VectorD =
        val u = new VectorD (dim - from)
        cfor (from, dim) { i => u(i-from) = v(i)(col) }
        u
    end col

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Combine this matrix with matrix b, placing them along the diagonal and
     *  filling in the bottom left and top right regions with zeros; [this, b].
     *  @param b  the matrix to combine with this matrix
     */
    infix def diag (b: MatrixD): MatrixD =
        val m = dim + b.dim
        val n = dim2 + b.dim2
        val c = new MatrixD (m, n)

        cfor (0, m) { i => cfor (0, n) { j =>
            c.v(i)(j) = if i <  dim && j <  dim2 then v(i)(j)
                   else if i >= dim && j >= dim2 then b(i-dim, j-dim2)
                      else                            0.0
        }} // cfor
        c
    end diag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a matrix containing all but the first n rows of this matrix.
     *  @param n  the number of rows to be dropped
     */
    def drop (n: Int = 1): MatrixD = new MatrixD (dim - n, dim2, v.drop (n))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the rows from this matrix to form two matrices:  one from the rows in
     *  idx (e.g., testing set) and the other from rows not in idx (e.g., training set).
     *  Note split and split_ produce different row orders.
     *  @param idx  the set of row indices to include/exclude
     */
    def split (idx: ISet [Int]): (MatrixD, MatrixD) =
        val len = idx.size
        val a   = new MatrixD (len, dim2)
        val b   = new MatrixD (dim - len, dim2)
        var j, k = 0
        cfor (0, dim) { i =>
            if idx contains i then
                cfor (0, dim2) { l => a.v(j)(l) = v(i)(l) }
                j += 1
            else
                cfor (0, dim2) { l => b.v(k)(l) = v(i)(l) }
                k += 1
        } // cfor
        (a, b)
    end split

    inline def split (idx: IndexedSeq [Int]): (MatrixD, MatrixD) = split (idx.toSet [Int])

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the rows from this matrix to form two matrices:  one from the rows in
     *  idx (e.g., testing set) and the other from rows not in idx (e.g., training set).
     *  Concise, but less efficient than split.
     *  @param idx  the row indices to include/exclude
     */
    def split_ (idx: IndexedSeq [Int]): (MatrixD, MatrixD) = (apply(idx), not(idx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the dim-by-dim2 lower triangle of this matrix (rest are zero).
     */ 
    def lower: MatrixD =
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i => cfor (0, i) { j =>  a(i)(j) = v(i)(j) }}
        new MatrixD (dim, dim2, a)
    end lower  
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the dim2-by-dim2 upper triangle of this matrix (rest are zero).
     */ 
    def upper: MatrixD =
        val a = Array.ofDim [Double] (dim2, dim2)
        cfor (0, dim2) { i => cfor (i, dim2) { j => a(i)(j) = v(i)(j) }}
        new MatrixD (dim2, dim2, a)
    end upper

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the ELEMENT in row i, column j of this matrix.
     *  usage: x(i, j) = 5
     *  @param i  the row index
     *  @param j  the column index
     *  @param s  the scalar value to assign
     */
    inline def update (i: Int, j: Int, s: Double): Unit = v(i)(j) = s

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the elements in the i-th ROW of this matrix.
     *  usage: x(i) = u
     *  @param i  the row index
     *  @param u  the vector to assign
     */
    inline def update (i: Int, u: VectorD): Unit = v(i) = u.toArray

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the elements in the j-th COLUMN of this matrix.
     *  usage: x(?, 2) = u
     *  @param all  use the all rows indicator ?
     *  @param j    the column index
     *  @param u    the vector to assign
     */
    def update (@unused all: Char, j: Int, u: VectorD): Unit =
        cfor (0, dim) { i => v(i)(j) = u(i) }
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the elements for a given row in the COLUMNS in range jr of this matrix.
     *  usage: x(3, 2 until 4) = u
     *  @param i   use row index
     *  @param jr  the index range of columns to be updated
     *  @param u   the vector to assign
     */
    def update (i: Int, jr: Range, u: VectorD): Unit =
        val j1 = jr.start
        cfor (jr) { j => v(i)(j) = u(j-j1) }
    end update 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the main DIAGONAL of this matrix according to the given scalar.
     *  usage: x(?, ?) = 5
     *  @param d1  use the all diagonal elements indicator ?
     *  @param d2  use the all diagonal elements indicator ?
     *  @param s   the scalar value to assign
     */
    def update (@unused d1: Char, @unused d2: Char, s: Double): Unit =
        cfor (0, minDim) { i => v(i)(i) = s }
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the main DIAGONAL of this matrix according to the given vector.
     *  usage: x(?, ?) = u
     *  @param d1  use the all diagonal elements indicator ?
     *  @param d2  use the all diagonal elements indicator ?
     *  @param u   the vector to assign
     */
    def update (@unused d1: Char, @unused d2: Char, u: VectorD): Unit =
        cfor (0, minDim) { i => v(i)(i) = u(i) }
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Insert vector u into this matrix at j-th COLUMN after shifting j ... k-1 right.
     *  @param j  the start column index  [... j ... k k+1 ... ] -->
     *  @param k  the end column index    [... u j ... k+1 ... ]
     *  @param u  the vector to insert into column j 
     */
    def insert (j: Int, k: Int, u: VectorD): Unit =
        for jj <- k to j+1 by -1 do this(?, jj) = this(?, jj-1)       // shift columns right
        this(?, j) = u                                                // insert u
    end insert

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set this matrix's i-th ROW to the elements in vector u.  May have left-over
     *  elements in row unassigned.
     *  @param i  the row index
     *  @param u  the vector value to assign
     */
    def set (i: Int, u: VectorD): Unit = 
        if u.dim > dim2 then flaw ("set", "vector u is larger than the number of columns")
        cfor (0, u.dim) { j => v(i)(j) = u(j) }
    end set

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the k-th DIAGONAL of this matrix.
     *  @param k  how far above the main diagonal, e.g., (-1, 0, 1) for (sub, main, super)
     */
    def getDiag (k: Int = 0): VectorD =
        val dm = math.min (dim, dim2)
        val d  = new VectorD (dm - math.abs (k))
        val (j, l) = (math.max (-k, 0), math.min (dm-k, dm))
        cfor (j, l) { i => d(i-j) = v(i)(i+k) }
        d
    end getDiag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the k-th DIAGONAL of this matrix to the elements in vector u.
     *  @param u  the vector to set the diagonal to
     *  @param k  how far above the main diagonal, e.g., (-1, 0, 1) for (sub, main, super)
     */
    def setDiag (u: VectorD, k: Int = 0): Unit =
        val dm = math.min (dim, dim2)
        val (j, l) = (math.max (-k, 0), math.min (dm-k, dm))
        cfor (j, l) { i => v(i)(i+k) = u(i-j) }
    end setDiag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all the elements in the j-th COLUMN of this matrix to the scalar s.
     *  @param j  the column index
     *  @param s  the scalar value to assign
     */
    def setCol (j: Int, s: Double): Unit = cfor (0, dim) { i => v(i)(j) = s }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all the elements of this ENTIRE matrix to the scalar s.
     *  @param s  the scalar value to assign
     */
    def setAll (s: Double): Unit =
        cfor (0, dim) { i => cfor (0, dim2) { j => v(i)(j) = s }}
    end setAll

// Build new matrix from existing matrices

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shift the columns this matrix so the rows become diagonals, i,e., move
     *  v_ij -> v_i,(i+j).  Will produce a upper right and lower left triangles
     *  of zeros.
     */
    def shiftDiag: MatrixD =
        val a = Array.ofDim [Double] (dim + dim2 - 1, dim2)
        cfor (0, dim) { i => cfor (0, dim2) { j => a(i+j)(j) = v(i)(j) }}
        new MatrixD (dim + dim2 - 1, dim2, a)
    end shiftDiag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Unshift the columns this matrix so the diagonals become rows, i,e., move
     *  v_i,(i+j) -> v_ij.  Will lose elements in the upper right and lower left
     *  triangles.
     */
    def unshiftDiag: MatrixD =
        val a = Array.ofDim [Double] (dim - dim2 + 1, dim2)
        cfor (0, a.size) { i => cfor (0, dim2) { j => a(i)(j) = v(i+j)(j) }}
        new MatrixD (dim - dim2 + 1, dim2, a)
    end unshiftDiag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transpose this matrix (swap columns <=> rows).
     *  Note: new MatrixD (dim2, dim, v.transpose) does not work when a dimension is 0.
     */
    def transpose: MatrixD = 
        val a = Array.ofDim [Double] (dim2, dim)
        cfor (0, dim) { j =>
            val v_j = v(j)
            cfor (0, dim2) { i => a(i)(j) = v_j(i) }
        } // cfor
        new MatrixD (dim2, dim, a)
    end transpose

    inline def ᵀ: MatrixD = transpose                     // Unicode (ᵀ) symbol for transpose
//  inline def 𝐓: MatrixD = transpose                     // Unicode (𝐓) mathematical bold capital T

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate (row-wise) this matrix and matrix y (requires y to have the
     *  same column dimension as this).
     *  @param y  the other matrix
     */
    def ++ (y: MatrixD): MatrixD =
        if dim2 != y.dim2 then
            flaw ("++", s"requires same column dimensions: dim2 = $dim2 != y.dim2 = ${y.dim2}")

        new MatrixD (dim + y.dim, dim2, v ++ y.v)
    end ++

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate (column-wise) this matrix and matrix y (requires y to have the
     *  same row dimension as this).
     *  @param y  the other matrix
     */
    def ++^ (y: MatrixD): MatrixD =
        if dim != y.dim then
            flaw ("++^", s"requires same row dimensions: dim = $dim != y.dim = ${y.dim}")

        val n = dim2 + y.dim2
        val a = Array.ofDim [Double] (dim, n)
        cfor (0, dim) { i =>
            val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v(i)(j) }
            cfor (dim2, n) { j => a_i(j) = y.v(i)(j-dim2) }
        } // cfor
        new MatrixD (dim, n, a)
    end ++^

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate (row) vector u and this matrix, i.e., prepend u to this.
     *  @param u  the vector to be prepended as the new first row in new matrix
     */
    def +: (u: VectorD): MatrixD =
        if u.dim != dim2 then
            flaw ("+:", s"vector does not match row dimension: u.dim = ${u.dim} != dim2 = $dim2")

        val c = new MatrixD (dim + 1, dim2)
        cfor (0, c.dim) { i => c(i) = if i == 0 then u else apply(i-1) }
        c
    end +:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate (column) vector u and this matrix, i.e., prepend u to this.
     *  @param u  the vector to be prepended as the new first column in new matrix
     */
    def +^: (u: VectorD): MatrixD =
        if u.dim != dim then
            flaw ("+^:", s"vector does not match column dimension: u.dim = ${u.dim} != dim = $dim")

        val c = new MatrixD (dim, dim2 + 1)
        cfor (0, c.dim2) { j => c(?, j) = if j == 0 then u else apply(?, j-1) }
        c
    end +^:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate this matrix and (row) vector u, i.e., append u to this.
     *  @param u  the vector to be appended as the new last row in new matrix
     */
    def :+ (u: VectorD): MatrixD =
        if u.dim != dim2 then
            flaw (":+", s"vector does not match row dimension: u.dim = ${u.dim} != dim2 = $dim2")

        val c = new MatrixD (dim + 1, dim2)
        cfor (0, c.dim) { i => c(i) = if i < dim then apply(i) else u }
        c
    end :+

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate this matrix and (column) vector u, i.e., append u to this.
     *  @param u  the vector to be appended as the new last column in new matrix
     */
    def :^+ (u: VectorD): MatrixD =
        if u.dim != dim then
            flaw (":^+", s"vector does not match column dimension: u.dim = ${u.dim} != dim = $dim")

        val c = new MatrixD (dim, dim2 + 1)
        cfor (0, c.dim2) { j => c(?, j) = if j < dim2 then apply(?, j) else u }
        c
    end :^+

// Add (+) matrix and (matrix, vector, scalar)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this matrix and matrix y (requires y to have at least the dimensions of this).
     *  Alias rows to avoid double subscripting.
     *  @param y  the other matrix
     */
    def + (y: MatrixD): MatrixD =
        if y.dim < dim || y.dim2 < dim2 then
            flaw ("+", s"matrix + matrix - incompatible dimensions: this = $dims, y = ${y.dims}")

        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val y_i = y.v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) + y_i(j) }
        } // cfor
        new MatrixD (dim, dim2, a) 
    end +

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add (in-place) this matrix and matrix y (requires y to have at least the
     *  dimensions of this).  Alias rows to avoid double subscripting.
     *  @param y  the other matrix
     */
    def += (y: MatrixD): MatrixD =
        if y.dim < dim || y.dim2 < dim2 then
            flaw ("+", s"matrix + matrix - incompatible dimensions: this = $dims, y = ${y.dims}")

        cfor (0, dim) { i =>
            val v_i = v(i); val y_i = y.v(i)
            cfor (0, dim2) { j => v_i(j) += y_i(j) }
        } // cfor
        this
    end +=

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this matrix and (row) vector u.
     *  @param u  the vector to add
     */
    def + (u: VectorD): MatrixD =
        if u.dim < dim2 then
            flaw ("+", s"matrix + vector - incompatible dimensions: this = $dims, u = ${u.dim}")

        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) + u(j) }
        } // cfor
        new MatrixD (dim, dim2, a) 
    end +

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this matrix and (column) vector u.
     *  @param u  the vector to add
     */
    def +^ (u: VectorD): MatrixD =
        if u.dim < dim then
            flaw ("+", s"matrix + vector - incompatible dimensions: this = $dims, u = ${u.dim}")

        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) + u(i) }
        } // cfor
        new MatrixD (dim, dim2, a)
    end +^

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this matrix and scaler u.
     *  @param u  the scalar to add
     */
    def + (u: Double): MatrixD =
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) + u }
        } // cfor
        new MatrixD (dim, dim2, a)
    end +

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add (in-place) this matrix and scaler u.
     *  @param u  the scalar to add
     */
    def += (u: Double): MatrixD =
        cfor (0, dim) { i =>
            val v_i = v(i)
            cfor (0, dim2) { j => v_i(j) += u }
        } // cfor
        this
    end +=

// Subtract (-) from matrix, (matrix, vector, scalar)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the negative of this matrix (unary minus).
     */
    def unary_- : MatrixD =
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = -v_i(j) }
        } // cfor
        new MatrixD (dim, dim2, a)
    end unary_-

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract from this matrix the matrix y (requires y to have at least the dimensions of this).
     *  Alias rows to avoid double subscripting.
     *  @param y  the other matrix
     */
    def - (y: MatrixD): MatrixD =
        if y.dim < dim || y.dim2 < dim2 then
            flaw ("-", s"matrix - matrix - incompatible dimensions: this = $dims, y = ${y.dims}")

        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val y_i = y.v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) - y_i(j) }
        } // cfor
        new MatrixD (dim, dim2, a)
    end -

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract (in-place) from this matrix the matrix y (requires y to have at
     *  least the dimensions of this).  Alias rows to avoid double subscripting.
     *  @param y  the other matrix
     */
    def -= (y: MatrixD): MatrixD =
        if y.dim < dim || y.dim2 < dim2 then
            flaw ("-", s"matrix - matrix - incompatible dimensions: this = $dims, y = ${y.dims}")

        cfor (0, dim) { i =>
            val v_i = v(i); val y_i = y.v(i)
            cfor (0, dim2) { j => v_i(j) -= y_i(j) }
        } // cfor
        this
    end -=

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract from this matrix, the (row) vector u.
     *  @param u  the vector to subtract
     */
    def - (u: VectorD): MatrixD =
        if u.dim < dim2 then
            flaw ("-", s"matrix - vector - incompatible dimensions: this = $dims, u = ${u.dim}")

        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) - u(j) }
        } // cfor
        new MatrixD (dim, dim2, a)
    end -

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract from this matrix, the (column) vector u.
     *  @param u  the vector to subtract
     */
    def -^ (u: VectorD): MatrixD =
        if u.dim < dim2 then
            flaw ("-", s"matrix - vector - incompatible dimensions: this = $dims, u = ${u.dim}")

        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) - u(i) }
        } // cfor
        new MatrixD (dim, dim2, a)
    end -^

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract from this matrix, the scalar u.
     *  @param u  the scalar to subtract
     */
    def - (u: Double): MatrixD =
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) - u }
        } // cfor
        new MatrixD (dim, dim2, a)
    end -

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract (in-place) from this matrix, the scalar u.
     *  @param u  the scalar to subtract
     */
    def -= (u: Double): MatrixD =
        cfor (0, dim) { i =>
            val v_i = v(i)
            cfor (0, dim2) { j => v_i(j) -= u }
        } // cfor
        this
    end -=

// Multiply element-wise (*~) matrix and (matrix, vector)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply element-wise, this matrix and matrix y (requires y to have at least
     *  the dimensions of this).  Alias rows to avoid double subscripting.
     *  Also known as Hadamard product.
     *  @param y  the other matrix
     */
    def *~ (y: MatrixD): MatrixD =
        if y.dim < dim || y.dim2 < dim2 then
            flaw ("*~", s"matrix *~ matrix - incompatible dimensions: this = $dims, y = ${y.dims}")

        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val y_i = y.v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) * y_i(j) }
        } // cfor
        new MatrixD (dim, dim2, a)
    end *~

    inline def ⊙ (y: MatrixD): MatrixD = *~ (y)                     // Unicode XNOR gate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this matrix by vector u to produce another matrix v_ij * u_j.
     *  E.g., multiply a matrix by a diagonal matrix represented as a vector.
     *  @param u  the vector to multiply by
     */
    def *~ (u: VectorD): MatrixD =
        val dm = math.min (dim2, u.dim)
        val a  = Array.ofDim [Double] (dim, dm)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dm) { j => a_i(j) = v_i(j) * u(j) }
        } // cfor
        new MatrixD (dim, dm, a)
    end *~

    inline def ⊙ (y: VectorD): MatrixD = *~ (y)                     // Unicode XNOR gate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply vector u by this matrix to produce another matrix u_i * v_ij.
     *  E.g., multiply a diagonal matrix represented as a vector by a matrix.
     *  This operator is right associative (vector *~: matrix).
     *  @param u  the vector to multiply by
     */
    def *~: (u: VectorD): MatrixD =
        val dm = math.min (dim2, u.dim)
        val a  = Array.ofDim [Double] (dim, dm)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dm) { j => a_i(j) = u(i) * v_i(j) }
        } // cfor
        new MatrixD (dim, dm, a)
    end *~:

// Multiply (*) matrix and (matrix, vector, scalar)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this matrix and matrix y (requires y to have at least the dimensions of this).
     *  Alias rows to avoid double subscripting, use tiling/blocking, and optimized i, k, j loop order.
     *  @see software.intel.com/content/www/us/en/develop/documentation/advisor-cookbook/top/
     *  optimize-memory-access-patterns-using-loop-interchange-and-cache-blocking-techniques.html
     *  @param y  the other matrix
     */
    def * (y: MatrixD): MatrixD =
        if y.dim != dim2 then
            flaw ("*", s"matrix * matrix - incompatible cross dimensions: dim2 = $dim2, y.dim = ${y.dim}")

        val a = Array.ofDim [Double] (dim, y.dim2)

        cfor (0, dim, TSZ) { ii =>
            val i2 = math.min (ii + TSZ, dim)
            cfor (0, dim2, TSZ) { kk =>
                val k2 = math.min (kk + TSZ, dim2)
                cfor (0, y.dim2, TSZ) { jj =>
                    val j2 = math.min (jj + TSZ, y.dim2)

                    cfor (ii, i2) { i =>
                        val v_i = v(i); val a_i = a(i)
                        cfor (kk, k2) { k =>
                            val y_k = y.v(k); val v_ik = v_i(k)
                            cfor (jj, j2) { j => a_i(j) += v_ik * y_k(j) }
                        } // cfor
                    } // cfor

        }}} // cfor
        new MatrixD (dim, y.dim2, a)
    end *

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this matrix and vector y (requires y to have at least dim2 elements).
     *  Alias rows to avoid double subscripting and use array ops.
     *  @param y  the vector to multiply by
     */
    def * (y: VectorD): VectorD =
        if y.dim < dim2 then
            flaw ("*", s"matrix * vector - dimension of vector y: y.dim = ${y.dim} < dim2 = $dim2")

        val a = Array.ofDim [Double] (dim)
        cfor (0, dim) { i => val x_i = apply(i); a(i) = x_i dot y }
        new VectorD (dim, a)
    end *

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply (row) vector y by this matrix.  Note '*:' is right associative.
     *  vector = vector *: matrix
     *  @param y  the vector to multiply by
     */
    def *: (y: VectorD): VectorD = this.transpose * y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this matrix and scaler u.
     *  @param u  the scalar to multiply by
     */
    def * (u: Double): MatrixD =
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) * u }
        } // cfor
        new MatrixD (dim, dim2, a)
    end *

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply (in-place) this matrix and scaler u.
     *  @param u  the scalar to multiply by
     */
    def *= (u: Double): MatrixD =
        cfor (0, dim) { i =>
            val v_i = v(i)
            cfor (0, dim2) { j => v_i(j) *= u }
        } // cfor
        this
    end *=

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this matrix and matrix y (requires y to have at least the dimensions of this).
     *  Alias rows to avoid double subscripting, transpose y first and use array ops.
     *  A simpler, less efficient version of '*'.
     *  @param y  the other matrix
     */
    infix def mul (y: MatrixD): MatrixD =
        if dim2 != y.dim then
            flaw ("mul", s"matrix mul matrix - incompatible cross dimensions: dim2 = $dim2, y.dim = ${y.dim}")

        val a  = Array.ofDim [Double] (dim, y.dim2)
        val yt = y.v.transpose

        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, y.dim2) { j =>
                val y_j = yt(j)
                a_i(j) = Σ (0, dim2) { k => v_i(k) * y_j(k) }
            } // cfor
        } // cfor
        new MatrixD (dim, y.dim2, a) 
    end mul

// Divide (/) matrix by (matrix, vector, scalar)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divide element-wise, this matrix by matrix y (requires y to have at least
     *  the dimensions of this).  Alias rows to avoid double subscripting.
     *  @param y  the other matrix
     */
    def / (y: MatrixD): MatrixD =
        if y.dim < dim || y.dim2 < dim2 then
            flaw ("/", s"matrix / matrix - incompatible dimensions: this = $dims, y = ${y.dims}")

        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val y_i = y.v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) / y_i(j) }
        } // cfor
        new MatrixD (dim, dim2, a)
    end /

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divide element-wise, this matrix by (row) vector u.
     *  @param u  the vector to divide by
     */
    def / (u: VectorD): MatrixD =
        if u.dim < dim2 then
            flaw ("/", s"matrix / vector - incompatible dimensions: this = $dims, u = ${u.dim}")

        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) / u(j) }
        } // cfor
        new MatrixD (dim, dim2, a)
    end /

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divide element-wise this matrix by scaler u.
     *  @param u  the scalar to divide by
     */
    def / (u: Double): MatrixD =
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) / u }
        } // cfor
        new MatrixD (dim, dim2, a)
    end /

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divide (in-place) element-wise this matrix by scaler u.
     *  @param u  the scalar to divide by
     */
    def /= (u: Double): MatrixD =
        cfor (0, dim) { i =>
            val v_i = v(i)
            cfor (0, dim2) { j => v_i(j) /= u }
        } // cfor
        this
    end /=

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the matrix consisting of the reciprocal of each element of this matrix.
     */
    def recip: MatrixD = 
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = 1.0 / v_i(j) }
        } // cfor
        new MatrixD (dim, dim2, a)
    end recip

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the inverse of this matrix using the inverse method in the `Fac_LU` object.
     *  Note, other factorizations also compute the inverse.
     *  @see `Fac_Inv`, `Fac_Cholesky`, `Fac-QR`.
     */
    def inverse: MatrixD = Fac_LU.inverse (this)()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Raise the elements in this matrix to the p-th power (e.g., x~^2 = x *~ x)
     *  Being element-wise, x~^2 is not x * x.
     *  @param p  the scalar power (double)
     */
    def ~^ (p: Double): MatrixD =
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) ~^ p }
        } // cfor
        new MatrixD (dim, dim2, a)
    end ~^

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Raise the elements in this matrix to the p-th power (e.g., x↑2 = x *~ x)
     *  Extended to handle a negative base.
     *  @param p  the scalar power (rational number)
     */
    def ↑ (p: Rat): MatrixD =
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) ↑ p }
        } // cfor
        new MatrixD (dim, dim2, a)
    end ↑

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Raise this matrix to the p-th power (for some integer p >= 1) using
     *  a divide and conquer algorithm and matrix multiplication (x~^^2 = x * x).
     *  @param p  the power to raise this matrix to
     */
    def ~^^ (p: Int): MatrixD =
        if p < 1 then       flaw ("~^^", "power p must be an integer >= 1")
        if dim != dim2 then flaw ("~^^", "only defined on square matrices")

        if p == 2 then          this * this
        else if p == 1 then     this
        else if p % 2 == 1 then this * this ~^^ (p - 1)
        else { val c = this ~^^ (p / 2); c * c }
    end ~^^

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply each column of this matrix by its other columns to forms a matrix
     *  consisting of all 2-way cross terms [ x_i * x_j ] for j < i.
     */
    def crossAll: MatrixD =
        val n = dim2
        if n < 2 then flaw ("crossAll", s"requires at least 2 columns, but n = $n")
        val nn = n * (n - 1) / 2
        debug ("crossAll", s"create matrix with dims = ($dim, $nn)")
        val xx = new MatrixD (dim, nn)
        var k = 0
        cfor (0, dim2) { i => cfor (0, i) { j =>
            xx(?, k) = apply(?, i) * apply(?, j)
            k += 1
        }} // cfor
        xx
    end crossAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply each column of this matrix by its other two columns to forms a matrix
     *  consisting of all 3-way cross terms [ x_i * x_j * x_k ] for k < j < i.
     */
    def crossAll3: MatrixD =
        val n = dim2
        if n < 3 then flaw ("crossAll3", s"requires at least 3 columns, but n = $n")
        val nn = n * (n - 1) * (n - 2) / 6
        debug ("crossAll3", s"create matrix with dims = ($dim, $nn)")
        val xx = new MatrixD (dim, nn)
        var l = 0
        cfor (0, dim2) { i => cfor (0, i) { j => cfor (0, j) { k =>
            xx(?, l) = apply(?, i) * apply(?, j) * apply(?, k)
            l += 1
        }}} // cfor
        xx
    end crossAll3

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the dot product of this matrix and matrix y (requires y to have at
     *  least the dimensions of this).  Alias rows to avoid double subscripting,
     *  use tiling/blocking, and use array ops.
     *  @see en.wikipedia.org/wiki/Matrix_multiplication_algorithm
     *  @param y  the other matrix
     */
    infix def dot (y: MatrixD): MatrixD =
        if dim2 != y.dim then
            flaw ("dot", s"matrix dot matrix - incompatible cross dimensions: dim2 = $dim2, y.dim = ${y.dim}")

        val a = Array.ofDim [Double] (dim, y.dim)
        cfor (0, dim, TSZ) { ii =>
            cfor (0, y.dim2, TSZ) { jj =>
                cfor (0, dim2, TSZ) { kk =>
                    val k2 = math.min (kk + TSZ, dim2)

                    cfor (ii, math.min (ii + TSZ, dim)) { i =>
                        val v_i = v(i); val a_i = a(i)
                        cfor (jj, math.min (jj + TSZ, y.dim2)) { j =>
                            val y_j = y.v(j)
                            var sum = 0.0
                            cfor (kk, k2) { k => sum += v_i(k) * y_j(k) }
                            a_i(j) = sum
                        } // cfor
                    } // cfor

        }}} // cfor
        new MatrixD (dim, y.dim, a)
    end dot

    inline def ∙ (y: MatrixD): MatrixD = dot (y)                  // Unicode bullet point

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the dot product of this matrix and vector y.
     *  @param y  the vector to take the dot product with
     */
    infix def dot (y: VectorD): VectorD =
        if y.dim < dim then
            flaw ("dot", s"matrix dot vector - dimension of vector y: y.dim = ${y.dim} < dim = $dim")

        val a = Array.ofDim [Double] (dim2)
        cfor (0, dim2) { j =>
            val v_j = apply(?, j)
            var sum = 0.0
            cfor (0, dim) { i => sum += v_j(i) * y(i) }
            a(j) = sum
        } // cfor
        new VectorD (dim2, a)
    end dot

    inline def ∙ (y: VectorD): VectorD = dot (y)                  // Unicode bullet point

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'valid' (no padding) convolution of cofilter matrix c and input matrix x.
     *  Take the Hadamard product of c (this) with a slice of x and sum, then shift
     *  by one and repeat.
     *  Usage:  c conv x
     *  Caveat:  does not include reversal.
     *  @see `scalation.modeling.neuralnet.CoFilter_1D
     *  @param x  the input/data matrix
     */
    infix def conv (x: MatrixD): MatrixD =
        val y = new MatrixD (x.dim - dim + 1, x.dim2 - dim2 + 1)
        cfor (0, y.dim) { k => cfor (0, y.dim2) { l =>
            y(k, l) = (this *~ x(k until k + dim, l until l + dim2)).sum
        }} // cfor
        y
    end conv

    inline def *+ (x: MatrixD): MatrixD = conv (x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'valid' (no padding) convolution of cofilter matrix c and input matrix x.
     *  Computes the discrete convolution of cofilter matrix c and input matrix x.
     *  Usage:  c conv_ x
     *  @param x  the input/data matrix
     */
    inline infix def conv_ (x: MatrixD): MatrixD = reverse.conv (x)           // FIX - may need another reverse method

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'same' (with padding) convolution of cofilter matrix c and input matrix x.
     *  Same means that the size of the result is the same as the input.
     *  Usage:  c convs x
     *  @param x  the input/data matrix
     */
    infix def convs (x: MatrixD): MatrixD =
        val y = new MatrixD (x.dim, x.dim2)
        cfor (0, y.dim) { k => cfor (0, y.dim2) { l =>
            var sum = 0.0
            cfor (0, dim) { i => cfor (0, dim2) { j =>
                if (k-i in (0, x.dim-1)) && (l-j in (0, x.dim2-1)) then
                    sum += v(i)(j) * x(k-i, l-j)
            }} // cfor   
            y(k, l) = sum
        }} // cfor   
        y
    end convs

    inline def *~+ (x: MatrixD): MatrixD = convs (x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'full' convolution of cofilter c and input matrix x.
     *  @param x  the input/data matrix
     */
    infix def convf (x: MatrixD): MatrixD =
        val y = new MatrixD (dim + x.dim - 1, dim2 + x.dim2 - 1)
        cfor (0, y.dim) { k => cfor (0, y.dim2) { l =>
            var sum = 0.0
            cfor (0, math.min (k+1, dim)) { i => cfor (0, math.min (l+1, dim2)) { j =>
                if k-i < x.dim && l-j < x.dim2 then
                    sum += v(i)(j) * x(k-i, l-j)
            }} // cfor   
            y(k, l) = sum
        }} // cfor
        y
    end convf

    inline def *++ (x: MatrixD): MatrixD = convf (x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Flatten this matrix in row-major fashion, returning a vector containing
     *  all the elements from the matrix.
     */
    def flatten: VectorD =
        val a = Array.ofDim [Double] (dim * dim2)
        var k = 0
        cfor (0, dim) { i =>
            val v_i = v(i)
            cfor (0, dim2) { j => a(k) = v_i(j); k += 1 }
        } // cfor
        new VectorD (a.length, a)
    end flatten

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for x using back substitution (/~) in the equation
     *      u * x = y
     *  where this matrix (u) must be upper triangular.
     *  @see MatLab's / operator
     *  @param y  the constant vector
     */
    def /~ (y: VectorD): VectorD =
        val a = Array.ofDim [Double] (dim2)                    // array to hold solution
        val b = y.v                                            // y's internal array 
        for k <- dim2-1 to 0 by -1 do                          // solve for x in u*x = y
            val u_k = v(k)                                     // k-th row
            var sum = 0.0
            cfor (k+1, dim2) { j => sum += u_k(j) * a(j) }
            a(k) = (b(k) - sum) / v(k)(k)
        end for
        new VectorD (dim2, a)                                  // return vector x
    end /~

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this matrix and matrix y are nearly equal.
     *  @param y  the other matrix
     */
    def =~ (y: MatrixD): Boolean =
        if dim != y.dim || dim2 != y.dim2 then return false
        
        var close = true
        breakable {
            cfor (0, dim) { i => cfor (0, dim2) { j =>
                if ! (v(i)(j) =~ y.v(i)(j)) then { close = false; break () }
            }} // cfor
        } // breakable
        close
    end =~

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show how this matrix and matrix y differ, the first element and row that differ.
     *  @param y  the other matrix
     */
    def showDiff (y: MatrixD): Unit =
        if dim  != y.dim  then println (s"showDiff: dim = $dim != y.dim = ${y.dim}")
        if dim2 != y.dim2 then println (s"showDiff: dim2 = $dim2 != y.dim2 = ${y.dim2}")
        
        breakable {
            cfor (0, dim) { i => cfor (0, dim2) { j =>
                if ! (v(i)(j) =~ y.v(i)(j)) then
                    println (s"showDiff: v($i)($j) = ${v(i)(j)} != y.v($i)($j) = ${y.v(i)(j)}")
                    println (s"showDiff: v($i) = ${v(i)} \n y.v($i) = ${y.v(i)}")
                    break ()
            }} // cfor
        } // breakable
    end showDiff

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Iterate over this matrix row by row applying the given function.
     *  @param f  the function to apply
     */
    def foreach [U] (f: VectorD => U): Unit = { var i = 0; while i < dim do { f (this(i)); i += 1 } }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map each row of this matrix by applying function f to each row vector and
     *  returning the collected result as a vector.
     *  VectorD (for i <- indices yield f(apply(i)))
     *  @param f  the vector to scalar function to apply
     */
    def map (f: FunctionV2S): VectorD =
        VectorD (indices.map { i => f(apply(i)) })
    end map

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map each row of this matrix by applying function f to each row vector and
     *  returning the collected result as a matrix.
     *  MatrixD (for i <- indices yield f(apply(i)))
     *  @param f  the vector to vector function to apply
     */
    def mmap (f: FunctionV2V): MatrixD =
        MatrixD (indices.map { i => f(apply(i)) })
    end mmap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map each column of this matrix by applying function f to each column vector and
     *  returning the collected result as a matrix.
     *  MatrixD (for j <- indices2 yield f(apply(?, j)))
     *  @param  f the vector to vector function to apply
     */
    def mmap_ (f: FunctionV2V): MatrixD =
        MatrixD (indices2.map { j => f(apply(?, j)) }).transpose
    end mmap_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map each element of this matrix by applying function f to each element and
     *  returning the collected result as a matrix.
     *  @param f  the scalar to scalar function to apply
     */
    def map_ (f: FunctionS2S): MatrixD =
        val x = new MatrixD (dim, dim2)
        cfor (0, dim) { i => cfor (0, dim2) { j => x.v(i)(j) = f(v(i)(j)) }}
        x
    end map_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Square root transform this matrix by using math.sqrt.
     */
    def sqrt: MatrixD = map_ (math.sqrt (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Log transform this matrix by using math.log.
     */
    def log: MatrixD = map_ (math.log (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Log (1 plus) transform this matrix by using math.log1p (avoiding the log (0) problem).
     */
    def log1p: MatrixD = map_ (math.log1p (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Exp transform this matrix by using math.exp (the inverse of log).
     */
    def exp: MatrixD = map_ (math.exp (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Exp transform (minus 1) this matrix by using math.expm1 (the inverse of log1p).
     */
    def expm1: MatrixD = map_ (math.expm1 (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cos transform this matrix by using math.cos (the inverse of acos).
     */
    def cos: MatrixD = map_ (math.cos (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Inverse cos transform this matrix by using math.acos (the inverse of cos).
     */
    def acos: MatrixD = map_ (math.acos (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sin transform this matrix by using math.sin (the inverse of asin).
     */
    def sin: MatrixD = map_ (math.sin (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Inverse sin transform this matrix by using math.asin (the inverse of sin).
     */
    def asin: MatrixD = map_ (math.asin (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sum of this matrix, i.e., the sum of all its elements.
     *  Σ (indices) { i => Σ (indices2) { j => v(i)(j) }}
     */
    def sum: Double =
        var sum = 0.0
        cfor (0, dim) { i => val v_i = v(i); cfor (0, dim2) { j => sum += v_i(j) }}
        sum
    end sum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the column sums of this matrix, i.e., the sums for each of its columns.
     */
    def sumV: VectorD =
        val s = new VectorD (dim2)
        cfor (0, dim) { i => val v_i = v(i); cfor (0, dim2) { j => s(j) += v_i(j) }}
        s
    end sumV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the row sums of this matrix, i.e., the sums for each of its rows.
     */
    def sumVr: VectorD =
        val s = new VectorD (dim)
        cfor (0, dim) { i => s(i) = v(i).sum }
        s
    end sumVr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum value for the entire matrix.
     */
    def mmax: Double =
        var x = v(0).max
        cfor (1, dim) { i => val z = v(i).max; if z > x then x = z }
        x
    end mmax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum value for each column in the matrix.
     *  VectorD (for j <- indices2 yield apply(?, j).max)
     */
    def max: VectorD =
        val a = Array.ofDim [Double] (dim2)
        cfor (0, dim2) { j => a(j) = apply(?, j).max }
        new VectorD (a.size, a)
    end max

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum value for the entire matrix.
     */
    def mmin: Double =
        var x = v(0).min
        cfor (1, dim) { i => val z = v(i).min; if z < x then x = z }
        x
    end mmin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum value for each column in the matrix.
     *  VectorD (for j <- indices2 yield apply(?, j).min)
     */
    def min: VectorD = 
        val a = Array.ofDim [Double] (dim2)
        cfor (0, dim2) { j => a(j) = apply(?, j).min }
        new VectorD (a.size, a)
    end min

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum and maximum value for each column in the matrix.
     */
    def min_max: MatrixD = MatrixD (min, max)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the trace of this matrix, i.e., the sum of the elements on the
     *  main diagonal.  Should also equal the sum of the eigenvalues.
     *  Σ (indices) { i => v(i)(i) }
     *  @see Eigen.scala
     */
    def trace: Double =
        if dim != dim2 then flaw ("trace", "trace only works on square matrices")

        var sum = 0.0
        cfor (0, dim2) { i => sum += v(i)(i) }
        sum
    end trace

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 1-norm of this matrix, i.e., the maximum 1-norm of the
     *  column vectors.  This is useful for comparing matrices (a - b).norm1.
     *  (for j <- indices2 yield apply(?, j).norm1).max
     *  @see en.wikipedia.org/wiki/Matrix_norm
     */
    def norm1: Double = 
        val a = Array.ofDim [Double] (dim2)
        cfor (0, dim2) { j => a(j) = apply(?, j).norm1 }
        a.max
    end norm1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Euclidean norm (2-norm) (or its square) of each column vector
     *  in this matrix.
     */
    def normSq: VectorD = VectorD (indices2.map (apply(?, _).normSq))
    def norm: VectorD   = VectorD (indices2.map (apply(?, _).norm))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the square of the Frobenius norm of this matrix, i.e.,
     *  the sum of the squared values over all the elements (sse).
     *  Σ (indices) { i => apply(i).normSq }
     */
    def normFSq: Double =
        var sum = 0.0
        cfor (0, dim) { i => sum += apply(i).normSq }
        sum
    end normFSq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Frobenius norm of this matrix, i.e., the square root of 
     *  the sum of the squared values over all the elements (sqrt (sse)).
     *  @see en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
     */
    inline def normF: Double = math.sqrt (normFSq)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the column means of this matrix.
     *  VectorD (for j <- indices2 yield apply(?, j).mean)
     */
    def mean: VectorD = 
        val a = Array.ofDim [Double] (dim2)
        cfor (0, dim2) { j => a(j) = apply(?, j).mean }
        new VectorD (a.size, a)
    end mean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the row means of this matrix.
     *  VectorD (for i <- indices yield apply(i).mean)
     */
    def meanRow: VectorD = 
        val a = Array.ofDim [Double] (dim)
        cfor (0, dim) { i => a(i) = apply(i).mean }
        new VectorD (a.size, a)
    end meanRow

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the column variances of this matrix.
     */
    def variance: VectorD =
        val a = Array.ofDim [Double] (dim2)
        cfor (0, dim2) { j => a(j) = apply(?, j).variance }
        new VectorD (a.size, a)
    end variance 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the row variances of this matrix.
     */
    def varianceRow: VectorD =
        val a = Array.ofDim [Double] (dim)
        cfor (0, dim) { i => a(i) = apply(i).variance }
        new VectorD (a.size, a)
    end varianceRow

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the column standard deviations of this matrix.
     */
    def stdev: VectorD = variance.sqrt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the row standard deviations of this matrix.
     */
    def stdevRow: VectorD = varianceRow.sqrt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the column mean and standard deviations of this matrix.
     */
    def mu_sig: MatrixD = MatrixD (mean, stdev)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the matrix/grand mean of this matrix.
     */
    def mmean: Double = sum / (dim * dim2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the matrix/overall average variance of this matrix.
     */
    def mvariance: Double = variance.mean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the matrix/overall average standard deviation of this matrix.
     */
    def mstdev: Double = math.sqrt (mvariance)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the column first Q1 (1/4) and third Q3 (3/4) quartiles of this matrix.
     */
    def q1_q3: MatrixD =
        val a = new MatrixD (2, dim2)
        cfor (0, dim2) { j => a(?, j) = apply(?, j).q1_q3 }
        a
    end q1_q3

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a matrix that is in the reverse row order of this matrix.
     */
    def reverse: MatrixD = new MatrixD (dim, dim2, v.reverse)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this matrix is square (# row = # columns)
     */
    inline def isSquare: Boolean = dim == dim2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this matrix is symmetric (i.e, equals its transpose).
     */
    def isSymmetric: Boolean =
        var symm = true
        breakable {
            cfor (0, dim) { i => cfor (0, i) { j =>
                if v(i)(j) != v(j)(i) then
                    symm = false
                    println (s"MatrixD.isSymmetric: v($i)($j) = ${v(i)(j)} != v($j)($i) = ${v(j)(i)}")
                    break ()
            }} // cfor
        } // breakable
        symm
    end isSymmetric

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this matrix is non-negative (has no elements less than 0).
     */
    def isNonnegative: Boolean =
        var nonneg = true
        breakable {
            cfor (0, dim) { i => cfor (0, dim2) { j =>
                if v(i)(j) < 0.0 then { nonneg = false; break () }
            }} // cfor
        } // breakable
        nonneg
    end isNonnegative

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap (in-place) rows i and k in this matrix.
     *  @param i  the first row in the swap
     *  @param k  the second row in the swap
     */
    inline def swap (i: Int, k: Int): Unit =
        val tmp = v(i); v(i) = v(k); v(k) = tmp
    end swap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap (in-place) the elements in rows i and k starting from column col.
     *  @param i    the first row in the swap
     *  @param k    the second row in the swap
     *  @param col  the starting column for the swap
     */
    inline def swap (i: Int, k: Int, col: Int): Unit =
        val a = this; var tmp = 0.0
        cfor (col, dim2) { j => tmp = a(k, j); a(k, j) = a(i, j); a(i, j) = tmp }
    end swap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap (in-place) columns j and l in this matrix.
     *  @param j  the first column in the swap
     *  @param l  the second column in the swap
     */
    inline def swapCol (j: Int, l: Int): Unit =
        var tmp = 0.0
        cfor (0, dim) { i => tmp = v(i)(l); v(i)(l) = v(i)(j); v(i)(j) = tmp }
    end swapCol

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Center this matrix to zero mean, column-wise, by subtracting the mean.
     *  @param mu_x  the vector of column means for this matrix
     */
    def center (mu_x: VectorD = mean): MatrixD = this - mu_x

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sample covariance matrix for the columns of this matrix.
     */
    def cov: MatrixD =
        val z = center ()
        (z.transpose * z) / (dim.toDouble - 1.0)
    end cov

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the population covariance matrix for the columns of this matrix.
     */
    def cov_ : MatrixD =
        val z = center ()
        (z.transpose * z) / dim.toDouble
    end cov_

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the correlation matrix for the columns of this matrix.
     *  If either variance is zero (column i, column j), will result in Not-a-Number (NaN),
     *  return one if the vectors are the same, or -0 (indicating undefined).
     *  Note:  sample vs. population results in essentially the same values.
     *  @see the related cos function
     */
    def corr: MatrixD =
        val covv = cov                                         // sample covariance matrix
        val cor  = MatrixD.eye (covv.dim, covv.dim)            // correlation matrix

        cfor (0, covv.dim) { i =>
            val var_i = covv (i, i)                            // variance of column i
            cfor (0, i) { j =>
                cor(i, j) = covv (i, j) / math.sqrt (var_i * covv (j, j))
                if cor(i, j).isNaN then cor(i, j) = if v(i) == v(j) then 1.0 else -0.0
                cor(j, i) = cor (i, j)
        }} // cfor
        cor
    end corr

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the correlation vector for the columns of this matrix with vector y.
     *  VectorD (for j <- skip until dim2 yield apply(?, j) corr y)
     *  @param y     the vector to compute correlations with
     *  @param skip  the number of initial columns to skip (e.g., first column of all ones)
     */
    def corr (y: VectorD, skip: Int = 0): VectorD =
        VectorD ((skip until dim2).map { j => apply(?, j) corr y })
    end corr

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cosine similarity matrix for the columns of matrix 'x'.
     *  If the vectors are centered, will give the correlation.
     *  @see stats.stackexchange.com/questions/97051/]
     *       building-the-connection-between-cosine-similarity-and-correlation-in-r
     */
    def cosSim: MatrixD =
        val cs = MatrixD.eye (dim2, dim2)                      // cosine matrix

        cfor (0, cs.dim) { i =>
            val y  = apply(?, i)                               // ith column vector
            val ny = y.norm
            cfor (0, i) { j =>
                val z  = apply(?, j)                           // jth column vector
                val nz = z.norm
                cs(i, j) = (y dot z) / (ny * nz)
                cs(j, i) = cs (i, j)
        }} // cfor
        cs
    end cosSim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this matrix to the same matrix, i.e., return this matrix.
     */
    def toMatrixD: MatrixD = this

    private [mathstat] def toArray: Array [Array [Double]] = v

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this matrix to a matrix where all the elements have integer values.
     */
    def toInt: MatrixD = 
        val x = new MatrixD (dim, dim2)
        cfor (0, dim) { i => cfor (0, dim2) { j => x.v(i)(j) = round (v(i)(j)).toDouble }}
        x
    end toInt
        
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this matrix to a string.
     */
    override def toString: String = 
        val sb = new StringBuilder ("\nMatrixD (")
        if dim == 0 || dim2 == 0 then return sb.append (")").mkString
        cfor (0, dim) { i => cfor (0, dim2) { j =>
            sb.append (fString.format (v(i)(j)))
            if j == dim2-1 then sb.replace (sb.length-1, sb.length, "\n \t")
        }} // cfor
        sb.replace (sb.length-4, sb.length, ")").mkString
    end toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write this matrix to a CSV-formatted text file with name fileName.
     *  Each row in the matrix will be stored a line in the file, with values comma separated.
     *  @param fileName  the name of file to hold the data
     *  @param header    the optional header (first line) for the file, giving the column headings
     *  @param fullPath  flag indicating whether the user wants full or relative file paths
     *                   defaults to false (relative paths)
     */
    def write (fileName: String, header: Array [String] = null,
               fullPath: Boolean = false): Unit =
        val path = if fullPath then fileName
                   else DATA_DIR + fileName                                // relative to DATA_DIR
        val out = new PrintWriter (path)
        if header != null then
            cfor (0, dim2) { j =>
                out.print (header(j))
                if j < dim2-1 then out.print (",")
            } // cfor
            out.println ()
        end if

        cfor (0, dim) { i =>
            cfor (0, dim2) { j =>
                out.print (v(i)(j))
                if j < dim2-1 then out.print (",")
            } // cfor
            out.println ()
        } // cfor
        println (s"MatrixD.write: matrix with dims = $dims written to $path")
        out.close
    end write

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Helper methods for Autograd
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the dimensions of the matrix as a list.
     *  The first element is the number of rows and the second is the number of columns.
     *  @return a List [Int] containing the matrix dimensions.
     */
    def shape: List [Int] = List (dim, dim2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Creates a new matrix with the same dimensions as this matrix where every element
     *  is set to zero.
     *  @return a MatrixD filled with zeros.
     */
    def zerosLike: MatrixD = MatrixD.fill (dim, dim2, 0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Creates a new matrix with the same dimensions as this matrix where every element
     *  is set to one.
     *  @return a MatrixD filled with ones.
     */
    def onesLike: MatrixD = MatrixD.fill (dim, dim2, 1)

end MatrixD


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MatrixD companion object provides factory methods.
 */
object MatrixD:

    private val flaw = flawf ("MatrixD")                       // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix from repeated values.
     *  @param dim  the (row, column) dimensions
     *  @param u    the repeated values
     */
    def apply (dim: (Int, Int), u: Double*): MatrixD =
        val a = Array.ofDim [Double] (dim._1, dim._2)
        cfor (0, dim._1) { i => cfor (0, dim._2) { j => a(i)(j) = u(i * dim._2 + j) }}
        new MatrixD (dim._1, dim._2, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a square matrix of dimension dim where all elements equal zero.
     *  @param dim  the square dimensions
     */
    def apply (dim: Int): MatrixD = new MatrixD (dim, dim)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix from a variable argument list of vectors (row-wise).
     *  Use transpose to make it column-wise.
     *  @param vs  the vararg list of vectors
     */
    def apply (vs: VectorD*): MatrixD =
        val (m, n) = (vs.length, vs(0).length)
        val a = Array.ofDim [Array [Double]] (m)
        cfor (0, vs.size) { i => a(i) = vs(i).v }
        new MatrixD (m, n, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix from an mutable `IndexedSeq` of vectors (row-wise).
     *  Use transpose to make it column-wise.
     *  @param vs  the indexed sequence of vectors
     */
    def apply (vs: IndexedSeq [VectorD]): MatrixD =
        val (m, n) = (vs.length, vs(0).length)
        val a = Array.ofDim [Array [Double]] (m)
        cfor (0, vs.size) { i => a(i) = vs(i).v }
        new MatrixD (m, n, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix from an immutable `IndexedSeq` of vectors (row-wise),
     *  as produce by for yield.  Use transpose to make it column-wise.
     *  @param vs  the indexed sequence of vectors
     */
    def apply (vs: collection.immutable.IndexedSeq [VectorD]): MatrixD =
        val (m, n) = (vs.length, vs(0).length)
        val a = Array.ofDim [Array [Double]] (m)
        cfor (0, vs.size) { i => a(i) = vs(i).v }
        new MatrixD (m, n, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a lower triangular matrix from repeated values for elements below
     *  the main diagonal.  The rest of the elements are all zero.
     *  @param dim  the (row, column) dimensions
     *  @param u    the repeated values
     */
    def low (dim: Int)(u: Double*): MatrixD =
        val a = Array.ofDim [Double] (dim, dim)
        var k = 0
        cfor (0, dim) { i => cfor (0, i) { j => a(i)(j) = u(k); k += 1 }}
        new MatrixD (dim, dim, a)
    end low

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an m-by-1 matrix from an m-vector (column-wise).
     *  @param v  the vector to build the matrix from
     */
    def fromVector (v: VectorD): MatrixD =
        val x =  new MatrixD (v.dim, 1)
        cfor (0, x.dim) { i => x(i, 0) = v(i) }
        x
    end fromVector

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Extract the i-th vectors from a pair of matrices.
     *  @param x_y  the matrix pair
     *  @param i    the extraction index
     */
    inline def at (x_y: (MatrixD, MatrixD), i: Int): (VectorD, VectorD) = (x_y._1(i), x_y._2(i))

    private val DEF_SEP  = ','                                 // default character separating the values
    private val PROGRESS = 1000                                // give feedback at progress count

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix by reading from a text file, e.g., a CSV file.
     *  @see the `write` method in the class to write a matrix to a CSV file.
     *  @param fileName  the name of file holding the data
     *  @param skip      the initial number of lines/rows to skip
     *  @param skipCol   the initial number of columns to skip
     *  @param sp        the character used to separate values (',', '\t', ...)
     *  @param fullPath  flag indicating whether to use full-path or path relative to 'DATA_DIR'
     *                   defaults to false (relative paths)
     *  @param stop      the line/row number to stop before reaching the EOF (exclusive)
     */
    def load (fileName: String, skip: Int = 0, skipCol: Int = 0,
              sp: Char = DEF_SEP, fullPath: Boolean = false, stop: Int = MAX_INTEGER): MatrixD =
        val lines = readFileIntoArray (fileName, fullPath)     // array of strings/lines
        val m  = lines.length                                  // number lines in the file
        val ms = min (m, stop)                                 // line number to stop before
        val mm = ms - skip                                     // number of lines with data
        val a  = Array.ofDim [Array [Double]] (mm)             // array buffer to hold data values
        var n  = -1                                            // number of values in a row (assigned below)

        cfor (skip, ms) { i =>
            val j = i - skip
            a(j) = for str <- lines(i).split (sp).drop (skipCol) yield str.mkDouble
            if (j+1) % PROGRESS == 0 then println (s"load: read $j data rows so far ...")
            if n < 0 then n = a(j).length
            else if a(j).length != n then flaw ("load", s"row $j has the wrong length ${a(j).length} != $n")
        } // cfor
        println (s"load: read in an $mm-by-$n matrix from $fileName")
        new MatrixD (mm, n, a)
    end load

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix and a header by reading from a text file, e.g., a CSV file.
     *  Assumes the column names (header) is in the first line of the file.
     *  @see the `write` method in the class to write a matrix to a CSV file.
     *  @param fileName  the name of file holding the data
     *  @param skipCol   the initial number of columns to skip
     *  @param sp        the character used to separate values (',', '\t', ...)
     *  @param fullPath  flag indicating whether to use full-path or path relative to 'DATA_DIR'
     *                   defaults to false (relative paths)
     *  @param stop      the line/row number to stop before reaching the EOF (exclusive)
     */
    def loadH (fileName: String, skipCol: Int = 0, sp: Char = DEF_SEP,
               fullPath: Boolean = false, stop: Int = MAX_INTEGER): (MatrixD, Array [String]) =
        val lines = readFileIntoArray (fileName, fullPath)     // array of strings/lines
        val m  = lines.length                                  // number lines in the file
        val ms = min (m, stop)                                 // line number to stop before
        val hd = lines(0).split (sp).drop (skipCol)            // get the column header from line 0
        val mm = ms - 1                                        // number of lines with data
        val a  = Array.ofDim [Array [Double]] (mm)             // array buffer to hold data values
        var n  = -1                                            // number of values in a row (assigned below)

        cfor (1, ms) { i =>
            val j = i - 1 
            a(j) = for str <- lines(i).split (sp).drop (skipCol) yield str.mkDouble
            if (j+1) % PROGRESS == 0 then println (s"load: read $j data rows so far ...")
            if n < 0 then n = a(j).length
            else if a(j).length != n then flaw ("load", s"row $j has the wrong length ${a(j).length} != $n")
        } // cfor
        println (s"load: read in an $mm-by-$n matrix from $fileName")
        (new MatrixD (mm, n, a), hd)
    end loadH

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix by reading from a text file, e.g., a CSV file.
     *  Convert string columns into ordinal/integer columns.
     *  @param fileName  the name of file holding the data
     *  @param skip      the initial number of lines/rows to skip
     *  @param skipCol   the initial number of columns to skip
     *  @param sp        the character used to separate values (',', '\t', ...)
     *  @param fullPath  flag indicating whether to use full-path or path relative to 'DATA_DIR'
     *                   defaults to false (relative paths)
     *  @param ordCols   the set of ordinal columns (column indices)
     *  @param ordStrs   the corresponding strings for the ordinal/integer values
     */
    def loadStr (fileName: String, skip: Int = 0, skipCol: Int = 0,
                 sp: Char = DEF_SEP, fullPath: Boolean = false)
                (ordCols: Set [Int], ordStrs: VectorS*): MatrixD =
        val lines = readFileIntoArray (fileName, fullPath)     // array of strings/lines
        val m  = lines.length                                  // number lines in the file
        val mm = m - skip                                      // number of lines with data
        val a  = Array.ofDim [Array [Double]] (mm)             // array buffer to hold data values
        var n  = -1                                            // number of values in a row (TBD)

        cfor (skip, m) { i =>
            val j = i - skip
            var col, ordCol = -1
            a(j) = for str <- lines(i).split (sp).drop (skipCol) yield
                   col += 1
                   if ordCols contains col then
                        ordCol += 1
                        mkOrdinal (str, ordStrs(ordCol))
                   else str.mkDouble
            if (j+1) % PROGRESS == 0 then println (s"load: read $j data rows so far ...")
            if n < 0 then n = a(j).length
            else if a(j).length != n then flaw ("load", s"row $j has the wrong length ${a(j).length} != $n")
        } // cfor
        println (s"load: read in an $mm-by-$n matrix from $fileName")
        new MatrixD (mm, n, a)
    end loadStr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a string value, convert it to an ordinal/integer based on the ordStr mapping.
     *  @param str     the string to be mapped to an ordinal value
     *  @param ordStr  the VectorS containing strings that can be ordered, e.g.,
     *                 VectorS ("low", "medium", "high") for 0, 1, 2
     */
    def mkOrdinal (str: String, ordStr: VectorS): Int =
        ordStr.map2Int._2 (str)                                 // @see `VectorS`, return str mapped to an integer
    end mkOrdinal

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix-vector pair (x, y) by reading from a text file, e.g., a CSV file.
     *  Use readFileIter to only read the necessary columns from the file.
     *  @param fileName  the name of file holding the data
     *  @param xCols     the columns that are to make up the x-matrix
     *  @param yCol      the column that is to make up the y-vector (use the default -1 to skip this)
     *  @param skip      the initial number of lines to skip
     *  @param sp        the character used to separate values (',', '\t', ...)
     *  @param fullPath  flag indivating whether to use full-path or path relative to 'DATA_DIR'
     *                   defaults to false (relative paths)
     */
    def loadIter (fileName: String, xCols: Array [Int], yCol: Int = -1, skip: Int = 0,
                  sp: Char = DEF_SEP, fullPath: Boolean = true): (MatrixD, VectorD) =
        val (it, buffer) = readFileIter (fileName, fullPath)   // iterator of strings/lines, io buffer
        val xAb = ArrayBuffer [Array [Double]] ()              // array buffer to hold x-matrix
        val yAb = ArrayBuffer [Double] ()                      // array buffer to hold y-vector
        var n   = -1                                           // number of values in a row (determined below)

        var i = 0                                              // line number
        while it.hasNext do
            val line = it.next()                               // read next line
            if i >= skip then
                val k = i - skip                               // row number
                val token = line.split (sp)
                xAb += (for j <- xCols yield token(j).mkDouble).toArray
                if yCol >= 0 then yAb += token (yCol).mkDouble
                if (k+1) % PROGRESS == 0 then println (s"loadIter: read $k data rows so far ...")
                if n < 0 then n = xAb(k).length
                else if xAb(k).length != n then flaw ("loadIter", s"row $k has the wrong length ${xAb(k).length} != $n")
            end if
            i += 1
        end while
        buffer.close ()

        val m = xAb.size
        (new MatrixD (m, n, xAb.toArray), new VectorD (m, yAb.toArray))
    end loadIter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix of dimension dim by 1 that consists of all ones.
     *  @param dim  the row dimension
     */
    def one (dim: Int): MatrixD =
        val a = Array.fill (dim, 1)(1.0)
        new MatrixD (dim, 1, a)
    end one

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an identity matrix of dimensions dim by dim2 where all elements equal zero,
     *  except the main diagonal where elements are set to one.
     *  @param dim   the row dimension
     *  @param dim2  the column dimension
     */
    def eye (dim: Int, dim2: Int): MatrixD = 
        val x = new MatrixD (dim, dim2)
        x(?, ?) = 1.0                                          // set diagonal to one
        x
    end eye

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a diagonal matrix of dimensions dim by dim2 where all elements equal zero,
     *  except the main diagonal where elements are set to y_i.
     *  @param dim   the row dimension
     *  @param dim2  the column dimension
     *  @param y     the vector to set the main diagonal to
     */
    def diag (dim: Int, dim2: Int, y: VectorD): MatrixD = 
        val x = new MatrixD (dim, dim2)
        val n = x.minDim
        if y.dim < n then flaw ("diag", s"the dimension of vector y = ${y.dim} is too small to fill diagonal")
        cfor (0, x.minDim) { i => x(i, i) = y(i) }             // set diagonal to y_i
        x
    end diag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix of dimensions dim by dim2 where all elements equal to the given value. 
     *  @param dim    the row dimension
     *  @param dim2   the column dimension
     *  @param value  the given value to assign to all elements
     */
    def fill (dim: Int, dim2: Int, value: Double): MatrixD = 
        val a = Array.fill (dim, dim2)(value)
        new MatrixD (dim, dim2, a)
    end fill

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** A null matrix of type `MatrixD`.
     */
    val nullm: MatrixD = null.asInstanceOf [MatrixD]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the outer product of vector x and vector y.  The result of the
     *  outer product is a matrix where element (i, j) is the product of i-th element
     *  of x with the j-th element of y, an x.dim by y.dim matrix.
     *  @param x  the first vector
     *  @param y  the second vector
     */
    def outer (x: VectorD, y: VectorD): MatrixD =
        val a = Array.ofDim [Double] (x.dim, y.dim)
        cfor (0, x.dim) { i => cfor (y.indices) { j => a(i)(j) = x(i) * y(j) }}
        new MatrixD (x.dim, y.dim, a)
    end outer

    inline def ⊗ (x: VectorD, y: VectorD): MatrixD = outer (x, y)    // Unicode tensor product

end MatrixD


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MatrixDOps` object provides extension methods to support scalar op matrix 
 *  operations, so that one can write 2.0 + x as well as x + 2.0.
 */
object MatrixDOps:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of scalar a
     *  and matrix x, e.g., 2.0 * x.
     *  @param a  the scalar first operand
     *  @param x  the vector second operand
     */
    extension (a: Double)
        def + (x: MatrixD): MatrixD = x + a
        def - (x: MatrixD): MatrixD = -x + a
        def * (x: MatrixD): MatrixD = x * a
        def / (x: MatrixD): MatrixD = x.recip * a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the outer product of vector x and vector y.
     *  @param x  the vector first operand
     *  @param y  the vector second operand
     */
    extension (x: VectorD)
        inline def ⊗ (y: VectorD): MatrixD = MatrixD.outer (x, y)

end MatrixDOps


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MatrixDExample` object provides example instances of the`MatrixD` class.
 */
object MatrixDExample:

    val x = MatrixD ((8, 8), 1, 2,  3,  4,  5,  6,  7,  8,
                             2, 3,  4,  5,  6,  7,  8,  9,
                             3, 4,  5,  6,  7,  8,  9, 10,
                             4, 5,  6,  7,  8,  9, 10, 11,
                             5, 6,  7,  8,  9, 10, 11, 12,
                             6, 7,  8,  9, 10, 11, 12, 13,
                             7, 8,  9, 10, 11, 12, 13, 14,
                             8, 9, 10, 11, 12, 13, 14, 15)

    val y = MatrixD ((8, 8), 1, 2,  3,  4,  5,  6,  7,  8,
                             2, 3,  4,  5,  6,  7,  8,  9,
                             3, 4,  5,  6,  7,  8,  9, 10,
                             4, 5,  6,  7,  8,  9, 10, 11,
                             5, 6,  7,  8,  9, 10, 11, 12,
                             6, 7,  8,  9, 10, 11, 12, 13,
                             7, 8,  9, 10, 11, 12, 13, 14,
                             8, 9, 10, 11, 12, 13, 14, 15)

end MatrixDExample

import MatrixDExample.{x, y}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixDTest` main function tests the `MatrixD` class.  Compares the performance
 *  of matrix addition implementations
 *  > runMain scalation.mathstat.matrixDTest
 */
@main def matrixDTest (): Unit =

    import MatrixDOps._

    println (s"x = $x")
    val c = 3.0

    banner ("Test apply methods")

    println (s" x(3, 2)                 = ${x(3, 2)}")                  // element (3, 2)
    println (s" x(3 until 6, 2 until 4) = ${x(3 until 6, 2 until 4)}")  // slice of rows and columns
    println (s" x(3)                    = ${x(3)}")                     // row 3
    println (s" x(3 until 6)            = ${x(3 until 6)}")             // slice of rows
    println (s" x(?, 2)                 = ${x(?, 2)}")                  // column 2
    println (s" x(?, 2 until 4)         = ${x(?, 2 until 4)}")          // slice of columns

    banner ("Test element-wise methods")

    println (s" x + y  = ${x + y}")
    println (s" x - y  = ${x - y}")
    println (s" x *~ y = ${x *~ y}")
    println (s" x / y  = ${x / y}")
    println (s" x ~^ 2 = ${x ~^ 2}")

    println (s"c + x   = ${c + x}")                                     // add scalar c and x
    println (s"c - x   = ${c - x}")                                     // subtract from scalar c, x
    println (s"c * x   = ${c * x}")                                     // multiply by scalar c and x
    println (s"c / x   = ${c / x}")                                     // divide scalar c by x

    println (s" x.crossAll = ${x.crossAll}")

    val a = new MatrixD (1000, 1000)
    val b = new MatrixD (1000, 1000)
    cfor (0, a.dim) { i => cfor (0, a.dim2) { j => a(i, j) = i + j; b(i, j) = a(i, j) }}

    cfor (1, 11) { it =>
        banner (s"Timing results to iteration $it")
        val t1 = gauge { a + b };  println (s" a + b  = $t1")
        val t2 = gauge { a - b };  println (s" a - b  = $t2")
        val t3 = gauge { a *~ b }; println (s" a *~ b = $t3")
        val t4 = gauge { a / b };  println (s" a / b  = $t4")
    } // cfor

end matrixDTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixDTest2` main function tests the `MatrixD` class.  Compares the performance
 *  of matrix multiplication implementations.
 *  > runMain scalation.mathstat.matrixDTest2
 */
@main def matrixDTest2 (): Unit =

    println (s" x mul y  = ${x mul y}")
    println (s" x * y    = ${x * y}")
    println (s" x dot y  = ${x dot y}")
    println (s" x * y(0) = ${x * y(0)}")

    val a = new MatrixD (1000, 1000)
    val b = new MatrixD (1000, 1000)
    cfor (0, a.dim) { i => cfor (0, a.dim2) { j => a(i, j) = i + j; b(i, j) = a(i, j) }}

    cfor (1, 11) { it =>
        banner (s"Timing results to iteration $it")
        val t1 = gauge { a mul b };  println (s" a mul b  = $t1")
        val t2 = gauge { a * b };    println (s" a * b    = $t2")
        val t3 = gauge { a dot b };  println (s" a dot b  = $t3")
        val t4 = gauge { a * b(0) }; println (s" a * b(0) = $t4")
    } // cfor

end matrixDTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixDTest3` main function tests the `MatrixD` class.  Test the back substitution
 *  (/~) operator for solving for x in u * x = y.
 *  It also computes an outer product
 *  > runMain scalation.mathstat.matrixDTest3
 */
@main def matrixDTest3 (): Unit =

    import MatrixD.⊗

    val u = MatrixD ((3, 3), 1, 2, 3,
                             0, 4, 5,
                             0, 0, 6)
    val y = VectorD (1, 2, 3)

    banner ("Back Substitution")
    val x = u /~ y
    println (s"x = u /~ y = $x")
    assert (u * x == y)
    println (s"y = $y")

    banner ("Outer Product")
    println ("⊗ (x, y) = " + ⊗ (x, y))

end matrixDTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixDTest4` main function tests the `MatrixD` class.  Test the split
 *  methods: split and split_ as well as shiftDiag and unshiftDiag
 *  > runMain scalation.mathstat.matrixDTest4
 */
@main def matrixDTest4 (): Unit =

    val x = MatrixD ((10, 3), 1,  1,  1,
                              2,  2,  2,
                              3,  3,  3,
                              4,  4,  4,
                              5,  5,  5,
                              6,  6,  5,
                              7,  7,  7,
                              8,  8,  8,
                              9,  9,  9,
                             10, 10, 10)

    val idx = VectorI (1, 2, 5, 9)

    banner ("Test split methods")
    val (x_e, x_) = x.split (idx)
    val (z_e, z_) = x.split_ (idx)

    println (s"split:  x_e = $x_e")
    println (s"split_: z_e = $z_e")

    println (s"split:  x_  = $x_")
    println (s"split_: z_  = $z_")

    assert (z_e =~ x_e)
    assert (z_ =~ x_)

    banner ("Test shiftDiag methods")
    val _x  = x.shiftDiag
    val _x_ = _x.unshiftDiag

    println (s"x   = x              = $x")
    println (s"_x  = x.shiftDiag    = $_x")
    println (s"_x_ = _x.unshiftDiag = $_x_")

    assert (_x_ =~ x)

end matrixDTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixDTest5` main function tests the `MatrixD` class.  Test the covariance
 *  and correlation.
 *  > runMain scalation.mathstat.matrixDTest5
 */
@main def matrixDTest5 (): Unit =

    val x = MatrixD ((6, 2), 1, 1,
                             2, 3,
                             3, 3,
                             4, 5,
                             5, 4,
                             6, 4)

    println (s"Data Matrix x           = $x")
    println (s"Samp. Covariance x.cov  = ${x.cov}")
    println (s"Pop.  Covariance x.cov_ = ${x.cov_}")
    println (s"Correlation x.corr      = ${x.corr}")

end matrixDTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixDTest6` main function tests the `MatrixD` class.  Test the insert method.
 *  > runMain scalation.mathstat.matrixDTest6
 */
@main def matrixDTest6 (): Unit =

    val x = MatrixD ((4, 5), 1, 2, 3, 4, 5, 
                             1, 2, 3, 4, 5,
                             1, 2, 3, 4, 5,
                             1, 2, 3, 4, 5)

    println (s"Matrix x = $x")

    banner ("x.insert (1, 3, u)")
    val u = VectorD (1, 2, 3, 4)
    x.insert (1, 3, u)
    println (s"Matrix x = $x")

end matrixDTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixDTest7` main function tests the `MatrixD` class.  Test the convolution
 *  operators.
 *  > runMain scalation.mathstat.matrixDTest7
 */
@main def matrixDTest7 (): Unit =


    val x  = MatrixD ((5, 5), 0, 0, 2, 1, 0,
                              0, 0, 0, 1, 2,
                              1, 2, 2, 0, 2,
                              2, 0, 0, 0, 1,
                              2, 2, 2, 0, 1)

    val c  = MatrixD ((2, 2), 1, 1,
                              0, 1)

    banner ("Convolution Operators")
    println (s"c conv x  = ${c conv x}")                 // conv   valid convolution, no reversal
    println (s"c *+ x    = ${c *+ x}")                   // *+     valid convolution, no reversal
    println (s"c conv_ x = ${c conv_ x}")                // conv_  valid convolution, with reversal
    println (s"c convs x = ${c convs x}")                // convs  same convolution, with reversal
    println (s"c *~+ x   = ${c *~+ x}")                  // *~+    same convolution, with reversal
    println (s"c convf x = ${c convf x}")                // convf  full convolution, with reversal
    println (s"c *++ x   = ${c convf x}")                // *++    full convolution, with reversal

end matrixDTest7

