
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Thu May 10 15:50:15 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Tensor (3D) Algebra
 *
 *  @see www.stat.uchicago.edu/~lekheng/work/icm1.pdf
 *  @see www.math.ias.edu/csdm/files/13-14/Gnang_Pa_Fi_2014.pdf
 *  @see www.kolda.net/publication/TensorReview.pdf
 *  @see tspace.l
 */

package scalation
package mathstat

import scala.annotation.unused
import scala.collection.mutable.IndexedSeq
import scala.runtime.ScalaRunTime.stringOf

import scalation.modeling.ActivationFun
import scalation.modeling.ActivationFun.{eLU, setA2, gaussian, geLU, logistic, logit, lreLU, setA, reLU, sigmoid}

import TensorD._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Tensorize a vector function (V2V) by applying it to each (row, column) of a tensor.
 *  @param f  the vector function to tensorize
 *  @param x  the tensor to apply the function to
 */
def tensorize (f: FunctionV2V)(x: TensorD): TensorD =
    val t = new TensorD (x.dim, x.dim2, f(x(0, 0)).dim)
    cfor (x.indices) { i => cfor (x.indices2) { j => t(i, j) = f(x(i, j)) }}
    t
end tensorize


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the complement of index positions idx, e.g.,
 *  comple (Array (1, 3), 5) = Array (0, 2, 4).
 *  param idx  the index positions to be complemented 
 *  param dim  the exclusive upper bound
 */
def comple (idx: Array [Int], dim: Int): Array [Int] =
    val a = Array.ofDim [Int] (dim - idx.size)
    var j, l = 0
    cfor (0, idx.length) { i =>
        while j < idx(i) do
            a(l) = j
            j += 1
            l += 1
        end while
        j += 1
    } // cfor
    a
end comple


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TensorD` class is a simple implementation for 3-dimensional tensors.
 *  The names of the dimensions corresponds to MATLAB (row, column, sheet).
 *  @see www.kolda.net/publication/TensorReview.pdf for details on layout
 *  @see `RTensorD` for non-rectangular (ragged) tensors.
 *  @param dim   size of the 1st level/dimension (row) of the tensor (height)
 *  @param dim2  size of the 2nd level/dimension (column) of the tensor (width)
 *  @param dim3  size of the 3rd level/dimension (sheet) of the tensor (depth)
 *  @param v     the 3D array for holding the tensor elements
 */
class TensorD (val dim: Int, val dim2: Int, val dim3: Int,
               private [mathstat] var v: Array [Array [Array [Double]]] = null)
      extends Serializable:

    private val flaw =  flawf ("TensorD")                     // flaw flag
    private val TAB  = "\t\t"                                 // use "\t" for scala and "\t\t" for sbt

    private val _shape = List (dim, dim2, dim3)               // list of the dimensions of the tensor

    val indices  = 0 until dim                                // index range for the first level/dimension
    val indices2 = 0 until dim2                               // index range for the second level/dimension
    val indices3 = 0 until dim3                               // index range for the third level/dimension

    /** Multi-dimensional array storage for tensor
     */
    if v == null then
        v = Array.ofDim [Double] (dim, dim2, dim3)
    else if dim != v.length || dim2 != v(0).length || dim3 != v(0)(0).length then
        flaw ("init", "dimensions are wrong")

    /** Format string used for printing vector values (change using setFormat)
     */
    protected var fString = "%g,\t"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a dim by dim by dim cubic tensor.
     *  @param dim  the row and column dimension
     */
    def this (dim: Int) = { this (dim, dim, dim) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a tensor from three dimensional array.
     *  @param u  the three dimensional array
     */
    def this (u: Array [Array [Array [Double]]]) = { this (u.size, u(0).size, u(0)(0).size, u) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the row, column and sheet dimensions of this tensor.
     */
    inline def dims: (Int, Int, Int) = (dim, dim2, dim3)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the format to the newFormat.
     *  @param newFormat  the new format string
     */
    def setFormat (newFormat: String): Unit = fString = newFormat

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the i, j, k-th SCALAR element from the tensor x_ijk.
     *  @param i  the 1st dimension (row) index of the tensor
     *  @param j  the 2nd dimension (column) index of the tensor
     *  @param k  the 3rd dimension (sheet) index of the tensor
     */
    def apply (i: Int, j: Int, k: Int): Double = v(i)(j)(k)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the i, j-th VECTOR from the tensor x_ij:.
     *  @param i  the 1st dimension (row) index of the tensor
     *  @param j  the 2nd dimension (column) index of the tensor
     */
    def apply (i: Int, j: Int): VectorD = VectorD (v(i)(j))
//  def apply (i: Int, j: Int): VectorD = VectorD (v(i)(j).toIndexedSeq)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the i, k-th VECTOR from the tensor x_i:k.
     *  @param i    the 1st dimension (row) index of the tensor
     *  @param all  use the all columns indicator ?
     *  @param k    the 3rd dimension (sheet) index of the tensor
     */
    def apply (i: Int, @unused all: Char, k: Int): VectorD = 
        val a = Array.ofDim [Double] (dim2)
        cfor (0, dim2) { j => a(j) = v(i)(j)(k) }
        new VectorD (dim2, a)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the j, k-th VECTOR from the tensor x_:jk.
     *  @param all  use the all rows indicator ?
     *  @param j    the 2nd dimension (column) index of the tensor
     *  @param k    the 3rd dimension (sheet) index of the tensor
     */
    def apply (@unused all: Char, j: Int, k: Int): VectorD = 
        val a = Array.ofDim [Double] (dim)
        cfor (0, dim) { i => a(i) = v(i)(j)(k) }
        new VectorD (dim, a)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the i-th ROW FIXED MATRIX from the tensor (horizontal slice x_i::).
     *  @see www.kolda.net/publication/TensorReview.pdf
     *  @param i  the 1st dimension (row) index of the tensor
     */
    def apply (i: Int): MatrixD = new MatrixD (dim2, dim3, v(i))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the j-th COLUMN FIXED MATRIX from the tensor (lateral slice x_:j:).
     *  @see www.kolda.net/publication/TensorReview.pdf
     *  @param all  use the all rows indicator ?
     *  @param j    the 2nd dimension (column) index of the tensor
     */
    def apply (@unused all: Char, j: Int): MatrixD =
        val a = Array.ofDim [Double] (dim, dim3)
        cfor (0, dim) { i => cfor (0, dim3) { k => a(i)(k) = v(i)(j)(k) }}
        new MatrixD (dim, dim3, a)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the k-th SHEET FIXED MATRIX from the tensor (frontal slice x_::k).
     *  @see www.kolda.net/publication/TensorReview.pdf
     *  @param all   use the all rows indicator ?
     *  @param all2  use the all columns indicator ?
     *  @param k     the 3rd dimension (sheet) index of the tensor
     */
    inline def apply (@unused all: Char, @unused all2: Char, k: Int): MatrixD =
        val a = Array.ofDim [Double] (dim, dim2)
        cfor (0, dim) { i => cfor (0, dim2) { j => a(i)(j) = v(i)(j)(k) }}
        new MatrixD (dim, dim2, a)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the ii._1 to ii._2 row slice of the tensor.
     *  @param ii  1st dimension (row) (start, end) indices of the tensor
     */
    def apply (ii: (Int, Int)): TensorD = new TensorD (v.slice (ii._1, ii._2))

    inline def apply (ir: Range): TensorD = apply ((ir.start, ir.end))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the ii._1 to ii._2, jj._1 to jj._2 row-column slice of the tensor.
     *  @param ii  1st dimension (row) indices of the tensor (null => all)
     *  @param jj  2nd dimension (column) indices of the tensor
     */
    def apply (ii: (Int, Int), jj: (Int, Int)): TensorD =
        val (i1, i2) = if ii == null then (0, dim) else ii
        val u = v.slice (i1, i2)
        cfor (u.indices) { i => u(i) = u(i).slice (jj._1, jj._2)}
        new TensorD (u)
    end apply

    inline def apply (ir: Range, jr: Range): TensorD = apply ((ir.start, ir.end), (jr.start, jr.end))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the ii._1 to ii._2, jj._1 to jj._2, kk._1 to kk._2
     *  row-column-sheet slice of the tensor.
     *  @param ii  1st dimension (row) indices of the tensor (null => all)
     *  @param jj  2nd dimension (column) indices of the tensor (null => all)
     *  @param kk  3rd dimension (sheet) indices of the tensor
     */
    def apply (ii: (Int, Int), jj: (Int, Int), kk: (Int, Int)): TensorD =
        val (i1, i2) = if ii == null then (0, dim) else ii
        val (j1, j2) = if jj == null then (0, dim2) else jj
        val u = v.slice (i1, i2)
        cfor (u.indices) { i => u(i) = u(i).slice (j1, j2) }
        cfor (u.indices) { i =>
            cfor (u(i).indices) { j => u(i)(j) = u(i)(j).slice (kk._1, kk._2) }
        } // cfor
        new TensorD (u)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     */
    def apply (is: Array [Int]): TensorD = 
        val u = Array.ofDim [Double] (is.size, dim2, dim3)
        cfor (is.indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => u(i)(j)(k) = v(is(i))(j)(k) }
            } // cfor
        } // cfor
        new TensorD (u)
    end apply

    inline def apply (is: VectorI): TensorD = apply (is.toArray)

    inline def apply (is: Seq[Int]): TensorD = apply (is.toArray)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is, js row-column selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor (null => all)
     *  @param js  2nd dimension (column) indices of the tensor
     */
    def apply (is: Array [Int], js: Array [Int]): TensorD =
        if is == null then
            val u = Array.ofDim [Double] (dim, js.length, dim3)
            cfor (indices) { i =>
                cfor (js.indices) { j =>
                    cfor (indices3) { k => u(i)(j)(k) = v(i)(js(j))(k) }
                } // cfor
            } // cfor
            new TensorD (u)
        else
            val u = Array.ofDim [Double] (is.length, js.length, dim3)
            cfor (is.indices) { i =>
                cfor (js.indices) { j =>
                    cfor (indices3) { k => u(i)(j)(k) = v(is(i))(js(j))(k) }
                } // cfor
            } // cfor
            new TensorD (u)
        end if
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is, js, ks row-column-sheet selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor (null => all)
     *  @param js  2nd dimension (column) indices of the tensor (null => all)
     *  @param ks  3rd dimension (sheet) indices of the tensor
     */
    def apply (is: Array [Int], js: Array [Int], ks: Array [Int]): TensorD = 
        if is == null && js == null then
            val u = Array.ofDim [Double] (dim, dim2, ks.length)
            cfor (indices) { i =>
                cfor (indices2) { j =>
                    cfor (ks.indices) { k => u(i)(j)(k) = v(i)(j)(ks(k)) }
                } // cfor
            } // cfor
            new TensorD (u)
        else if is == null then
            val u = Array.ofDim [Double] (dim, js.size, ks.size)
            cfor (indices) { i =>
                cfor (js.indices) { j =>
                    cfor (ks.indices) { k => u(i)(j)(k) = v(i)(js(j))(ks(k)) }
                } // cfor
            } // cfor
            new TensorD (u)
        else if js == null then
            val u = Array.ofDim [Double] (is.size, dim2, ks.size)
            cfor (is.indices) { i =>
                cfor (indices2) { j =>
                    cfor (ks.indices) { k => u(i)(j)(k) = v(is(i))(j)(ks(k)) }
                } // cfor
            } // cfor
            new TensorD (u)
        else
            val u = Array.ofDim [Double] (is.size, js.size, ks.size)
            cfor (is.indices) { i =>
                cfor (js.indices) { j =>
                    cfor (ks.indices) { k => u(i)(j)(k) = v(is(i))(js(j))(ks(k)) }
                } // cfor
            } // cfor
            new TensorD (u)
        end if
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve a slice of the tensor as a MatrixD, based on a row range and specified sheet.
     *  @param ir     the range of rows to include in the slice.
     *  @param all2   a character indicating all columns should be included (typically '?').
     *  @param sheet  the index of the sheet to extract from.
     */
    def apply (ir: Range, @unused all2: Char, sheet: Int): MatrixD =
        val slicedArray = v.slice (ir.start, ir.end).map (_.map (_(sheet)))
        new MatrixD (ir.size, dim2, slicedArray)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve a sub-tensor based on the three ranges.
     *  @param ir  the range of rows to include in the slice.
     *  @param jr  the range of columns to include in the slice.
     *  @param kr  the range of sheet to include in the slice.
     */
    def apply (ir: Range, jr: Range, kr: Range): TensorD =
        val i1 = ir.start;
        val j1 = jr.start;
        val k1 = kr.start
        val slicedArray = Array.ofDim [Double] (ir.size, jr.size, kr.size)
        cfor (ir) { i =>
            val v_i = v(i);
            val a_i = slicedArray (i - i1)
            cfor (jr) { j =>
                val v_ij = v_i(j);
                val a_ij = a_i(j - j1)
                cfor (kr) { k => a_ij(k - k1) = v_ij(k) }
            } // cfor
        } // cfor
        new TensorD (slicedArray)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the complement of the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     */
    def not (is: Array [Int]): TensorD = apply (Array.range (0, dim) diff is)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the complement of the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     *  @param js  2nd dimension (column) indices of the tensor
     */
    def not (is: Array [Int], js: Array [Int]): TensorD =
        apply (comple (is, dim), comple (js, dim2))
    end not

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the complement of the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     *  @param js  2nd dimension (column) indices of the tensor
     *  @param ks  3rd dimension (sheet) indices of the tensor
     */
    def not (is: Array [Int], js: Array [Int], ks: Array [Int]): TensorD =
        apply (comple (is, dim), comple (js, dim2), comple (ks, dim3))
    end not

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single SCALAR element of the tensor to the given value.
     *  Usage: z(i, j, k) = x
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     *  @param k  3rd dimension (sheet) index of the tensor
     *  @param x  the value for updating the tensor at the above position
     */
    def update (i: Int, j: Int, k: Int, x: Double): Unit = v(i)(j)(k) = x

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single VECTOR of the tensor to the given vector.
     *  Usage: z(i, j) = x
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     *  @param x  the vector for updating the tensor at the above position
     */
    def update (i: Int, j: Int, x: VectorD): Unit = v(i)(j) = x.toArray

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single VECTOR of the tensor to the given vector.
     *  Usage: z(i, ?, k) = x
     *  @param i    1st dimension (row) index of the tensor
     *  @param all  use the all columns indicator ?
     *  @param k    3rd dimension (sheet) index of the tensor
     *  @param x    the vector for updating the tensor at the above position
     */
    def update (i: Int, @unused all: Char, k: Int, x: VectorD): Unit =
        cfor (indices2) { j => v(i)(j)(k) = x(j)}
    end update

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single VECTOR of the tensor to the given vector.
     *  Usage: z(?, j, k) = x
     *  @param all  use the all rows indicator ?
     *  @param j    2nd dimension (column) index of the tensor
     *  @param k    3rd dimension (sheet) index of the tensor
     *  @param x    the vector for updating the tensor at the above position
     */
    def update (@unused all: Char, j: Int, k: Int, x: VectorD): Unit =
        cfor (indices) { i => v(i)(j)(k) = x(i)}
    end update

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single MATRIX of the tensor (for ROW i) to the given matrix.
     *  Usage: z(i) = x
     *  @param i  1st dimension (row) index of the tensor
     *  @param x  the matrix for updating the tensor at the above position
     */
    def update (i: Int, x: MatrixD): Unit =
        cfor (indices2) { j => v(i)(j) = x(j).toArray}
    end update

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single MATRIX of the tensor (for COLUMN j) to the given matrix.
     *  Usage: z(?, j) = x
     *  @param all  use the all rows indicator ?
     *  @param j    2nd dimension (column) index of the tensor
     *  @param x    the matrix for updating the tensor at the above position 
     */
    def update (@unused all: Char, j: Int, x: MatrixD): Unit =
        cfor (indices) { i => cfor (indices3) { k => v(i)(j)(k) = x(i, k)}}
    end update

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single MATRIX of the tensor (for SHEET k) to the given matrix.
     *  Usage: z(?, ?, k) = x
     *  @param all   use the all rows indicator ?
     *  @param all2  use the all columns indicator ?
     *  @param k     the 3rd dimension (sheet) index of the tensor
     *  @param x     the matrix for updating the tensor at the above position
     */
    def update (@unused all: Char, @unused all2: Char, k: Int, x: MatrixD): Unit =
        cfor (indices) { i => cfor (indices2) { j => v(i)(j)(k) = x(i, j)}}
    end update

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a slice of the tensor with values from a given matrix.
     *  @param ir      the range of rows in the tensor to update.
     *  @param all2    a character indicating all columns should be updated (typically '?').
     *  @param sheet   the index of the sheet in the tensor to update.
     *  @param matrix  the matrix containing the values to update the tensor with.
     *  @throws IllegalArgumentException if the dimensions of the row range and matrix do not match.
     */
    def update (ir: Range, @unused all2: Char, sheet: Int, matrix: MatrixD): Unit =
        require (ir.size == matrix.dim && dim2 == matrix.dim2,
                 "Dimensions do not match the specified range and matrix.")

        cfor (ir.indices) { i =>
            cfor (indices2) { j => v(ir.start + i)(j)(sheet) = matrix(i, j) }
        } // cfor
    end update

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a slice of the tensor with values from a given 3D block (matrix over multiple sheets).
     *  @param all       a character indicating all rows should be updated (typically '?').
     *  @param all2       a character indicating all columns should be updated (typically '?').
     *  @param kr         the range of sheets in the tensor to update.
     *  @param tensorBlk  the 3D block (rows x columns x sheets) containing the values to update the tensor with.
     *  @throws IllegalArgumentException if the dimensions of the tensor block do not match the tensor's dimensions.
     */
    def update (@unused all: Char, @unused all2: Char, kr: Range, tensorBlk: TensorD): Unit =
        require (dim == tensorBlk.dim && dim2 == tensorBlk.dim2,
                 s"Row and column dimensions do not match: tensor.dim = $dim, $dim2; tensorBlk.dim = ${tensorBlk.dim}, ${tensorBlk.dim2}.")
        require (kr.size == tensorBlk.dim3,
                 s"Sheet dimensions do not match: kr.size = ${kr.size}, tensorBlk.dim3 = ${tensorBlk.dim3}.")

        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (kr.indices) { k => v(i)(j)(kr.start + k) = tensorBlk(i, j, k) }
            } // cfor
        } // cfor
    end update

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a slice of the tensor with values from a given 3D block.
     *  @param ir         the range of rows in the tensor to update.
     *  @param jr         the range of columns in the tensor to update.
     *  @param kr         the range of sheets in the tensor to update.
     *  @param tensorBlk  the 3D block containing the values to write.
     *  @throws IllegalArgumentException if the dimensions do not match.
     */
    def update (ir: Range, jr: Range, kr: Range, tensorBlk: TensorD): Unit =
        require (ir.size == tensorBlk.dim,
            s"Row dimensions do not match: ir.size = ${ir.size}, tensorBlk.dim = ${tensorBlk.dim}.")
        require (jr.size == tensorBlk.dim2,
            s"Column dimensions do not match: jr.size = ${jr.size}, tensorBlk.dim2 = ${tensorBlk.dim2}.")
        require (kr.size == tensorBlk.dim3,
            s"Sheet dimensions do not match: kr.size = ${kr.size}, tensorBlk.dim3 = ${tensorBlk.dim3}.")

        val i0 = ir.start
        val j0 = jr.start
        val k0 = kr.start

        cfor (ir.indices) { ii =>
            cfor (jr.indices) { jj =>
                cfor (kr.indices) { kk =>
                    v(i0 + ii)(j0 + jj)(k0 + kk) = tensorBlk(ii, jj, kk)
                } // cfor
            } // cfor
        } // cfor
    end update

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all the tensor element values to x.
     *  @param x  the value to set all elements to
     */
    def set (x: Double): Unit = 
        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => v(i)(j)(k) = x }
            } // cfor
        } // cfor
    end set

// Replace element operations: addition, subtraction, multiplication
// with Generalized function to perform element-wise operations with broadcasting.
// Supports TensorD, MatrixD, VectorD, and Double.

    type Broadcastable = TensorD | MatrixD | VectorD | Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    private def elementWiseDispatch (b: Broadcastable, op: (Double, Double) => Double,
                                     inPlace: Boolean = false): TensorD =
        b match
            case t: TensorD => elementWiseOp (t, op, inPlace)
            case m: MatrixD => broadcastAndApply (m, op, inPlace)
            case v: VectorD => broadcastAndApply (v, op, inPlace)
            case s: Double  => elementWiseScalarOp (s, op, inPlace)
    end elementWiseDispatch

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generalized function to perform element-wise operations with broadcasting.
     *  @param that  the input tensor/matrix/vector/double
     *  @param op    the operation (e.g., addition, subtraction, multiplication)
     */
    private def elementWiseOp (that: TensorD, op: (Double, Double) => Double,
                               inPlace: Boolean = false): TensorD =
        val outShape = broadcastShapes (shape, that.shape)
        val left     = broadcastTo (this, outShape)
        val right    = broadcastTo (that, outShape)

        val (d1, d2, d3) = (outShape(0), outShape(1), outShape(2))
        val c =
            if inPlace then
                require (outShape == shape,
                    s"In-place op not allowed when broadcast changes shape: $shape -> $outShape")
                this
            else
                new TensorD (d1, d2, d3)

        cfor (0, d1) { i =>
            cfor (0, d2) { j =>
                cfor (0, d3) { k => c(i, j, k) = op(left(i, j, k), right(i, j, k)) }
            } // Vfor
        } // cfor
        c
    end elementWiseOp

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    private def elementWiseScalarOp (scalar: Double, op: (Double, Double) => Double,
                                     inPlace: Boolean = false): TensorD =
        val c =
            if inPlace then
                this
            else
                new TensorD (dim, dim2, dim3)

        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => c(i, j, k) = op(this (i, j, k), scalar) }
            } // cfor
        } // cfor
        c
    end elementWiseScalarOp

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generalized function to broadcast MatrixD or VectorD and apply an element-wise operation.
     *  @param b   the matrix or vector
     *  @param op  the operation (e.g., addition, subtraction)
     */
    private def broadcastAndApply (b: MatrixD | VectorD, op: (Double, Double) => Double,
                                   inPlace: Boolean): TensorD =
        val broadcastedTensor = b match
            case m: MatrixD =>
                val shape = broadcastShapes (this.shape, List(m.dim, m.dim2, 1))
                broadcastMatrix (m, Some((shape(0), shape(1), shape(2))))
            case v: VectorD =>
                val shape = broadcastShapes (this.shape, List(1, v.dim, 1))
                broadcastVector (v, 0, Some ((shape(0), shape(1), shape(2))))

        elementWiseOp (broadcastedTensor, op, inPlace)
    end broadcastAndApply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Element-wise Operations
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    inline def + (b: Broadcastable): TensorD  = elementWiseDispatch (b, _ + _)

    inline def - (b: Broadcastable): TensorD  = elementWiseDispatch (b, _ - _)

    inline def * (b: Broadcastable): TensorD  = elementWiseDispatch (b, _ * _)

    inline def / (b: Broadcastable): TensorD  = elementWiseDispatch (b, _ / _)

    // FIX: Implement TensorD *~ for generalized tensor-tensor multiplication (tensordot)
    inline def *~ (b: Broadcastable): TensorD = elementWiseDispatch (b, _ * _)

    inline def ~^ (b: Broadcastable): TensorD = elementWiseDispatch (b, math.pow)

    inline def max (that: TensorD): TensorD   = elementWiseOp (that, math.max)

    inline def min (that: TensorD): TensorD   = elementWiseOp (that, math.min)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Element-wise Operations (In-Place)
    // Avoiding for now to prevent accidental data modification in Autograd
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

//  inline def += (b: Broadcastable): TensorD  = elementWiseDispatch (b, _ + _, inPlace = true)
//
//  inline def -= (b: Broadcastable): TensorD  = elementWiseDispatch (b, _ - _, inPlace = true)
//
//  inline def *= (b: Broadcastable): TensorD  = elementWiseDispatch (b, _ * _, inPlace = true)
//
//  inline def /= (b: Broadcastable): TensorD  = elementWiseDispatch (b, _ / _, inPlace = true)
//
//  FIX: Implement TensorD *~ for generalized tensor-tensor multiplication (tensordot)
//  inline def *~= (b: Broadcastable): TensorD = elementWiseDispatch (b, _ * _, inPlace = true)
//
//  inline def ~^= (b: Broadcastable): TensorD = elementWiseDispatch (b, math.pow, inPlace = true)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Scalar Reductions
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sum of all elements.
     */
    def sum: Double =
        var total = 0.0
        cfor (0, dim) { i =>
            cfor (0, dim2) { j =>
                cfor (0, dim3) { k => total += this (i, j, k) }
            } // cfor
        } // cfor
        total
    end sum

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Mean of all elements.
     */
    def mean: Double = sum / (dim * dim2 * dim3).toDouble

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Variance of all elements.
     */
    def variance: Double =
        val mu = mean
        map_ (x => math.pow (x - mu, 2)).sum / (dim * dim2 * dim3).toDouble

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Standard deviation of all elements.
     */
    def std: Double = math.sqrt (variance)

    def normFSq: Double = map_ (x => x * x).sum

    def normF: Double = math.sqrt (normFSq)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Axis-wise Reductions
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    def meanAlongAxis (axis: Int): TensorD = TensorD.meanAlongAxis (this, axis)

    def stdAlongAxis (axis: Int): TensorD = TensorD.stdAlongAxis (this, axis)

    def sumAlongAxis (axis: Int): TensorD = TensorD.sumAlongAxis (this, axis)

    def varianceAlongAxis (axis: Int): TensorD = TensorD.varianceAlongAxis (this, axis)

    def standardize (axis: Int): TensorD = TensorD.standardize (this, axis)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Unary and Scalar Element-wise Operations
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Absolute value.
     */
    def abs: TensorD = map_ (math.abs)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Ceil each element.
     */
    def ceil: TensorD = map_ (math.ceil)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clip elements between min and max.
     */
    def clipByValue (minVal: Double, maxVal: Double): TensorD =
        require (minVal <= maxVal, s"clipByValue: minVal ($minVal) should not be greater than maxVal ($maxVal)")
        map_ (x => math.max (minVal, math.min (maxVal, x)))
    end clipByValue

    def clipByNorm (maxNorm: Double): TensorD =
        val currentNorm = normF
        if currentNorm > maxNorm and currentNorm > 0.0 then
            this * (maxNorm / currentNorm)
        else
            this
    end clipByNorm

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Exponential.
     */
    def exp: TensorD = map_ (math.exp)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Floor each element.
     */
    def floor: TensorD = map_ (math.floor)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Log base e (natural log).
     */
    def log: TensorD =
        map_ (v => if v > 0 then math.log (v)
                   else throw new ArithmeticException (s"log is not defined for non-positive value: $v"))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Log base 10. */
    def log10: TensorD =
        map_ (v => if v > 0 then math.log10 (v)
                   else throw new ArithmeticException (s"log10 is not defined for non-positive value: $v"))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Log base-n. */
    def logBase (base: Double): TensorD =
        map_ (v => if v > 0 then math.log (v) / math.log (base)
                   else throw new ArithmeticException (s"log base $base is undefined for non-positive: $v"))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Max with a scalar.
     */
    def maxScalar (s: Double): TensorD = map_(x => math.max (x, s))

    def maxValue: Double = flattenToVector.reduce (math.max)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Min with a scalar.
     */
    def minScalar (s: Double): TensorD = map_(x => math.min (x, s))

    def minValue: Double = flattenToVector.reduce (math.min)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reciprocal.
     */
    def reciprocal: TensorD =
        map_ (v => if v != 0.0 then 1.0 / v
                   else throw new ArithmeticException ("Division by zero in reciprocal"))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Round each element.
     */
    def round: TensorD = map_ (x => math.round (x).toDouble)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sign of each element (-1, 0, 1).
     */
    def sign: TensorD = map_ (math.signum)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Square root.
     */
    def sqrt: TensorD =
        map_ (v => if v >= 0 then math.sqrt (v)
                   else throw new ArithmeticException (s"sqrt is not defined for negative value: $v"))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Raise each element to an integer power.
     */
    def ~^ (s: Int): TensorD = elementWiseScalarOp (s, math.pow)
//  def ** (s: Int): TensorD = elementWiseScalarOp (s, math.pow)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Negate the tensor (unary `-`).
     */
    inline def unary_- : TensorD = this * (-1.0)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Activation Functions
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    def id: TensorD = TensorD.id_ (this)

    def reLU: TensorD = TensorD.reLU_ (this)

    def lreLU (alpha: Double = 0.2): TensorD = TensorD.lreLU_ (this, alpha)

    def eLU(alpha: Double = 1.0): TensorD = TensorD.eLU_ (this, alpha)

    def tanh: TensorD = TensorD.tanh_ (this)

    def sigmoid: TensorD = TensorD.sigmoid_ (this)

    def gaussian: TensorD = TensorD.gaussian_ (this)

    def geLU: TensorD = TensorD.geLU_ (this)

    def softmax: TensorD = TensorD.softmax_ (this)

    def logit: TensorD = TensorD.logit_ (this)

    def logistic (a: Double = 1.0, b: Double = 1.0, c: Double = 1.0): TensorD =
        TensorD.logistic_ (this, a, b, c)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Other Operations
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    def zerosLike: TensorD = TensorD.zerosLike (this)

    def onesLike: TensorD = TensorD.onesLike (this)

    def fullLike (value: Double): TensorD = TensorD.fullLike (this, value)

    override def equals (obj: Any): Boolean =
        obj match
            case that: TensorD if this.dims == that.dims =>
                indices.forall { i =>
                    indices2.forall { j =>
                        indices3.forall { k => math.abs (this(i, j, k) - that(i, j, k)) <= 1e-9 }
                    } // forall
                } // forall
            case _ => false
    end equals

    override def hashCode (): Int =
        dims.hashCode * 31 +
            (for i <- indices; j <- indices2; k <- indices3 yield this (i, j, k).##).##
    end hashCode

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the dot product of two vectors stored as tensors.
     *  Both tensors must have shape (n, 1, 1), representing row vectors.
     *  Leverages the existing VectorD dot method.
     *  Returns a scalar wrapped in a tensor of shape (1, 1, 1).
     *  @param b the tensor to take the dot product with
     */
    infix def dot (b: TensorD): TensorD =
        val (mA, nA, dA) = dims
        val (mB, nB, dB) = b.dims
        require (nA == 1 && dA == 1 && nB == 1 && dB == 1 && mA == mB,
            s"dot is only for vectors with shape (1, n, 1). Got shapes ${dims} and ${b.dims}")

        // Extract the row from each tensor as a VectorD.
        val vA = new VectorD (mA, Array.tabulate (mA)(i => this(i, 0, 0)))
        val vB = new VectorD (mB, Array.tabulate (mB)(i => b(i, 0, 0)))

        TensorD ((1, 1, 1), vA dot vB)
    end dot

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Matrix‑matrix product for batch‑first tensors.
     * Expected shapes
     *   A : (1, m, k)
     *   B : (1, k, n)
     * Returns
     *   C : (1, m, n)
     */
    infix def matmul (b: TensorD): TensorD =
        val (bA, _, kA) = dims
        val (bB, kB, _) = b.dims

        require (bA == 1 && bB == 1 && kA == kB,
            s"matmul requires shapes (1,m,k) × (1,k,n); got $dims × ${b.dims}")

        bmm (b)
    end matmul

    def slice (i: Int): MatrixD = this(i)

    def setSlice (i: Int, d: MatrixD): Unit = this.update (i, d)

    infix def bmm (b: TensorD): TensorD =
        val (dA, mA, kA) = dims
        val (dB, kB, nB) = b.dims

        require(kA == kB,
            s"BMM requires matching inner dims: got kA=$kA vs kB=$kB")

        val dOut =
            if dA == dB then dA
            else if dA == 1 then dB
            else if dB == 1 then dA
            else throw IllegalArgumentException (s"BMM batch dims must match or one must be 1; got ($dA, $dB)")

        val out = TensorD.fill (dOut, mA, nB, 0.0)

        val a0 = slice (0)
        val c0 = b.slice (0)

        for b_ <- 0 until dOut do
            val a = if dA == 1 then a0 else slice (b_)
            val c = if dB == 1 then c0 else b.slice (b_)
            out.setSlice (b_, a * c)
        out
    end bmm

// Comments out old implementation of element-wise operations

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this tensor and tensor b.
     *  @param b  the tensor to add (requires leDimensions)
    def + (b: TensorD): TensorD =
        val c = new TensorD (dim, dim2, dim3)
        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => c.v(i)(j)(k) = v(i)(j)(k) + b.v(i)(j)(k) }
            } // cfor
        } // cfor
        c
    end +
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this tensor and scalar s.
     *  @param s  the scalar to add
    def + (s: Double): TensorD =
        val c = new TensorD (dim, dim2, dim3)
        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => c.v(i)(j)(k) = v(i)(j)(k) + s }
            } // cfor
        } // cfor
        c
    end +
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** From this tensor subtract tensor b.
     *  @param b  the tensor to add (requires leDimensions)
    def - (b: TensorD): TensorD =
        val c = new TensorD (dim, dim2, dim3)
        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => c.v(i)(j)(k) = v(i)(j)(k) - b.v(i)(j)(k) }
            } // cfor
        } // cfor
        c
    end -
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** From this tensor subtract scalar s.
     *  @param s  the scalar to add
    def - (s: Double): TensorD =
        val c = new TensorD (dim, dim2, dim3)
        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => c.v(i)(j)(k) = v(i)(j)(k) - s }
            } // cfor
        } // cfor
        c
    end -
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this tensor by scalar s.
     *  @param s  the scalar to multiply by
    def * (s: Double): TensorD =
        val c = new TensorD (dim, dim2, dim3)
        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => c.v(i)(j)(k) = v(i)(j)(k) * s }
            } // cfor
        } // cfor
        c
    end *
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply element-wise (Hadamard product) this tensor by tensor b.
     *  @param b  the tensor to add (requires leDimensions)
    def *~ (b: TensorD): TensorD =
        val c = new TensorD (dim, dim2, dim3)
        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => c.v(i)(j)(k) = v(i)(j)(k) * b.v(i)(j)(k) }
            } // cfor
        } // cfor
        c
    end *~ 
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply (multi-linear product) this tensor by three matrices b, c and d.
     *      this * (a, b, c)
     *  @see www.stat.uchicago.edu/~lekheng/work/icm1.pdf - equation 15.1
     *  @param b  the first matrix to multiply by (requires leDimensions)
     *  @param c  the second matrix to multiply by (requires leDimensions)
     *  @param d  the third matrix to multiply by (requires leDimensions)
     */
    def * (b: MatrixD, c: MatrixD, d: MatrixD): TensorD =
        val (m1, n1) = (b.dim, b.dim2)
        val (m2, n2) = (c.dim, c.dim2)
        val (m3, n3) = (d.dim, d.dim2)
        if n1 > dim2 || n2 > dim2 || n3 > dim3 then flaw ("*", "dimensions don't match")

        val e = new TensorD (m1, m2, m3)
        cfor (b.indices) { i =>
            cfor (c.indices) { j =>
                cfor (d.indices) { k =>
                    var sum = 0.0
                    cfor (b.indices2) { l1 =>
                        cfor (c.indices2) { l2 =>
                            cfor (d.indices2) { l3 => sum += b(i, l1) * c(j, l2) * d(k, l3) * v(l1)(l2)(l3) }
                        } // cfor
                    } // cfor
                    e.v(i)(j)(k) = sum
                } // cfor
            } // cfor
        } // cfor
        e
    end *

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map each row of this tensor by applying function f to each row matrix and
     *  returning the collected result as a matrix.
     *  @param f  the matrix to vector function to apply
     */
    def map (f: FunctionM2V): MatrixD =
        val a = Array.ofDim [VectorD] (dim)
        cfor (0, dim) { i => a(i) = f(apply(i)) }
        MatrixD (a.toIndexedSeq)
    end map

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map each row of this tensor by applying function f to each row matrix and
     *  returning the collected result as a tensor.
     *  @param f  the matrix to matrix function to apply
     */
    def mmap (f: FunctionM2M): TensorD =
        val a = Array.ofDim [MatrixD] (dim)
        cfor (0, dim) { i => a(i) = f(apply(i)) }
        TensorD (a.toIndexedSeq)
    end mmap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map each element of this tensor by applying function f to each element and
     *  returning the collected result as a tensor.
     *  @param f  the scalar to scalar function to apply
     */
    def map_ (f: FunctionS2S): TensorD =
        val x = new TensorD (dim, dim2, dim3)
        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => x.v(i)(j)(k) =  f(v(i)(j)(k)) }
            } // cfor
        } // cfor
        x
    end map_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the shape of this tensor as a list of integers.
     *  @return the shape of the tensor
     */
    inline def shape: List [Int] = _shape

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Flatten this tensor in row-major fashion, returning a matrix containing
     *  all the elements from the tensor.
     *  @return a `MatrixD` containing all elements of the tensor in row-major order.
     */
    def flatten: MatrixD =
        val a = Array.ofDim [Double] (dim * dim2, dim3)
        var k = 0
        cfor (indices) { i =>
            val v_i = v(i)
            var j = 0
            cfor (j < dim2, j += 1) { a(k) = v_i(j); k += 1 }
        } // cfor
        new MatrixD (a.length, a(0).length, a)
    end flatten

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Flatten this 3D tensor into a 1D vector in row-major order.
     *  This method iterates through all elements of the tensor and stores them
     *  sequentially in a 1D array, which is then wrapped in a `VectorD` object.
     *  @return a `VectorD` containing all elements of the tensor in row-major order.
     */
    def flattenToVector: VectorD =
        val arr = new Array [Double] (dim * dim2 * dim3)
        var idx = 0
        cfor (0, dim) { i =>
            cfor (0, dim2) { j =>
                cfor (0, dim3) { k => arr(idx) = v(i)(j)(k); idx += 1 }
            } // cfor
        } // cfor
        new VectorD (arr.length, arr)
    end flattenToVector

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reshape this tensor to a new shape while preserving the order of elements.
     *  This method ensures that the total number of elements remains the same
     *  and rearranges the elements into the specified dimensions.
     *  @param newShape  the desired shape as a sequence of three integers
     *  @return a new `TensorD` with the specified shape
     *  @throws IllegalArgumentException if the total number of elements does not match
     */
    def reshape (newShape: Seq [Int]): TensorD =
        val (newDim, newDim2, newDim3) = (newShape(0), newShape(1), newShape(2))

        require (dim * dim2 * dim3 == newDim * newDim2 * newDim3,
            s"reshape requires the same number of elements: current=${dim * dim2 * dim3}, new=${newDim * newDim2 * newDim3}")

        val out = TensorD.fill (newDim, newDim2, newDim3, 0.0)

        val flat = flattenToVector
        var idx = 0
        cfor (0, newDim) { i =>
            cfor (0, newDim2) { j =>
                cfor (0, newDim3) { k => out(i, j, k) = flat(idx); idx += 1 }
            } // cfor
        } // cfor
        out
    end reshape

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Permute the axes of this tensor according to the specified order.
     *  This method rearranges the dimensions of the tensor based on the given
     *  permutation of axes. The input `axes` specifies the new order of the axes.
     *  For example, if the original tensor has shape (a, b, c) and `axes` is
     *  Seq(1, 2, 0), the resulting tensor will have shape (b, c, a).
     *  @param axes  the sequence specifying the new order of the axes
     *  @return a new `TensorD` with permuted axes
     *  @throws IllegalArgumentException if `axes` does not contain a valid permutation
     */
    def permute (axes: Seq [Int]): TensorD =
        require (axes.length == 3 && axes.sorted == Seq (0, 1, 2),
            s"permute requires a valid permutation of axes 0, 1, 2, got: $axes")

        val oldShape = shape
        val newShape = axes.map (oldShape)

        val out = TensorD.fill (newShape(0), newShape(1), newShape(2), 0.0)

        // Compute inverse axes: tells where each new index came from
        val invAxes = Array.ofDim [Int](3)
        cfor (0, 3) { i => invAxes (axes(i)) = i }

        cfor (0, newShape(0)) { i =>
            cfor (0, newShape(1)) { j =>
                cfor (0, newShape(2)) { k =>
                    val newIdx = Array (i, j, k)
                    val origI = newIdx(invAxes(0))
                    val origJ = newIdx(invAxes(1))
                    val origK = newIdx(invAxes(2))
                    out(i, j, k) = v(origI)(origJ)(origK)
                } // cfor
            } // cfor
        } // cfor
        out
    end permute

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transpose/swap two axes of this tensor.
     *  @param i  first axis index (0..2)
     *  @param j  second axis index (0..2)
     *  @return a new TensorD with axes i and j swapped
     *  @throws IllegalArgumentException if an axis index is out of range
     */
    def transpose (i: Int, j: Int): TensorD =
        val axes = Array (0, 1, 2)
        val tmp = axes(i)
        axes(i) = axes(j)
        axes(j) = tmp
        permute (axes.toIndexedSeq)
    end transpose

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether the dimensions of this tensor are less than or equal to
     *  le those of the other tensor b.
     *  @param b  the other matrix
     */
    def leDimensions (b: TensorD): Boolean =
        dim <= b.dim && dim2 <= b.dim2 && dim3 <= b.dim3
    end leDimensions

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this tensor to a matrix where all the elements have integer values.
     */
    def toInt: TensorD =
        val x = new TensorD (dim, dim2, dim3)
        cfor (indices) { i =>
            cfor (indices2) { j =>
                cfor (indices3) { k => x.v(i)(j)(k) = math.round (v(i)(j)(k)).toDouble }
            } // cfor
        } // cfor
        x
    end toInt

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this tensor to a string with a double line break after each sheet
     *  and a single line break after each row.
     */
    override def toString: String = 
        val sb = new StringBuilder ("\nTensorD (")
        if dim == 0 then return sb.append (")").mkString
        cfor (indices3) { k =>
            cfor (indices) { i =>
                cfor (indices2) { j => sb.append (s"${v(i)(j)(k)}, ") }
                sb.append ("\n" + TAB)
            } // cfor
            sb.append ("\n" + TAB)
        } // cfor
        sb.replace (sb.length-5, sb.length, ")").mkString
    end toString

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this tensor to a string with a line break after each sheet.
     */
    def toString2: String = 
        val sb = new StringBuilder ("\nTensorD( ")
        if dim == 0 then return sb.append (")").mkString
        cfor (indices) { i =>
            cfor (indices2) { j =>
                sb.append (stringOf (v(i)(j)) + ", ")
                if j == dim2 - 1 then sb.replace (sb.length - 1, sb.length, "\n\t")
            } // cfor
        } // cfor
        sb.replace (sb.length-3, sb.length, ")").mkString
    end toString2

end TensorD


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TensorD` companion object provides factory methods for the `TensorD` class.
 */
object TensorD:

//  private val flaw = flawf ("TensorD")                               // flaw function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a tensor from the scaler argument list x.
     *  @param n1  the first dimension
     *  @param n2  the second dimension
     *  @param n3  the third dimension
     *  @param x   the list/vararg of scacollection.immutable.IndexedSeq [MatrixD]lars
     */
    def apply (n: (Int, Int, Int), x: Double*): TensorD =
        val t = new TensorD (n._1, n._2, n._3)
        var l = 0
        cfor (0, n._3) { k => cfor (0, n._1) { i => cfor (0, n._2) { j =>
            t(i, j, k) = x(l)
            l += 1
        }}} // cfor
        t
    end apply 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a tensor from the vector argument list x.
     *  @param n   the first dimension
     *  @param vs  the list/vararg of vectors
     */
    def apply (n: Int, vs: VectorD*): TensorD =
        val t = new TensorD (n, vs.length, vs(0).dim)
        var l = 0
        cfor (t.indices) { i => cfor (t.indices2) { j =>
            t(i, j) = vs(l)
            l += 1
        }} // cfor
        t
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a tensor from the vector argument list x.
     *  @param n   the first dimension
     *  @param vs  the indexed sequence of vectors
     */
    def apply (n: Int, vs: IndexedSeq [VectorD]): TensorD =
        val t = new TensorD (n, vs.length, vs(0).dim)
        var l = 0
        cfor (t.indices) { i => cfor (t.indices2) { j =>
            t(i, j) = vs(l)
            l += 1
        }} // cfor
        t
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a tensor from the vector argument list x.
     *  @param n   the first dimension
     *  @param vs  the indexed sequence of vectors
     */
    def apply (n: Int, vs: collection.immutable.IndexedSeq [VectorD]): TensorD =
        val t = new TensorD (n, vs.length, vs(0).dim)
        var l = 0
        cfor (t.indices) { i => cfor (t.indices2) { j =>
            t(i, j) = vs(l)
            l += 1
        }} // cfor
        t
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tensor from a variable argument list of matrices (row-wise).
     *  Use transpose to make it column-wise.
     *  @param vs  the vararg list of matrices
     */
    def apply (vs: MatrixD*): TensorD =
        val (m, n, p) = (vs.length, vs(0).dim, vs(0).dim2)
        val a = Array.ofDim [Array [Array [Double]]] (m)
        cfor (vs.indices) { i => a(i) = vs(i).v}
        new TensorD (m, n, p, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tensor from an mutable `IndexedSeq` of matrices (row-wise).
     *  Use transpose to make it column-wise.
     *  @param vs  the indexed sequence of matrices
     */
    def apply (vs: IndexedSeq [MatrixD]): TensorD =
        val (m, n, p) = (vs.length, vs(0).dim, vs(0).dim2)
        val a = Array.ofDim [Array [Array [Double]]] (m)
        cfor (vs.indices) { i => a(i) = vs(i).v}
        new TensorD (m, n, p, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tensor from an immutable `IndexedSeq` of matrices (row-wise),
     *  as produce by for yield.  Use transpose to make it column-wise.
     *  @param vs  the indexed sequence of matrices
     */
    def apply (vs: collection.immutable.IndexedSeq [MatrixD]): TensorD =
        val (m, n, p) = (vs.length, vs(0).dim, vs(0).dim2)
        val a = Array.ofDim [Array [Array [Double]]] (m)
        cfor (vs.indices) { i => a(i) = vs(i).v}
        new TensorD (m, n, p, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tensor of dimensions dim by dim2 by dim3 where all elements equal
     *  to the given value.
     *  @param dim    the row dimension
     *  @param dim2   the column dimension
     *  @param dim2   the sheet dimension
     *  @param value  the given value to assign to all elements
     */
    def fill (dim: Int, dim2: Int, dim3: Int, value: Double): TensorD =
        val a = Array.fill (dim, dim2, dim3)(value)
        new TensorD (dim, dim2, dim3, a)
    end fill

    // ----------------------------------------------------------------
    // Additional methods for autograd purposes
    // ----------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a new tensor filled with zeros, having the same dimensions as the given tensor.
     *  @param tensor  the tensor to mimic in dimensions.
     *  @return A new tensor filled with zeros.
     */
    def zerosLike (tensor: TensorD): TensorD = fill (tensor.dim, tensor.dim2, tensor.dim3, 0.0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a new tensor filled with ones, based on the given dimensions tuple.
     *  @param dims  a tuple representing the shape of the tensor (dim, dim2, dim3).
     *  @return a new tensor filled with ones.
     */
    def ones (dims: (Int, Int, Int)): TensorD = fill (dims._1, dims._2, dims._3, 1.0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a new tensor filled with ones, having the same dimensions as the given tensor.
     *  @param tensor  the tensor to mimic in dimensions.
     *  @return a new tensor filled with ones.
     */
    def onesLike (tensor: TensorD): TensorD = ones (tensor.dim, tensor.dim2, tensor.dim3)

    def fullLike (t: TensorD, value: Double): TensorD = fill (t.dim, t.dim2, t.dim3, value)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sum all elements of the tensor along the specified axis.
     *  @param tensor  the tensor to sum over.
     *  @param axis    the axis along which to sum (0 = rows, 1 = columns, 2 = sheets).
     *  @return A new TensorD with the reduced dimension.
     */
    def sumAlongAxis (tensor: TensorD, axis: Int): TensorD =
        axis match
            case 0 =>                                                // sum along rows (collapse row dimension)
                val result = new TensorD (1, tensor.dim2, tensor.dim3)
                cfor (tensor.indices2) { j =>
                    cfor (tensor.indices3) { k =>
                        var sum = 0.0
                        cfor (tensor.indices) { i => sum += tensor(i, j, k) }
                        result(0, j, k) = sum
                    } // cfor
                } // cfor
                result

            case 1 =>                                                // sum along columns (collapse column dimension)
                val result = new TensorD (tensor.dim, 1, tensor.dim3)
                cfor (tensor.indices) { i =>
                    cfor (tensor.indices3) { k =>
                        var sum = 0.0
                        cfor (tensor.indices2) { j => sum += tensor(i, j, k) }
                        result(i, 0, k) = sum
                    } // cfor
                } // cfor
                result

            case 2 =>                                                // sum along sheets (collapse sheet dimension)
                val result = new TensorD (tensor.dim, tensor.dim2, 1)
                cfor (tensor.indices) { i =>
                    cfor (tensor.indices2) { j =>
                        var sum = 0.0
                        cfor (tensor.indices3) { k => sum += tensor(i, j, k) }
                        result(i, j, 0) = sum
                    } // cfor
                } // cfor
                result

            case _ => throw new IllegalArgumentException (s"Invalid axis: $axis. Must be 0, 1, or 2.")
    end sumAlongAxis

    def meanAlongAxis (x: TensorD, axis: Int): TensorD =
        require (axis >= 0 && axis < x.shape.length, s"Invalid axis: $axis")
        sumAlongAxis (x, axis) / x.shape(axis).toDouble
    end meanAlongAxis

    def varianceAlongAxis (x: TensorD, axis: Int): TensorD =
        require (axis >= 0 && axis < x.shape.length, s"Invalid axis: $axis")
        val mu = meanAlongAxis (x, axis)
        val variance = sumAlongAxis ((x - mu).map_ (v => v * v), axis) / (x.shape(axis).toDouble + 1e-8)
        variance
    end varianceAlongAxis

    def stdAlongAxis (x: TensorD, axis: Int): TensorD =
        require (axis >= 0 && axis < x.shape.length, s"Invalid axis: $axis")
        val variance = varianceAlongAxis (x, axis)
        variance.map_ (math.sqrt)
    end stdAlongAxis

    def standardize (x: TensorD, axis: Int): TensorD =
        require (axis >= 0 && axis < x.shape.length, s"Invalid axis: $axis")
        val meanVal = meanAlongAxis (x, axis)
        val stdVal = stdAlongAxis (x, axis)
        (x - meanVal) / (stdVal + 1e-8)
    end standardize

    def diag (s: Double, size: Int): TensorD =
        val tensor = new TensorD (size, size, size)
        cfor (0, size) { i => tensor(i, i, i) = s }
        tensor
    end diag

    def scalar (s: Double): TensorD = TensorD ((1, 1, 1), s)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a TensorD from a VectorD with default shape (length, 1, 1).
     *  @param v  the vector to convert
     *  @return A TensorD of shape (length, 1, 1)
     */
    def fromVector (v: VectorD, axis: Int = 0): TensorD = broadcastVector (v, axis)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a TensorD from a MatrixD with default shape (1, rows, cols).
     *  @param m  the matrix to convert
     *  @return A TensorD of shape (1, rows, cols)
     */
    def fromMatrix (m: MatrixD, shape: Option [(Int, Int, Int)] = None): TensorD =
        TensorD.broadcastMatrix (m, shape)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Broadcast a MatrixD into a 3D tensor (TensorD) in batch‑first layout.
     *  - Base tensor is created with batch=1, rows=m.dim, cols=m.dim2
     *  - We explicitly fill base(0,i,j) = m(i,j) so there’s n
     *  - If you request a larger batch, we replicate that slice across batch
     */
    def broadcastMatrix (m: MatrixD, shape: Option [(Int, Int, Int)] = None): TensorD =
        val rows = m.dim      // r
        val cols = m.dim2     // c
        val sliceSize = rows * cols

        // Flatten in **column‑major** order: for each col, for each row
        val flat = Array.tabulate (sliceSize) { idx =>
            val col = idx / rows
            val row = idx % rows
            m(row, col)
        }

        // Build the base (1 × rows × cols)
        val base = TensorD ((1, rows, cols), flat*)

        // If a larger batch is requested, replicate the same slice
        shape match
            case Some ((b, r, c)) =>
                require (r == rows && c == cols,
                    s"broadcastMatrix: ($r, $c) must match ($rows, $cols)")
                if b == 1 then base
                else
                    val outFlat = new Array [Double](b * sliceSize)
                    var off = 0
                    var bb  = 0
                    while bb < b do
                        System.arraycopy(flat, 0, outFlat, off, sliceSize)
                        off += sliceSize; bb += 1
                    end while
                    TensorD ((b, rows, cols), outFlat*)
            case None =>
                base
        end match
    end broadcastMatrix

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Broadcast a VectorD into a 3D tensor (TensorD), allowing partial broadcasting.
     *  The default shape is determined by the axis parameter:
     *  - If axis == 0, the default is a column vector: (v.dim, 1, 1)
     *  - If axis == 1, the default is a row vector:    (1, v.dim, 1)
     *  - If axis == 2, the default is a sheet vector:  (1, 1, v.dim)
     *  If a shape is provided, a base tensor is created with the default shape and then
     *  expanded to the given shape using `broadcastTo`.
     */
    def broadcastVector (v: VectorD, axis: Int = 0, shape: Option [(Int, Int, Int)] = None): TensorD =
        val data = v.toArray

        val baseShape: (Int, Int, Int) = axis match
            case 0 => (v.dim, 1, 1)   // column vector
            case 1 => (1, v.dim, 1)   // row vector
            case 2 => (1, 1, v.dim)   // sheet vector
            case _ => throw new Exception ("Axis must be 0 (column), 1 (row), or 2 (sheet)")

        val base = TensorD(baseShape, data *)

        shape match
            case Some ((d1, d2, d3)) => broadcastTo (base, List (d1, d2, d3))
            case None                => base
    end broadcastVector

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the broadcasted shape for two 3D shapes aShape and bShape.
     *  - If one dimension is 1 and the other is R, pick R.
     *  - If both are the same, pick that value.
     *  - Otherwise, throw an error for mismatched dimensions.
     */
    def broadcastShapes (aShape: List [Int], bShape: List [Int]): List [Int] =
        require (aShape.size == 3 && bShape.size == 3, "Only supports 3D shapes currently")

        // Perform broadcasting logic across all dimensions
        aShape.zip (bShape).map { case (a, b) =>
            if a == b then a
            else if a == 1 then b
            else if b == 1 then a
            else throw new IllegalArgumentException(s"Incompatible shapes: $aShape vs $bShape")
        } // map
    end broadcastShapes

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand a TensorD 'src' to 'newShape' if needed.
     *  If src.shape == newShape, just return src. Otherwise replicate data along
     *  any dimension that was 1 in src.shape but is >1 in newShape.
     */
    def broadcastTo (src: TensorD, newShape: List [Int]): TensorD =
        val oldShape = src.shape
        if oldShape == newShape then src
        else
            val (d1, d2, d3) = (newShape(0), newShape(1), newShape(2))
            val out = new TensorD (d1, d2, d3)

            cfor (0, d1) { i =>
                val iSrc = if oldShape(0) == 1 then 0 else i
                cfor (0, d2) { j =>
                    val jSrc = if oldShape(1) == 1 then 0 else j
                    cfor (0, d3) { k =>
                        val kSrc = if oldShape(2) == 1 then 0 else k
                        out(i, j, k) = src(iSrc, jSrc, kSrc)
                    } // cfor
                } // cfor
            } // cfor
            out
    end broadcastTo

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate a sequence of 3D tensors along the specified axis (0, 1, or 2).
     *  @param xs    sequence of TensorD to concatenate
     *  @param axis  dimension along which to concatenate (0, 1, or 2)
     */
    def concat(xs: Seq [TensorD], axis: Int): TensorD =
        require (xs.nonEmpty, "concat: input sequence is empty")
        require (axis >= 0 && axis <= 2, s"concat: invalid axis $axis")

        // All tensors must be 3D
        xs.foreach { x =>
            require (x.shape.length == 3,
            s"concat: all tensors must be 3D, got shape ${x.shape}") }

        // Shapes of all tensors
        val shapes = xs.map (_.shape)

        val B0 = shapes.head(0)
        val T0 = shapes.head(1)
        val D0 = shapes.head(2)

        // Validate shapes based on axis
        axis match
        case 0 => // concat along rows
            xs.foreach { x =>
                val s = x.shape
                require (s(1) == T0 && s(2) == D0,
                s"concat axis=0: T and D must match. Expected ($T0,$D0), got (${s(1)},${s(2)})") }

        case 1 => // concat along sequence
            xs.foreach { x =>
                val s = x.shape
                require (s(0) == B0 && s(2) == D0,
                s"concat axis=1: B and D must match. Expected ($B0,$D0), got (${s(0)},${s(2)})") }

        case 2 => // concat along features
            xs.foreach { x =>
                val s = x.shape
                require (s(0) == B0 && s(1) == T0,
                s"concat axis=2: B and T must match. Expected ($B0,$T0), got (${s(0)},${s(1)})") }
        end match

        // Compute output shape
        val B_out = axis match
            case 0 => shapes.map (_(0)).sum
            case 1 => B0
            case 2 => B0
        val T_out = axis match
            case 0 => T0
            case 1 => shapes.map (_(1)).sum
            case 2 => T0
        val D_out = axis match
            case 0 => D0
            case 1 => D0
            case 2 => shapes.map (_(2)).sum

        // Allocate result and copy blocks into output
        val out = TensorD.fill (B_out, T_out, D_out, 0.0)

        var cursor = 0
        axis match
        // axis = 0: grow batch dimension
        case 0 =>
            for x <- xs do
                val Bi = x.shape.head
                out(cursor until cursor + Bi, 0 until T0, 0 until D0) = x
                cursor += Bi
        // axis = 1: grow sequence dimension
        case 1 =>
            for x <- xs do
                val Ti = x.shape(1)
                out(0 until B0, cursor until cursor + Ti, 0 until D0) = x
                cursor += Ti
        // axis = 2: grow feature dimension
        case 2 =>
            for x <- xs do
                val Di = x.shape(2)
                out(0 until B0, 0 until T0, cursor until cursor + Di) = x
                cursor += Di
        end match
        out
    end concat

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Activations
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    def id_ (yp: TensorD): TensorD = yp

    def reLU_ (yp: TensorD): TensorD = yp.map_ (reLU)

    def lreLU_ (yp: TensorD, alpha: Double): TensorD = { setA (alpha); yp.map_ (lreLU) }

    def eLU_ (yp: TensorD, alpha: Double): TensorD = { setA2 (alpha); yp.map_ (eLU) }

    def tanh_ (yp: TensorD): TensorD = yp.map_ (math.tanh)

    def sigmoid_ (yp: TensorD): TensorD = yp.map_ (sigmoid)

    def gaussian_ (yp: TensorD): TensorD = yp.map_ (gaussian)

    def geLU_ (yp: TensorD): TensorD = yp.map_ (geLU)

    def softmax_ (yp: TensorD): TensorD = tensorize (ActivationFun.softmax_)(yp)

    def logit_ (yp: TensorD): TensorD = yp.map_ (logit)

    def logistic_ (yp: TensorD, a: Double, b: Double, c: Double): TensorD = yp.map_ (logistic(_, a, b, c))

    // Other operations

    def max (x: TensorD, y: TensorD): TensorD = x.max (y)

    def min (x: TensorD, y: TensorD): TensorD = x.min (y)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the cross-correlation tensor for the given data matrix for up to
     *  maxLags.
     *  @param x        the given data matrix (row are instances, columns are variables)
     *  @param maxLags  the maximum number of lags to consider
     *
    def crossCorr (x: MatrixD, maxLags: Int = 10): TensorD =
        val n = x.dim2
        if 2 * maxLags >= x.dim then flaw ("crossCorr", "not enough data for maxLags = $maxLags") 
        val ccorr = new TensorD (maxLags+1, n, n)
        for l <- 0 to maxLags do ccorr(l) = x.laggedCorr (l)
        ccorr
    end crossCorr
     */

end TensorD


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tensorDTest` main function is used to test the `TensorD` class.
 *  > runMain scalation.mathstat.tensorDTest
 */
@main def tensorDTest (): Unit =

    val s = 2.0
    val a = new TensorD (2, 3, 2)
    val b = new TensorD (2, 3, 2)
    //                                            row column sheet
    val c = TensorD ((2, 3, 2), 1,  2,  3,       // 0   0-2     0
                                4,  5,  6,       // 1   0-2     0

                                7,  8,  9,       // 0   0-2     1
                               10, 11, 12)       // 1   0-2     1

    for i <- 0 until 2; j <- 0 until 3; k <- 0 until 2 do
        val sum = i + j + k
        a(i, j, k) = sum
        b(i, j, k) = sum
    end for

    println ("s          = " + s)
    println ("a          = " + a)
    println ("b          = " + b)
    println ("c          = " + c)
    println ("c(0)       = " + c(0))
    println ("c(0, 0)    = " + c(0, 0))
    println ("c(0, 0, 0) = " + c(0, 0, 0))

    banner ("Test operators")
    println ("a + b   = " + (a + b))
    println ("a + s   = " + (a + b))
    println ("a - b   = " + (a - b))
    println ("a - s   = " + (a - s))
    println ("c * s   = " + c * s)
    println ("a *~ c  = " + a *~ c)

    val x = MatrixD ((2, 2), 1, 2,
                             3, 4)
    val y = MatrixD ((2, 3), 1, 2, 3,
                             4, 5, 6)
    val z = MatrixD ((2, 2), 5, 6,
                             7, 8)

    println ("c * (x, y, z) = " + c * (x, y, z))

    banner ("Test slice")
    println ("c = " + c)
    println ("slice row 0:1 = " + c((0, 1)))

    println ("slice row col: 0:1, 0:2 = " + c((0, 1), (0, 2)))
    println ("slice col:    null, 0:2 = " + c(null,   (0, 2)))

    println ("slice row col sheet: 0:1, 0:2,  0:1 = " + c((0, 1), (0, 2), (0, 1)))
    println ("slice sheet:        null, null, 0:1 = " + c(null,   null,   (0, 1)))
    println ("slice row sheet:     0:1, null, 0:1 = " + c((0, 1), null,   (0, 1)))
    println ("slice col sheet     null, 0:2,  0:1 = " + c(null,   (0, 2), (0, 1)))

    banner ("Test select")
    println ("c = " + c)
    println ("select row 0 = " + c(Array [Int] (0)))

    println ("select row col: 0, 0,2 = " + c(Array [Int] (0), Array [Int] (0, 2)))
    println ("select col:  null, 0,2 = " + c(null,     Array [Int] (0, 2)))

    println ("select row col sheet: 0,  0,2, 1 = " + c(Array [Int] (0), Array [Int] (0, 2), Array [Int] (1)))
    println ("select sheet:      null, null, 1 = " + c(null,     null,        Array [Int] (1)))
    println ("select row sheet:     0, null, 1 = " + c(Array [Int] (0), null,        Array [Int] (1)))
    println ("select col sheet   null,  0,2, 1 = " + c(null,     Array [Int] (0, 2), Array [Int] (1)))

    banner ("Test not")
    println ("c = " + c)
    println ("not row 0 = " + c.not(Array [Int] (0)))
    println ("not row col: 0, 0,2 = " + c.not(Array [Int] (0), Array [Int] (0, 2)))
    println ("not row col sheet: 0, 0,2, 1 = " + c.not(Array [Int] (0), Array [Int] (0, 2), Array [Int] (1)))

end tensorDTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tensorDTest2` main function is used to test the `TensorD` class.
 *  It tests pulling matrices and vectors from the tensor.
 *  > runMain scalation.mathstat.tensorDTest2
 */
@main def tensorDTest2 (): Unit =

    // 4 rows, 3 columns, 2 sheets - x_ijk
    //                                              row columns sheet
    val x = TensorD ((4, 3, 2), 1,  2,  3,       //  0   0,1,2    0
                                4,  5,  6,       //  1   0,1,2    0
                                7,  8,  9,       //  2   0,1,2    0
                               10, 11, 12,       //  3   0,1,1    0

                               13, 14, 15,       //  0   0,1,2    1
                               16, 17, 18,       //  1   0,1,2    1
                               19, 20, 21,       //  2   0,1,2    1
                               22, 23, 24)       //  3   0,1,2    1

     banner ("Tensor with dimensions (rows, columns, sheets) = (4, 3, 2)")
     println ("x          = " + x)

     // SCALARS
     banner ("Scalar element at index position (i, j, k) = (0, 0, 0)")
     println ("x(0, 0, 0) = " + x(0, 0, 0))                      // x_000  - element i=0, j=0, k=0

     // VECTORS
     banner ("Vector at index position (i, j) = (0, 0)")
     println ("x(0, 0)    = " + x(0, 0))                         // x_00:  - vector  i=0, j=0, k=all
     banner ("Vector at index position (i, ?, k) = (0, all, 0)")
     println ("x(0, ?, 0) = " + x(0, ?, 0))                      // x_0:0  - vector  i=0, j=all, k=0
     banner ("Vector at index position (?, j, k) = (all, 0, 0)")
     println ("x(?, 0, 0) = " + x(?, 0, 0))                      // x_:00  - vector  i=all, j=0, k=0

     // MATRICES
     banner ("Matrix from tensor with row i fixed at 0")
     println ("x(0)       = " + x(0))                            // x_0::  - matrix with row i fixed
     banner ("Matrix from tensor with column j fixed at 0")
     println ("x(?, 0)    = " + x(?, 0))                         // x_:0:  - matrix with column j fixed
     banner ("Matrix from tensor with sheet k fixed at 0")
     println ("x(?, ?, 0) = " + x(?, ?, 0))                      // x_::0  - matrix with sheet k fixed
     banner ("Ranged matrix from tensor with sheet k fixed at 0")
     println ("x(1 until 3,?, 0) = " + x(1 until 3,?, 0))                      // x_::1  - matrix with sheet k fixed

end tensorDTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tensorDTest3` main function is used to test the `TensorD` class.
 *  It tests the use of tensors and matrices for convolutional operation needed in
 *  Convolutional Nets.
 *  > runMain scalation.mathstat.tensorDTest3
 */
@main def tensorDTest3 (): Unit =

    val a = new TensorD (2, 9, 9)
    for i <- a.indices; j <- a.indices2; k <- a.indices3 do a(i, j, k) = i + j + k
    println (s"a = $a")

    val image0 = a(0)
    val image1 = a(1)
    println (s"image0 = $image0")
    println (s"image1 = $image1")

    val kernel = MatrixD ((3, 3), 1, 2, 1,
                                  2, 3, 2,
                                  1, 2, 1)
    println (s"kernel = $kernel")

    val sp = new MatrixD (image0.dim - kernel.dim2 + 1, image0.dim2 - kernel.dim2 + 1)
//  for i <- sp.indices; j <- sp.indices2 do sp(i, j) = kernel **+ (image0, i, j)           // FIX **+ only in MatrixD.scala.sav
    println (s"sp = $sp")

end tensorDTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tensorDTest4` main function is used to test the `TensorD` class.
 *  It tests all element-wise operations with TensorD, MatrixD, VectorD, and scalars
 *  that can actually broadcast properly.
 *  > runMain scalation.mathstat.tensorDTest4
 */
@main def tensorDTest4 (): Unit =

    // Create a base 3D tensor of shape (2,3,4)
    val t1 = TensorD ( (2, 3, 4),
        // 24 elements for shape (2,3,4)
        1,  2,  3,  4,   5,  6,  7,  8,   9, 10, 11, 12,
        13,14, 15, 16,  17, 18,19, 20,  21,22, 23, 24
    )
    println (s"Tensor t1 => shape = (2,3,4):\n$t1")

    // Another same-shape tensor (2,3,4)
    val t2 = TensorD ( (2, 3, 4),
        // 24 more elements
        2,  3,  4,  5,  6,  7,  8,  9,  10,11, 12, 13,
        14,15, 16,17,  18,19, 20,21,  22,23, 24, 25
    )
    println (s"Tensor t2 => shape = (2,3,4):\n$t2")

    // Matrix of shape (2,3) => broadcasts to (2,3,1), then final (2,3,4)
    val m1 = MatrixD ((2, 3), 1, 2, 3,
                              4, 5, 6)
    println (s"Matrix m1 => shape = (2,3):\n$m1")

    // Vector of length 3 => broadcasts to (1,3,1), then final (2,3,4)
    val v1 = VectorD (1, 2, 3)
    println (s"Vector v1 => length = 3:\n$v1")

    // Scalar
    val s = 2.0
    println (s"Scalar s => $s")

    // ------------------- Broadcasting Tests -------------------
    banner ("Addition Tests")
    println (s"t1 + t2:\n${t1 + t2}")
    println (s"t1 + m1:\n${t1 + m1}")
    println (s"t1 + v1:\n${t1 + v1}")
    println (s"t1 + s :\n${t1 + s}")

    banner ("Subtraction Tests")
    println (s"t1 - t2:\n${t1 - t2}")
    println (s"t1 - m1:\n${t1 - m1}")
    println (s"t1 - v1:\n${t1 - v1}")
    println (s"t1 - s :\n${t1 - s}")

    banner ("Multiplication Tests")
    println (s"t1 * t2:\n${t1 * t2}")
    println (s"t1 * m1:\n${t1 * m1}")
    println (s"t1 * v1:\n${t1 * v1}")
    println (s"t1 * s :\n${t1 * s}")

    banner ("Division Tests")
    println (s"t1 / t2:\n${t1 / t2}")
    println (s"t1 / m1:\n${t1 / m1}")
    println (s"t1 / v1:\n${t1 / v1}")
    println (s"t1 / s :\n${t1 / s}")

    banner ("Element-wise Hadamard Product ( *~ )")
    println (s"t1 *~ t2:\n${t1 *~ t2}")
    println (s"t1 *~ m1:\n${t1 *~ m1}")
    println (s"t1 *~ v1:\n${t1 *~ v1}")

    banner ("Negation Tests")
    println (s"-t1:\n${-t1}")

end tensorDTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tensorDTest5` main function is used to test the `TensorD` class.
 *  It tests other operations with TensorD, MatrixD, and VectorD.
 *  > runMain scalation.mathstat.tensorDTest5
 */
@main def tensorDTest5 (): Unit =

    // Create two TensorD objects that behave as row vectors (n, 1, 1)
    val A = VectorD (1, 2, 3)
    val B = VectorD (4, 5, 6)
    val tensorA = TensorD.fromVector (A)
    val tensorB = TensorD.fromVector (B)

    // Compute dot product
    val result = tensorA.dot (tensorB)

    // Expected result: (1*4 + 2*5 + 3*6) = 32.0
    println (s"Dot product result: $result")
    assert (result(0)(0)(0) == 32.0, s"Test failed! Expected 32.0 but got $result")

    println ("✅ dot product test passed!")

   // ---------------- MatrixD Inputs ----------------
    val C = MatrixD((2, 3), 1, 2, 3,
                            4, 5, 6)
    val D = MatrixD ((3, 2), 7, 8,
                             9, 10,
                            11, 12)

    println (s"C :\n$C")
    println (s"D :\n$D")

    // ---------------- Convert to TensorD (batch-first) ----------------
    val tensorC = TensorD.fromMatrix (C, Some((1, 2, 3))) // (1, 2, 3)
    val tensorD = TensorD.fromMatrix (D, Some((1, 3, 2))) // (1, 3, 2)

    println (s"tensorC : \n${tensorC(0)}")
    println (s"tensorD : \n${tensorD(0)}")
    println (s"C shape: ${C.dims}")
    println (s"D shape: ${D.dims}")
    println (s"tensorC shape: ${tensorC.shape}")
    println (s"tensorD shape: ${tensorD.shape}")

    // ---------------- Perform Matrix Multiplication ----------------
    val resultMat = tensorC.matmul (tensorD) // (1, 2, 2)
    println (s"Matmul result:\n${resultMat(0)}")

    // ---------------- Expected Result ----------------
    // Computed as: [1 2 3] * D = [58 64], [4 5 6] * D = [139 154]
    val expected = fromMatrix (MatrixD ((2, 2), 58.0000,        64.0000,
                                               139.000,        154.000))

    // ---------------- Assert and Pass ----------------
    assert (resultMat == expected, s"Test failed! Expected ${expected} but got ${resultMat}")
    println ("✅ matmul test passed!")

end tensorDTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tensorDTest6` main function is used to test the `TensorD` class.
 *  It tests operations with with and without broadcasting.
 *  > runMain scalation.mathstat.tensorDTest6
 */
@main def tensorDTest6 (): Unit =

    println ("==> Test 1: No Broadcasting")

    val A0 = MatrixD ((2, 3), 1, 2, 3,
                              4, 5, 6)
    val A1 = MatrixD ((2, 3), 7, 8, 9,
                             10, 11, 12)
    val B0 = MatrixD ((3, 2), 1, 2,
                              3, 4,
                              5, 6)
    val B1 = MatrixD ((3, 2), 7, 8,
                              9, 10,
                             11, 12)

    val tensorA1 = TensorD (A0, A1)    // Shape: (2, 2, 3)
    val tensorB1 = TensorD (B0, B1)    // Shape: (2, 3, 2)

    val result1   = tensorA1.bmm (tensorB1)
    val expected1 = TensorD (A0 * B0, A1 * B1)

    println (s"Result 1:\n$result1")
    assert (result1 == expected1, "❌ Test 1 failed!")
    println ("✅ Test 1 passed (No broadcasting)")

    // ---------------------------------------------------------

    println ("==> Test 2: Broadcast A")

    val tensorA2  = TensorD (A0) // Shape: (1, 2, 3)
    val result2   = tensorA2.bmm (tensorB1)
    val expected2 = TensorD (A0 * B0, A0 * B1)

    println (s"Result 2:\n$result2")
    assert (result2 == expected2, "❌ Test 2 failed!")
    println ("✅ Test 2 passed (Broadcast A)")

    // ---------------------------------------------------------

    println ("==> Test 3: Broadcast B")

    val tensorB3  = TensorD (B0) // Shape: (1, 3, 2)
    val result3   = tensorA1.bmm (tensorB3)
    val expected3 = TensorD (A0 * B0, A1 * B0)

    println (s"Result 3:\n$result3")
    assert (result3 == expected3, "❌ Test 3 failed!")
    println ("✅ Test 3 passed (Broadcast B)")

    println ("🎉 All bmm tests passed successfully!")

end tensorDTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tensorDTest7` main function is used to test the `TensorD` class.
 *  It tests permuting tensors.
 *  > runMain scalation.mathstat.tensorDTest7
 */
@main def tensorDTest7 (): Unit =

    banner ("TensorD Permute Function - Axis Permutation Tests")

    // Create a small reference tensor with unique values
    // Shape: (2, 3, 4)
    val t = TensorD ((2, 3, 4),
        1, 2, 3, 4,     5, 6, 7, 8,     9, 10, 11, 12,     // Slice 0 (i = 0)
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24     // Slice 1 (i = 1)
    )

    println (s"Original shape: ${t.shape}")
    println (s"Original tensor (flattened): ${t.flattenToVector.mkString(", ")}")

    // Define all valid permutations of (0, 1, 2)
    val permutations = Seq (
        Seq (0, 1, 2),   // identity
        Seq (0, 2, 1),
        Seq (1, 0, 2),
        Seq (1, 2, 0),
        Seq (2, 0, 1),
        Seq (2, 1, 0))

    // Check that double-permute restores the original shape and values
    for perm <- permutations do
        val permuted    = t.permute(perm)
        val reversePerm = perm.zipWithIndex.sortBy(_._1).map(_._2) // inverse permutation
        val unpermuted  = permuted.permute(reversePerm)

        val isSame  = t.flattenToVector == unpermuted.flattenToVector
        val shapeOk = t.shape == unpermuted.shape

        assert (isSame && shapeOk,
            s"❌ Failed on permutation $perm → reversed as $reversePerm")

        println (s"✅ Permute $perm → unpermute $reversePerm: Passed")
    end for

    println ("\n🎉 All permute tests passed successfully!")

end tensorDTest7


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tensorDTest8` main function is used to test the `TensorD` class.
 *  It tests concatenating tensors.
 *  > runMain scalation.mathstat.tensorDTest8
 */
@main def tensorDTest8 (): Unit =

    banner ("TensorD Concat Tests")

    // -----------------------------------
    // axis = 0
    banner ("Testing concat along axis 0")

    val a0 = TensorD ((1 ,2, 2), 1, 2,
                                 3, 4)

    val b0 = TensorD ((1, 2, 2), 5, 6,
                                 7, 8)

    val c0 = TensorD.concat (Seq (a0, b0), 0)

    assert ((c0(0).flatten.toList ++ c0(1).flatten.toList) ==
            (a0.flattenToVector.toList ++ b0.flattenToVector.toList),
            s"❌ concat axis 0 failed: got\n$c0")

    banner ("✅ concat axis 0 passed")

    // -----------------------------------
    // axis = 1
    banner ("Testing concat along axis 1")

    val a1 = TensorD ((1, 1, 2), 1, 2)

    val b1 = TensorD ((1, 2, 2), 3, 4,
                                 5, 6)

    val c1 = TensorD.concat (Seq (a1, b1), 1)

    assert (c1(?, 0).flatten.toList == a1.flattenToVector.toList and
            c1(0 until c1.dim, 1 until c1.dim2, 0 until c1.dim3).flattenToVector.toList == b1.flattenToVector.toList,
            s"❌ concat axis 1 failed:\n$c1")

    println ("✅ concat axis 1 passed")

    // -----------------------------------
    // axis = 2
    banner ("Testing concat along axis 2")

    val a2 = TensorD ((1, 2, 1), 1,
                                 2)

    val b2 = TensorD ((1, 2, 2), 3, 4,
                                 5, 6)

    val c2 = TensorD.concat (Seq (a2, b2), 2)

    assert (c2(?, ?, 0).flatten.toList == a2.flattenToVector.toList and
            c2(0 until c2.dim, 0 until c2.dim2, 1 until c2.dim3).flattenToVector.toList == b2.flattenToVector.toList,
            s"❌ concat axis 2 failed:\n$c2")

    println ("✅ concat axis 2 passed")

    println ("\n🎉 All concat tests passed successfully!")

end tensorDTest8

