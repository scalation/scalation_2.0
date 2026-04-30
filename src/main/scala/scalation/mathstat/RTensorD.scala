
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Thu May 10 15:50:15 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Ragged (R) Tensor (3D) Algebra
 *
 *  @see www.stat.uchicago.edu/~lekheng/work/icm1.pdf
 *  @see www.math.ias.edu/csdm/files/13-14/Gnang_Pa_Fi_2014.pdf
 *  @see tspace.l
 */

package scalation
package mathstat

import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the sizes of all the second level arrays in u.
 *  @param u  the 3D array
 */
def sizes2 (u: Array [Array [Array [Double]]]): VectorI =
    VectorI (for i <- u.indices yield u(i).size)
end sizes2

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Duplicate a fixed second.third dimension d into a vector with d repeated.
 *  @param dim   size of the 1st level/dimension (row) of the tensor (height)
 *  @param d     fixed size of the 2nd/3rd level/dimension (column/sheet) of the tensor
 */
def dupDim (dim: Int, d: Int): VectorI =
    VectorI (for _ <- 0 until dim yield d)
end dupDim


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RTensorD` class is an implementation for 3-dimensional ragged tensors.
 *  The tensor may only be ragged in the middle dimension (dim2) and whose sizes
 *  may only be dependent of the first index, e.g., i in (i, j, k).
 *  The names of the dimensions corresponds to MATLAB (row, column, sheet).
 *  @param dim   size of the 1st level/dimension (row) of the tensor (height)
 *  @param dim2  variable size of the 2nd level/dimension (column) of the tensor (width)
 *  @param dim3  size of the 3rd level/dimension (sheet) of the tensor (depth)
 *  @param v     the 3D array for holding the tensor elements
 */
class RTensorD (val dim: Int, val dim2: VectorI, val dim3: Int,
              private [mathstat] var v: Array [Array [Array [Double]]] = null)
      extends Serializable:

    private val flaw =  flawf ("RTensorD")                    // flaw flag
    private val TAB  = "\t\t"                                 // use "\t" for scala and "\t\t" for sbt

    val indices  = 0 until dim                                // index range for the first level/dimension
//  val indices2 = 0 until dim2                               // index range for the second level/dimension
    val indices3 = 0 until dim3                               // index range for the third level/dimension

    inline def indices2(i: Int): Range = 0 until dim2(i)      // must be a method as dim2 varies

    /** Multi-dimensional array storage for ragged tensor
     */
    if v == null then
        v = Array.ofDim [Array [Array [Double]]] (dim)
        for i <- indices do v(i) = Array.ofDim (dim2(i), dim3)
    else if dim != v.length || dim3 != v(0)(0).length then
        flaw ("init", "dimensions are wrong")

    /** Format string used for printing vector values (change using setFormat)
     */
    protected var fString = "%g,\t"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a tensor from three dimensional array.
     *  @param u  the three dimensional array
     */
    def this (u: Array [Array [Array [Double]]]) =
        this (u.size, sizes2 (u), u(0)(0).size, u)
    end this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the format to the newFormat.
     *  @param newFormat  the new format string
     */
    def setFormat (newFormat: String): Unit = fString = newFormat

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the i, j, k element from the tensor.
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     *  @param k  3rd dimension (sheet) index of the tensor
     */
    def apply (i: Int, j: Int, k: Int): Double = v(i)(j)(k)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the i, j vector from the tensor.
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     */
    def apply (i: Int, j: Int): VectorD = VectorD (v(i)(j).toIndexedSeq)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the i matrix from the tensor.
     *  @param i  1st dimension (row) index of the tensor
     */
    def apply (i: Int): MatrixD = new MatrixD (dim2(i), dim3, v(i))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the ii._1 to ii._2 row slice of the tensor.
     *  @param ii  1st dimension (row) indices of the tensor
     */
    def apply (ii: (Int, Int)): RTensorD = new RTensorD (v.slice (ii._1, ii._2))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the ii._1 to ii._2, jj._1 to jj._2 row-column slice of the tensor.
     *  @param ii  1st dimension (row) indices of the tensor (null => all)
     *  @param jj  2nd dimension (column) indices of the tensor
     */
    def apply (ii: (Int, Int), jj: (Int, Int)): RTensorD =
        val (i1, i2) = if ii == null then (0, dim) else ii
        val u = v.slice (i1, i2)
        for i <- u.indices do u(i) = u(i).slice (jj._1, jj._2)
        new RTensorD (u)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the ii._1 to ii._2, jj._1 to jj._2, kk._1 to kk._2
     *  row-column-sheet slice of the tensor.
     *  @param ii  1st dimension (row) indices of the tensor (null => all)
     *  @param jj  2nd dimension (column) indices of the tensor (null => all)
     *  @param kk  3rd dimension (sheet) indices of the tensor
    def apply (ii: (Int, Int), jj: (Int, Int), kk: (Int, Int)): RTensorD =
        val (i1, i2) = if ii == null then (0, dim) else ii
        val (j1, j2) = if jj == null then (0, dim2) else jj
        val u = v.slice (i1, i2)
        for i <- u.indices do u(i) = u(i).slice (j1, j2)
        for i <- u.indices; j <- u(i).indices do u(i)(j) = u(i)(j).slice (kk._1, kk._2)
        new RTensorD (u)
    end apply
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     */
    def apply (is: Array [Int]): RTensorD = 
        new RTensorD (for i <- is yield v(i))
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is, js row-column selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor (null => all)
     *  @param js  2nd dimension (column) indices of the tensor
     */
    def apply (is: Array [Int], js: Array [Int]): RTensorD = 
        if is == null then
            val u = Array.ofDim [Double] (dim, js.size, dim3)
            for i <- indices; j <- js.indices; k <- indices3 do u(i)(j)(k) = v(i)(js(j))(k)
            new RTensorD (u)
        else
            val u = Array.ofDim [Double] (is.size, js.size, dim3)
            for i <- is.indices; j <- js.indices; k <- indices3 do u(i)(j)(k) = v(is(i))(js(j))(k)
            new RTensorD (u)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is, js, ks row-column-sheet selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor (null => all)
     *  @param js  2nd dimension (column) indices of the tensor
     *  @param ks  3rd dimension (sheet) indices of the tensor
     */
    def apply (is: Array [Int], js: Array [Int], ks: Array [Int]): RTensorD = 
        if js == null then
            flaw ("apply", "second dimension (js) being variable may not be null")
            null
        else if is == null then
            val u = Array.ofDim [Double] (dim, js.size, ks.size)
            for i <- indices; j <- js.indices; k <- ks.indices do u(i)(j)(k) = v(i)(js(j))(ks(k))
            new RTensorD (u)
        else
            val u = Array.ofDim [Double] (is.size, js.size, ks.size)
            for i <- is.indices; j <- js.indices; k <- ks.indices do u(i)(j)(k) = v(is(i))(js(j))(ks(k))
            new RTensorD (u)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the complement of the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     */
    def not (is: Array [Int]): RTensorD = apply (Array.range (0, dim) diff is)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the complement of the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     *  @param js  2nd dimension (column) indices of the tensor
     *
    def not (is: Array [Int], js: Array [Int]): RTensorD = apply (comple (is, dim), comple (js, dim2))
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the complement of the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     *  @param js  2nd dimension (column) indices of the tensor
     *  @param ks  3rd dimension (sheet) indices of the tensor
     *
    def not (is: Array [Int], js: Array [Int], ks: Array [Int]): RTensorD =
        apply (comple (is, dim), comple (js, dim2), comple (ks, dim3))
    end not
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single element of the tensor to the given value.
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     *  @param k  3rd dimension (sheet) index of the tensor
     *  @param x  the value to be updated at the above position in the tensor
     */
    def update (i: Int, j: Int, k: Int, x: Double): Unit = v(i)(j)(k) = x

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single vector of the tensor to the given vector.
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     *  @param x  the vector to be updated at the above position in the tensor
     */
    def update (i: Int, j: Int, x: VectorD): Unit = v(i)(j) = x.toArray

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single matrix of the tensor to the given matrix.
     *  @param i  1st dimension (row) index of the tensor
     *  @param x  the matrix to be updated at the above position in the tensor
     */
    def update (i: Int, x: MatrixD): Unit = v(i) = x.toArray

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all the tensor element values to x.
     *  @param x  the value to set all elements to
     */
    def set (x: Double): Unit = 
        for i <-indices; j <- indices2(i); k <- indices3 do v(i)(j)(k) = x
    end set

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this tensor and tensor b.
     *  @param b  the tensor to add (requires leDimensions)
     */
    def + (b: RTensorD): RTensorD =
        val c = new RTensorD (dim, dim2, dim3)
        for i <- indices; j <- indices2(i); k <- indices3 do
            c.v(i)(j)(k) = v(i)(j)(k) + b.v(i)(j)(k)
        end for
        c
    end +

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this tensor and scalar s.
     *  @param s  the scalar to add
     */
    def + (s: Double): RTensorD =
        val c = new RTensorD (dim, dim2, dim3)
        for i <- indices; j <- indices2(i); k <- indices3 do
            c.v(i)(j)(k) = v(i)(j)(k) + s
        end for
        c
    end +

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** From this tensor subtract tensor b.
     *  @param b  the tensor to add (requires leDimensions)
     */
    def - (b: RTensorD): RTensorD =
        val c = new RTensorD (dim, dim2, dim3)
        for i <- indices; j <- indices2(i); k <- indices3 do
            c.v(i)(j)(k) = v(i)(j)(k) - b.v(i)(j)(k)
        end for
        c
    end -

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** From this tensor subtract scalar s.
     *  @param s  the scalar to add
     */
    def - (s: Double): RTensorD =
        val c = new RTensorD (dim, dim2, dim3)
        for i <- indices; j <- indices2(i); k <- indices3 do
            c.v(i)(j)(k) = v(i)(j)(k) - s
        end for
        c
    end -

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this tensor by scalar s.
     *  @param s  the scalar to multiply by
     */
    def * (s: Double): RTensorD =
        val c = new RTensorD (dim, dim2, dim3)
        for i <- indices; j <- indices2(i); k <- indices3 do
            c.v(i)(j)(k) = v(i)(j)(k) * s
        end for
        c
    end *

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply (multi-linear product) this tensor by three matrices b, c and d.
     *      this * (a, b, c)
     *  @see www.stat.uchicago.edu/~lekheng/work/icm1.pdf - equation 15.1
     *  @param b  the first matrix to multiply by (requires leDimensions)
     *  @param c  the second matrix to multiply by (requires leDimensions)
     *  @param d  the third matrix to multiply by (requires leDimensions)
     */
    def * (b: MatrixD, c: MatrixD, d: MatrixD): RTensorD =
        val (m1, n1) = (b.dim, b.dim2)
        val (m2, n2) = (c.dim, c.dim2)
        val (m3, n3) = (d.dim, d.dim2)
        if n1 > dim || dupDim (n1, n2) > dim2 || n3 > dim3 then flaw ("*", "dimensions don't match")

        val e = new RTensorD (m1, dupDim (m1, m2), m3)
        for i <- b.indices; j <- c.indices; k <- d.indices do
            var sum = 0.0
            for l1 <- b.indices2; l2 <- c.indices2; l3 <- d.indices2 do
                sum += b(i, l1) * c(j, l2) * d(k, l3) * v(l1)(l2)(l3)
            end for
            e.v(i)(j)(k) = sum
        end for
        e
    end *

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply element-wise (Hadamard product) this tensor by tensor b.
     *  @param b  the tensor to add (requires leDimensions)
     */
    def *~ (b: RTensorD): RTensorD =
        val c = new RTensorD (dim, dim2, dim3)
        for i <- indices; j <- indices2(i); k <- indices3 do
            c.v(i)(j)(k) = v(i)(j)(k) * b.v(i)(j)(k)
        end for
        c
    end *~ 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divide this tensor by scalar s.
     *  @param s  the scalar to divide by
     */
    def / (s: Double): RTensorD =
        val c = new RTensorD (dim, dim2, dim3)
        for i <- indices; j <- indices2(i); k <- indices3 do
            c.v(i)(j)(k) = v(i)(j)(k) / s
        end for
        c
    end /

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether the dimensions of this tensor are less than or equal to
     *  le those of the other tensor b.
     *  @param b  the other matrix
     */
    def leDimensions (b: RTensorD): Boolean =
        dim <= b.dim && dim2 <= b.dim2 && dim3 <= b.dim3
    end leDimensions

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this tensor to a string with a double line break after each sheet
     *  and a single line break after each row.
     */
    override def toString: String = 
        val sb = new StringBuilder ("\nRTensorD(")
        if dim == 0 then return sb.append (")").mkString
        for k <- indices3 do
            for i <- indices do
                for j <- indices2(i) do sb.append (s"${v(i)(j)(k)}, ")
                sb.append ("\n" + TAB)
            end for
            sb.append ("\n" + TAB)
        end for
        sb.replace (sb.length-5, sb.length, ")").mkString
    end toString

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this tensor to a string with a line break after each sheet.
     */
    def toString2: String = 
        val sb = new StringBuilder ("\nRTensorD( ")
        if dim == 0 then return sb.append (")").mkString
        for i <- indices; j <- indices2(i) do
            sb.append (stringOf (v(i)(j)) + ", ")
            if j == dim2(i)-1 then sb.replace (sb.length-1, sb.length, "\n\t")
        end for
        sb.replace (sb.length-3, sb.length, ")").mkString
    end toString2

end RTensorD


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RTensorD` companion object provides factory methods for the `RTensorD` class.
 */
object RTensorD:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build an empty rectangular tensor.
     *  @param n1  the first dimension
     *  @param n2  the second dimension
     *  @param n3  the third dimension
     */
    def apply (n1: Int, n2: Int, n3: Int): RTensorD =
        new RTensorD (n1, dupDim (n1, n2), n3)
    end apply 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a rectangular tensor from the var-argument list x.
     *  @param n  the three dimensions (n1, n2, n3)
     *  @param x  the values for the tensor
     */
    def apply (n: (Int, Int, Int), x: Double*): RTensorD =
        val t = new RTensorD (n._1, dupDim (n._1, n._2), n._3)
        var l = 0
        for k <- 0 until n._3; i <- 0 until n._1; j <- 0 until n._2 do
            t(i, j, k) = x(l)
            l += 1
        end for
        t
    end apply 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the joint frequency of occurrence of value pairs (x_ij, y_i) for
     *  each column j in the data matrix.  Value counts for each column determine
     *  dim2, while the value count for y gives dim3.
     *  @param x     the input/data integer-valued matrix
     *  @param dim2  the varying second dimension of the resulting tensor (# values for each xj)
     *  @param y     the output/response integer-valued vector
     *  @param dim3  the third dimension of the resulting tensor (# values for y)
     */
    def freq (x: MatrixD, dim2: VectorI, y: VectorI, dim3: Int): RTensorD =
        if ! MatrixI.isIntegerValued (x) then return null
        val t = new RTensorD (x.dim2, dim2, dim3)
        for i <- x.indices do
            val y_i = y(i)
            for j <- x.indices2 do t(j, x(i, j).toInt, y_i) += 1
        end for
        t
    end freq

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the joint frequency of occurrence of value triples (x_i, z_i, y_i).
     *  The value counts for x and z give dim2, while the value count for y gives dim3.
     *  @param x     the first input/data integer-valued vector
     *  @param z     the second input/data integer-valued vector
     *  @param dim2  the varying second dimension of the resulting tensor (# values for x and z)
     *  @param y     the output/response integer-valued vector
     *  @param dim3  the third dimension of the resulting tensor (# values for y)
     */
    def freq (x: VectorI, z: VectorI, dim2: VectorI, y: VectorI, dim3: Int): RTensorD =
        if z.dim < x.dim then
            flaw ("freq", "z is required to have at the number elements in x")
            return null
        if dim2.dim != 2 then
            flaw ("freq", "dim2 must have dimension 2: one for x and one for z")
            return null
        val t = RTensorD (dim2(0), dim2(1), dim3)
        for i <- x.indices do t(x(i), z(i), y(i)) += 1
        t
    end freq

end RTensorD


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rTensorDTest` main function is used to test the `RTensorD` class.
 *  > runMain scalation.mathstat.rTensorDTest
 */
@main def rTensorDTest (): Unit =

    val s = 2.0
    val a = new RTensorD (2, dupDim (2, 3), 2)
    val b = new RTensorD (2, VectorI (3, 3), 2)
    //                                              row column sheet
    val c = RTensorD ((2, 3, 2), 1,  2,  3,       // 0   0-2     0
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

/*  FIX
    println ("slice row col sheet: 0:1, 0:2,  0:1 = " + c((0, 1), (0, 2), (0, 1)))
    println ("slice sheet:        null, null, 0:1 = " + c(null,   null,   (0, 1)))
    println ("slice row sheet:     0:1, null, 0:1 = " + c((0, 1), null,   (0, 1)))
    println ("slice col sheet     null, 0:2,  0:1 = " + c(null,   (0, 2), (0, 1)))
*/

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
/*  FIX
    println ("not row col: 0, 0,2 = " + c.not(Array [Int] (0), Array [Int] (0, 2)))
    println ("not row col sheet: 0, 0,2, 1 = " + c.not(Array [Int] (0), Array [Int] (0, 2), Array [Int] (1)))
*/

end rTensorDTest

