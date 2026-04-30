
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Thu May 10 15:50:15 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Ragged (R) Tensor (4D) Algebra
 *
 *  @see www.stat.uchicago.edu/~lekheng/work/icm1.pdf
 *  @see www.math.ias.edu/csdm/files/13-14/Gnang_Pa_Fi_2014.pdf
 *  @see tspace.l
 */

package scalation
package mathstat

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the sizes of all the third level arrays in u.
 *  @param u  the 4D array
 */
def sizes2_ (u: Array [Array [Array [Array [Double]]]]): VectorI =
    VectorI (for i <- u.indices yield u(i).size)
end sizes2_

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the sizes of all the third level arrays in u.
 *  @param u  the 4D array
 */
def sizes3 (u: Array [Array [Array [Array [Double]]]]): VectorI =
    VectorI (for i <- u.indices yield u(i)(0).size)
end sizes3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RTensor4D` class is an implementation for 3-dimensional ragged tensors.
 *  The tensor may only be ragged in the middle dimension (dim2) and whose sizes
 *  may only be dependent of the first index, e.g., i in (i, j, k).
 *  The names of the dimensions corresponds to MATLAB (row, column, sheet).
 *  @param dim   size of the 1st level/dimension (row) of the tensor (height)
 *  @param dim2  variable size of the 2nd level/dimension (column) of the tensor (width)
 *  @param dim3  variable size of the 3rd level/dimension (sheet) of the tensor (depth)
 *  @param dim4  size of the 4th level/dimension (channel) of the tensor (spectra)
 *  @param v     the 4D array for holding the tensor elements
 */
class RTensor4D (val dim: Int, val dim2: VectorI, val dim3: VectorI, val dim4: Int,
                private [mathstat] var v: Array [Array [Array [Array [Double]]]] = null)
      extends Serializable:

    private val flaw =  flawf ("RTensor4D")                   // flaw flag
    private val TAB  = "\t\t"                                 // use "\t" for scala and "\t\t" for sbt

    val indices  = 0 until dim                                // index range for the first level/dimension
//  val indices2 = 0 until dim2                               // index range for the second level/dimension
//  val indices3 = 0 until dim3                               // index range for the third level/dimension
    val indices4 = 0 until dim4                               // index range for the fourth level/dimension

    inline def indices2(i: Int): Range = 0 until dim2(i)      // must be a method as dim2 varies
    inline def indices3(i: Int): Range = 0 until dim3(i)      // must be a method as dim3 varies

    /** Multi-dimensional array storage for ragged tensor
     */
    if v == null then
        v = Array.ofDim [Array [Array [Array [Double]]]] (dim)
        for i <- indices do v(i) = Array.ofDim (dim2(i), dim3(i), dim4)
    else if dim != v.length || dim4 != v(0)(0)(0).length then
        flaw ("init", "dimensions are wrong")
    end if

    /** Format string used for printing vector values (change using setFormat)
     */
    protected var fString = "%g,\t"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a tensor from four dimensional array.
     *  @param u  the three dimensional array
     */
    def this (u: Array [Array [Array [Array [Double]]]]) = 
        this (u.size, sizes2_ (u), sizes3 (u), u(0)(0)(0).size, u)
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
     *  @param l  4rd dimension (channel) index of the tensor
     */
    def apply (i: Int, j: Int, k: Int, l: Int): Double = v(i)(j)(k)(l)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the i, j, k vector from this tensor.
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     *  @param k  3rd dimension (sheet) index of the tensor
     */
    def apply (i: Int, j: Int, k: Int): VectorD = VectorD (v(i)(j)(k).toIndexedSeq)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the i, j matrix from this tensor.
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     */
    def apply (i: Int, j: Int): MatrixD = new MatrixD (dim3(i), dim4, v(i)(j))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the i 3-tensor from this tensor.
     *  @param i  1st dimension (row) index of the tensor
     */
    def apply (i: Int): TensorD = new TensorD (dim2(i), dim3(i), dim4, v(i))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the ii._1 to ii._2 row slice of the tensor.
     *  @param ii  1st dimension (row) indices of the tensor
     */
    def apply (ii: (Int, Int)): RTensor4D = new RTensor4D (v.slice (ii._1, ii._2))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the ii._1 to ii._2, jj._1 to jj._2 row-column slice of the tensor.
     *  @param ii  1st dimension (row) indices of the tensor (null => all)
     *  @param jj  2nd dimension (column) indices of the tensor
     */
    def apply (ii: (Int, Int), jj: (Int, Int)): RTensor4D =
        val (i1, i2) = if ii == null then (0, dim) else ii
        val u = v.slice (i1, i2)
        for i <- u.indices do u(i) = u(i).slice (jj._1, jj._2)
        new RTensor4D (u)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the ii._1 to ii._2, jj._1 to jj._2, kk._1 to kk._2
     *  row-column-sheet slice of the tensor.
     *  @param ii  1st dimension (row) indices of the tensor (null => all)
     *  @param jj  2nd dimension (column) indices of the tensor (null => all)
     *  @param kk  3rd dimension (sheet) indices of the tensor
    def apply (ii: (Int, Int), jj: (Int, Int), kk: (Int, Int)): RTensor4D =
        val (i1, i2) = if ii == null then (0, dim) else ii
        val (j1, j2) = if jj == null then (0, dim2) else jj
        val u = v.slice (i1, i2)
        for i <- u.indices do u(i) = u(i).slice (j1, j2)
        for i <- u.indices; j <- u(i).indices do u(i)(j) = u(i)(j).slice (kk._1, kk._2)
        new RTensor4D (u)
    end apply
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     */
    def apply (is: Array [Int]): RTensor4D = 
        new RTensor4D (for i <- is yield v(i))
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is, js row-column selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor (null => all)
     *  @param js  2nd dimension (column) indices of the tensor
     *
    def apply (is: Array [Int], js: Array [Int]): RTensor4D = 
        if is == null then
            val u = Array.ofDim [Double] (dim, js.size, dim3)
            for i <- indices; j <- js.indices; k <- indices3 do u(i)(j)(k) = v(i)(js(j))(k)
            new RTensor4D (u)
        else
            val u = Array.ofDim [Double] (is.size, js.size, dim3)
            for i <- is.indices; j <- js.indices; k <- indices3 do u(i)(j)(k) = v(is(i))(js(j))(k)
            new RTensor4D (u)
        end if
    end apply
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is, js, ks row-column-sheet selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor (null => all)
     *  @param js  2nd dimension (column) indices of the tensor
     *  @param ks  3rd dimension (sheet) indices of the tensor
     *
    def apply (is: Array [Int], js: Array [Int], ks: Array [Int]): RTensor4D = 
        if js == null then
            flaw ("apply", "second dimension (js) being variable may not be null")
            null
        else if is == null then
            val u = Array.ofDim [Double] (dim, js.size, ks.size)
            for i <- indices; j <- js.indices; k <- ks.indices do u(i)(j)(k) = v(i)(js(j))(ks(k))
            new RTensor4D (u)
        else
            val u = Array.ofDim [Double] (is.size, js.size, ks.size)
            for i <- is.indices; j <- js.indices; k <- ks.indices do u(i)(j)(k) = v(is(i))(js(j))(ks(k))
            new RTensor4D (u)
        end if
    end apply
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the is, js, ks row-column-sheet selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor (null => all)
     *  @param js  2nd dimension (column) indices of the tensor
     *  @param ks  3rd dimension (sheet) indices of the tensor
     *  @param ls  4th dimension (channel) indices of the tensor
     */
    def apply (is: Array [Int], js: Array [Int], ks: Array [Int], ls: Array [Int]): RTensor4D = ???

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the complement of the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     */
    def not (is: Array [Int]): RTensor4D = apply (Array.range (0, dim) diff is)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the complement of the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     *  @param js  2nd dimension (column) indices of the tensor
     *
    def not (is: Array [Int], js: Array [Int]): RTensor4D = apply (comple (is, dim), comple (js, dim2))
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the complement of the is row selections from the tensor.
     *  @param is  1st dimension (row) indices of the tensor
     *  @param js  2nd dimension (column) indices of the tensor
     *  @param ks  3rd dimension (sheet) indices of the tensor
     *
    def not (is: Array [Int], js: Array [Int], ks: Array [Int]): RTensor4D =
        apply (comple (is, dim), comple (js, dim2), comple (ks, dim3))
    end not
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single element of the tensor to the given value.
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     *  @param k  3rd dimension (sheet) index of the tensor
     *  @param l  4th dimension (channel) indices of the tensor
     *  @param x  the value to be updated at the above position in the tensor
     */
    def update (i: Int, j: Int, k: Int, l: Int, x: Double): Unit = v(i)(j)(k)(l) = x

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single vector of the tensor to the given vector.
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     *  @param k  3rd dimension (sheet) index of the tensor
     *  @param x  the vector to be updated at the above position in the tensor
     */
    def update (i: Int, j: Int, k: Int, x: VectorD): Unit = v(i)(j)(k) = x.toArray

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single matrix of the tensor to the given matrix.
     *  @param i  1st dimension (row) index of the tensor
     *  @param j  2nd dimension (column) index of the tensor
     *  @param x  the matrix to be updated at the above position in the tensor
     */
    def update (i: Int, j: Int, x: MatrixD): Unit = v(i)(j) = x.toArray

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update a single matrix of the tensor to the given matrix.
     *  @param i  1st dimension (row) index of the tensor
     *  @param x  the matrix to be updated at the above position in the tensor
     */
    def update (i: Int, x: RTensorD): Unit = ???        // FIX v(i) = x.toArray

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all the tensor element values to x.
     *  @param x  the value to set all elements to
     */
    def set (x: Double): Unit = 
        for i <-indices; j <- indices2(i); k <- indices3(i); l <- indices4 do v(i)(j)(k)(l) = x
    end set

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this tensor and tensor b.
     *  @param b  the tensor to add (requires leDimensions)
     */
    def + (b: RTensor4D): RTensor4D =
        val c = new RTensor4D (dim, dim2, dim3, dim4)
        for i <- indices; j <- indices2(i); k <- indices3(i); l <- indices4 do
            c.v(i)(j)(k)(l) = v(i)(j)(k)(l) + b.v(i)(j)(k)(l)
        end for
        c
    end +

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this tensor and scalar s.
     *  @param s  the scalar to add
     */
    def + (s: Double): RTensor4D =
        val c = new RTensor4D (dim, dim2, dim3, dim4)
        for i <- indices; j <- indices2(i); k <- indices3(i); l <- indices4 do
            c.v(i)(j)(k)(l) = v(i)(j)(k)(l) + s
        end for
        c
    end +

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** From this tensor subtract tensor b.
     *  @param b  the tensor to add (requires leDimensions)
     */
    def - (b: RTensor4D): RTensor4D =
        val c = new RTensor4D (dim, dim2, dim3, dim4)
        for i <- indices; j <- indices2(i); k <- indices3(i); l <- indices4 do
            c.v(i)(j)(k)(l) = v(i)(j)(k)(l) - b.v(i)(j)(k)(l)
        end for
        c
    end -

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** From this tensor subtract scalar s.
     *  @param s  the scalar to add
     */
    def - (s: Double): RTensor4D =
        val c = new RTensor4D (dim, dim2, dim3, dim4)
        for i <- indices; j <- indices2(i); k <- indices3(i); l <- indices4 do
            c.v(i)(j)(k)(l) = v(i)(j)(k)(l) - s
        end for
        c
    end -

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this tensor by scalar s.
     *  @param s  the scalar to multiply by
     */
    def * (s: Double): RTensor4D =
        val c = new RTensor4D (dim, dim2, dim3, dim4)
        for i <- indices; j <- indices2(i); k <- indices3(i); l <- indices4 do
            c.v(i)(j)(k)(l) = v(i)(j)(k)(l) * s
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
     *
    def * (b: MatrixD, c: MatrixD, d: MatrixD): RTensor4D =
        val (m1, n1) = (b.dim, b.dim2)
        val (m2, n2) = (c.dim, c.dim2)
        val (m3, n3) = (d.dim, d.dim2)
        if n1 > dim || dupDim2 (n1, n2) > dim2 || n3 > dim3 then flaw ("*", "dimensions don't match")

        val e = new RTensor4D (m1, dupDim2 (m1, m2), m3)
        for i <- b.indices; j <- c.indices; k <- d.indices do
            var sum = 0.0
            for l1 <- b.indices2; l2 <- c.indices2; l3 <- d.indices2 do
                sum += b(i, l1) * c(j, l2) * d(k, l3) * v(l1)(l2)(l3)
            end for
            e.v(i)(j)(k) = sum
        end for
        e
    end *
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply element-wise (Hadamard product) this tensor by tensor b.
     *  @param b  the tensor to add (requires leDimensions)
     */
    def *~ (b: RTensor4D): RTensor4D =
        val c = new RTensor4D (dim, dim2, dim3, dim4)
        for i <- indices; j <- indices2(i); k <- indices3(i); l <- indices4 do
            c.v(i)(j)(k)(l) = v(i)(j)(k)(l) * b.v(i)(j)(k)(l)
        end for
        c
    end *~ 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divide this tensor by scalar s.
     *  @param s  the scalar to divide by
     */
    def / (s: Double): RTensor4D =
        val c = new RTensor4D (dim, dim2, dim3, dim4)
        for i <- indices; j <- indices2(i); k <- indices3(i); l <- indices4 do
            c.v(i)(j)(k)(l) = v(i)(j)(k)(l) / s
        end for
        c
    end /

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether the dimensions of this tensor are less than or equal to
     *  le those of the other tensor b.
     *  @param b  the other matrix
     */
    def leDimensions (b: RTensor4D): Boolean =
        dim <= b.dim && dim2 <= b.dim2 && dim3 <= b.dim3
    end leDimensions

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this tensor to a string with a double line break after each sheet
     *  and a single line break after each row.
     */
    override def toString: String = 
        val sb = new StringBuilder ("\nRTensor4D(")
        if dim == 0 then return sb.append (")").mkString
        for l <- indices4 do
            for i <- indices do
                for k <- indices3(i) do
                    for j <- indices2(i) do sb.append (s"${v(i)(j)(k)(l)}, ")
                    sb.append ("\n" + TAB)
            end for
            sb.append ("\n" + TAB)
        end for
        sb.replace (sb.length-5, sb.length, ")").mkString
    end toString

end RTensor4D


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RTensor4D` companion object provides factory methods for the `RTensor4D` class.
 */
object RTensor4D:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build an empty rectangular tensor.
     *  @param n1  the first dimension
     *  @param n2  the second dimension
     *  @param n3  the third dimension
     *  @param n4  the fourth dimension
     */
    def apply (n1: Int, n2: Int, n3: Int, n4: Int): RTensor4D =
        new RTensor4D (n1, dupDim (n1, n2), dupDim (n1, n3), n4)
    end apply 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a rectangular tensor from the var-argument list x.
     *  @param n  the four dimensions (n1, n2, n3, n4)
     *  @param x  the values for the tensor
     */
    def apply (n: (Int, Int, Int, Int), x: Double*): RTensor4D =
        val t = new RTensor4D (n._1, dupDim (n._1, n._2), dupDim (n._1, n._3), n._4)
        var h = 0
        for l <- 0 until n._4; k <- 0 until n._3; i <- 0 until n._1; j <- 0 until n._2 do
            t(i, j, k, l) = x(h)
            h += 1
        end for
        t
    end apply 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the joint frequency of occurrence of value triple (x_ij, x_ip, y_i) for
     *  each column j in the data matrix.  Value counts for each column determine
     *  dim2, value count for parent p gives dim2, while the value count for y gives dim4.
     *  @param x     the input/data integer-valued matrix
     *  @param dim2  the varying second dimension of the resulting tensor (# values for each xj)
     *  @param pa    the parent vector giving xp for each xj
     *  @param dim3  the varying third  dimension of the resulting tensor (# values for each xp)
     *  @param y     the output/response integer-valued vector
     *  @param dim4  the third dimension of the resulting tensor (# values for y)
     */
    def freq (x: MatrixD, dim2: VectorI, pa: VectorI, dim3: VectorI, y: VectorI, dim4: Int): RTensor4D =
        if ! MatrixI.isIntegerValued (x) then return null
        val t = new RTensor4D (x.dim2, dim2, dim3, dim4)
        for i <- x.indices do
            val y_i = y(i)
            for j <- x.indices2 do
                val x_ip = if pa(j) < 0 then 0 else x(i, pa(j)).toInt
                t(j, x(i, j).toInt, x_ip, y_i) += 1
        end for
        t
    end freq

end RTensor4D


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rTensor4DTest` main function is used to test the `RTensor4D` class.
 *  > runMain scalation.mathstat.rTensor4DTest
 */
@main def rTensor4DTest (): Unit =

    val s = 2.0
    val a = new RTensor4D (2, dupDim (2, 3), dupDim (2, 2), 2)
    val b = new RTensor4D (2, VectorI (3, 3), VectorI (2, 2), 2)
    //                                                 row column sheet  channel
    val c = RTensor4D ((2, 3, 2, 2), 1,  2,  3,       // 0   0-2     0       0
                                     4,  5,  6,       // 1   0-2     0

                                     7,  8,  9,       // 0   0-2     1
                                    10, 11, 12,       // 1   0-2     1
 
    //--------------------------------------------------------------------------

                                    13, 14, 15,       // 1   0-2     0       1
                                    16, 17, 18,       // 1   0-2     0

                                    10, 20, 21,       // 0   0-2     1
                                    22, 23, 24)       // 1   0-2     1

    for i <- 0 until 2; j <- 0 until 3; k <- 0 until 2; l <- 0 until 2 do
        val sum = i + j + k + l
        a(i, j, k, l) = sum
        b(i, j, k, l) = sum
    end for

    println ("s             = " + s)
    println ("a             = " + a)
    println ("b             = " + b)
    println ("c             = " + c)
    println ("c(0)          = " + c(0))
    println ("c(0, 0)       = " + c(0, 0))
    println ("c(0, 0, 0)    = " + c(0, 0, 0))
    println ("c(0, 0, 0, 0) = " + c(0, 0, 0, 0))

    banner ("Test operators")
    println ("a + b   = " + (a + b))
    println ("a + s   = " + (a + b))
    println ("a - b   = " + (a - b))
    println ("a - s   = " + (a - s))
    println ("c * s   = " + c * s)
    println ("a *~ c  = " + a *~ c)

/*
    val x = MatrixD ((2, 2), 1, 2,
                             3, 4)
    val y = MatrixD ((2, 3), 1, 2, 3,
                             4, 5, 6)
    val z = MatrixD ((2, 2), 5, 6,
                             7, 8)

    println ("c * (x, y, z) = " + c * (x, y, z))
*/

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

/*  FIX
    println ("select row col: 0, 0,2 = " + c(Array [Int] (0), Array [Int] (0, 2)))
    println ("select col:  null, 0,2 = " + c(null,     Array [Int] (0, 2)))

    println ("select row col sheet: 0,  0,2, 1 = " + c(Array [Int] (0), Array [Int] (0, 2), Array [Int] (1)))
    println ("select sheet:      null, null, 1 = " + c(null,     null,        Array [Int] (1)))
    println ("select row sheet:     0, null, 1 = " + c(Array [Int] (0), null,        Array [Int] (1)))
    println ("select col sheet   null,  0,2, 1 = " + c(null,     Array [Int] (0, 2), Array [Int] (1)))
*/

    banner ("Test not")
    println ("c = " + c)
    println ("not row 0 = " + c.not(Array [Int] (0)))
/*  FIX
    println ("not row col: 0, 0,2 = " + c.not(Array [Int] (0), Array [Int] (0, 2)))
    println ("not row col sheet: 0, 0,2, 1 = " + c.not(Array [Int] (0), Array [Int] (0, 2), Array [Int] (1)))
*/

end rTensor4DTest

