
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Jun 17 19:29:23 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Vector Data Structure of Doubles
 *
 *  https://www.infoq.com/news/2023/10/foreign-function-and-memory-api/
 */

package scalation
package mathstat

//import java.lang.foreign.MemorySegment                          // @see foreign-function-and-memory-api
//import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.util.Arrays.copyOf

import scala.collection.generic.DefaultSerializable
import scala.collection.immutable.{IndexedSeq => IIndexedSeq, Set}
import scala.collection.mutable.{ArrayBuffer, IndexedSeq}
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

/** Top-level type definition for functions mapping:
 */
type FunctionS2V = Double  => VectorD                             // scalar `Double`  to vector `VectorD`
type FunctionV2S = VectorD => Double                              // vector `VectorD` to scalar `Double`
type FunctionV2V = VectorD => VectorD                             // vector `VectorD` to vector `VectorD`

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Vectorize a scalar function (S2S) to create a vector function (V2V).
 *  @param f  the scalar function to vectorize
 */
def vectorize (f: FunctionS2S): FunctionV2V = (x: VectorD) => x.map (f(_))


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VectorD` class stores and operates on Numeric Vectors of base type `Double`.
 *  @param dim  the dimension/size of the vector
 *  @param v    the 1D array used to store vector elements
 */
class VectorD (val dim: Int,
               private [mathstat] var v: Array [Double] = null)
      extends IndexedSeq [Double]
         with PartiallyOrdered [VectorD]
         with DefaultSerializable:
    
    private val EPSILON = 1E-9                                    // number close to zero
    private val flaw    = flawf ("VectorD")                       // partial invocation of flaw function
    private val fString = "%g,\t"                                 // output format spec

    if v == null then
        v = Array.ofDim [Double] (dim)
    else if dim > v.length then
        flaw ("init", s"vector dimension is larger than space: dim = $dim > v.length = ${v.length}")
        assert (dim <= v.length)                                  // make this a fatal flaw

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the length of this vector.
     */
    inline def length: Int = dim
    inline def nd          = dim.toDouble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the size (dim) of this vector by more elements.
     *  @param more  the number of new elements to add
     */
    def expand (more: Int = dim): VectorD =
        if more < 1 then this                                     // no change
        else new VectorD (dim + more, Array.concat (v, new Array [Double] (more)))
    end expand

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a deep copy of this vector (note: clone may not be deep).
     *  Uses Java's native `Arrays.copyOf` for efficiency.
     */
    def copy: VectorD = new VectorD (dim, copyOf (v, dim))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the i-th element of this vector.
     *  @param i  the index of the element to return
     */
    def apply (i: Int): Double = v(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements in range r of this vector, making sure not to go beyond
     *  the end of vector.
     *  Usage: x(2 until 5)
     *  Caveat:  Only verified for "a until b" ranges, not "a to b" ranges
     *  @param r  the index range of elements to return
     */
    def apply (r: Range): VectorD = 
        if r.end > dim then
            flaw ("apply", s"Range beyond end: r.start = ${r.start}, r.end = ${r.end} > dim = $dim")
            new VectorD (dim - r.start, v.slice (r.start, dim))
        else if r.end - r.start <= 0 then
//          flaw ("apply", s"Range is empty: r.start = ${r.start}, r.end = ${r.end} > dim = $dim")
            new VectorD (0, Array.ofDim [Double] (0))
        else
            new VectorD (r.end - r.start, v.slice (r.start, r.end))
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements in index sequence idx of this vector.
     *  @param idx  the index sequence of elements to return
     */
    def apply (idx: IndexedSeq [Int]): VectorD =
        new VectorD (idx.size, cfor (idx.size) { i => v(idx(i)) })
    end apply

    def apply (idx: IIndexedSeq [Int]): VectorD =
        new VectorD (idx.size, cfor (idx.size) { i => v(idx(i)) })
    end apply

    def apply (idx: Array [Int]): VectorD =
        new VectorD (idx.size, cfor (idx.size) { i => v(idx(i)) })
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements not equal to index ix of this vector.
     *  @param ix  the index to skip/exclude
     */
    def not (ix: Int): VectorD = 
        val a = Array.ofDim [Double] (dim-1)
        cfor (0, ix) { i => a(i) = v(i) }
        cfor (ix+1, dim) { i => a(i) = v(i) }
        new VectorD (dim-1, a)
    end not

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements not in index sequence idx of this vector.
     *  VectorD (for i <- indices if ! (idx `contains` i) yield v(i))
     *  @param idx  the index sequence of elements to skip/exclude
     */
    def not (idx: IndexedSeq [Int]): VectorD =
        val a = ArrayBuffer [Double] ()
        cfor (0, dim) { i => if ! (idx `contains` i) then a += v(i) }
        new VectorD (a.size, a.toArray)
    end not

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a vector containing all but the first n elements of this vector.
     *  @param n  the number of elements to be dropped
     */
    override def drop (n: Int = 1): VectorD = new VectorD (dim - n, v.drop (n))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a vector containing all but the last n elements of this vector.
     *  @author Yousef Fekri Dabanloo
     *  @param n  the number of elements to be dropped
     */
    override def dropRight (n: Int = 1): VectorD = new VectorD (dim - n, v.dropRight (n))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the elements from this vector to form two vectors:  one from the elements in
     *  idx (e.g., testing set) and the other from elements not in idx (e.g., training set).
     *  Note split and split_ produce different element orders.
     *  @param idx  the set of element indices to include/exclude
     */
    def split (idx: Set [Int]): (VectorD, VectorD) =
        val len = idx.size
        val a   = new VectorD (len)
        val b   = new VectorD (dim - len)
        var j, k = 0
        cfor (indices) { i =>
            if idx `contains` i then
                a.v(j) = v(i)
                j += 1
            else
                b.v(k) = v(i)
                k += 1
        } // cfor
        (a, b)
    end split

    inline def split (idx: IndexedSeq [Int]): (VectorD, VectorD) = split (idx.toSet [Int])

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the elements from this vector to form two vectors:  one from the elements in
     *  idx (e.g., testing set) and the other from elements not in idx (e.g., training set).
     *  Concise, but less efficient than split
     *  @param idx  the element indices to include/exclude
     */
    def split_ (idx: IndexedSeq [Int]): (VectorD, VectorD) = (this(idx), not(idx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the elements from this vector to form two vectors:  one from the
     *  the first i elements and the other from the rest of the elements.
     *  @param i  the split index
     */
    def split (i: Int): (VectorD, VectorD) = (new VectorD (i, v.slice (0, i)),
                                              new VectorD (dim - i, v.slice (i, dim)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Chop this vector into k sub-vectors of equal sizes (perhaps except for the last one).
     *  @param k  the number of pieces to chop this vector into
     */
    def chop (k: Int): Array [VectorD] =
        if k <= 0 then flaw ("chop", s"k = $k must be at least one")
        val pieces = Array.ofDim [VectorD] (k)
        val size = dim / k
        cfor (0, k-1) { i => pieces(i) = this (i*size until (i+1)*size) }
        pieces(k-1) = this ((k-1)*size until dim)
        pieces
    end chop

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the p prior values that come before t [ v(t-p), ..., v(t-1) ].
     *  When the index is negative, just return v(0).
     *  @param p  the number of prior values to return
     *  @param t  the index from which prior values are sought (exclusive)
     */
    def prior (p: Int, t: Int): VectorD =
        val a = Array.ofDim [Double] (p)
        cfor (t-p, t) { j => a(j+p-t) = if j <= 0 then v(0) else v(j) }
        new VectorD (p, a)
    end prior

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the i-th element (or in range) of this vector.
     *  @param i  the index of the element to update
     *  @param a  the updated value to be assigned
     */
    def update (i: Int, a: Double): Unit = v(i) = a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the i-th element (or in range) of this vector.
     *  @param i  the index of the element to update
     *  @param a  the update value to be assigned
     */
    def update (r: Range, a: Double): Unit = cfor (r) { i => v(i) = a }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the i-th element (or in range) of this vector.
     *  @param i  the index of the element to update
     *  @param y  the update vector/indexed sequence to be assigned
     */
    def update (r: Range, y: VectorD): Unit = cfor (r) { i => v(i) = y.v(i) }
    def update (r: Range, y: IndexedSeq [Double]): Unit = cfor (r) { i => v(i) = y(i) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update all the element of this vector and return this updated vector.
     *  @param y  the other vector to be assigned
     *
    def := (y: VectorD): VectorD = 
       if y.dim != dim then flaw (":=", s"dimension dim = $dim != y.dim = ${y.dim}")
       v = y.v
       this
    end :=
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all elements in this vector to scalar zero.
     */
    def clear (): Unit = cfor (v.indices) { i => v(i) = 0.0 }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all elements in this vector to scalar a.
     *  @param a  the scalar value to be assigned
     */
    def set (a: Double): Unit = cfor (v.indices) { i => v(i) = a }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all elements in this vector to vector y.
     *  @param y  the vector/indexed sequence value to be assigned
     */
    def set (y: VectorD): Unit = cfor (v.indices) { i => v(i) = y.v(i) }
    def set (y: IndexedSeq [Double]): Unit = cfor (v.indices) { i => v(i) = y(i) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Iterate over this vector element by element applying the given function.
     *  @param f  the function to apply
     */
    override def foreach [U] (f: Double => U): Unit = { var i = 0; while i < dim do { f (v(i)); i += 1 } }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map the elements of this vector by applying the mapping function f.
     *  @param f  the function to apply
     */
    def map (f: FunctionS2S): VectorD = new VectorD (v.size, v.map (f))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Try to compare this vector to bb (return None if they are not comparable).
     *  As a partial order some vectors may not be comparable.
     *  @param bb  the other vector
     */
    infix def tryCompareTo [B >: VectorD: AsPartiallyOrdered] (bb: B): Option [Int] =
        if ! bb.isInstanceOf [VectorD] then return None
        val b  = bb.asInstanceOf [VectorD]
        var le = true
        var ge = true
        cfor (v.indices) { i =>
            if      ge && v(i) < b(i) then ge = false
            else if le && v(i) > b(i) then le = false
        } // cfor
        if ge && le then Some (0)
        else if le  then Some (-1)
        else if ge  then Some (1)
        else None
    end tryCompareTo

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find values in this vector of infinite magnitude, returning all such index positions.
     */
    def findInfinity: IIndexedSeq [Int] = for i <- indices if v(i).isInfinite yield i

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this vector is non-negative (contains no negative values).
     */
    def isNonnegative: Boolean = v.forall (_ >= 0.0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate this vector and vector (or indexed-seq) y.
     *  @param y  the other vector/indexed sequence
     */
    def ++ (y: VectorD): VectorD = new VectorD (dim + y.size, v ++ y.v)
    def ++ (y: IndexedSeq [Double]): VectorD = new VectorD (dim + y.size, v ++ y)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Prepend (or append) this vector with scalar a.
     *  @param a  the scalar second operand
     */
    def +: (a: Double): VectorD = new VectorD (dim + 1, a +: v)
    def :+ (a: Double): VectorD = new VectorD (dim + 1, v :+ a)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the negative of this vector (unary minus).
     */
//  def unary_- : VectorD = VectorD (for i <- v.indices yield -v(i))       // for ... yield is too slow
    def unary_- : VectorD = new VectorD (dim, cfor (dim) { i => -v(i) })

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of vectors this and y.
     *  @param y  the other vector/indexed sequence
     */
    def + (y: VectorD): VectorD = new VectorD (dim, cfor (dim) { i => v(i) + y.v(i) })
    def + (y: IndexedSeq [Double]): VectorD = new VectorD (dim, cfor (dim) { i => v(i) + y(i) })

    def - (y: VectorD): VectorD = new VectorD (dim, cfor (dim) { i => v(i) - y.v(i) })
    def - (y: IndexedSeq [Double]): VectorD = new VectorD (dim, cfor (dim) { i => v(i) - y(i) })

    def * (y: VectorD): VectorD = new VectorD (dim, cfor (dim) { i => v(i) * y.v(i) })
    def * (y: IndexedSeq [Double]): VectorD = new VectorD (dim, cfor (dim) { i => v(i) * y(i) })

    def / (y: VectorD): VectorD = new VectorD (dim, cfor (dim) { i => v(i) / y.v(i) })
    def / (y: IndexedSeq [Double]): VectorD = new VectorD (dim, cfor (dim) { i => v(i) / y(i) })

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of this and scalar a.
     *  @param a  the scalar second operand
     */
    inline def + (a: Double): VectorD = new VectorD (dim, cfor (dim) { i => v(i) + a })
    inline def - (a: Double): VectorD = new VectorD (dim, cfor (dim) { i => v(i) - a })
    inline def * (a: Double): VectorD = new VectorD (dim, cfor (dim) { i => v(i) * a })
    inline def / (a: Double): VectorD = new VectorD (dim, cfor (dim) { i => v(i) / a })

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of vectors this and y.
     *  Perform operations in-place (destructive) to reduce memory allocations.
     *  @param y  the other vector/indexed sequence
     */
    def += (y: VectorD): VectorD = { cfor (0, dim) { i => v(i) += y.v(i) }; this }
    def += (y: IndexedSeq [Double]): VectorD = { cfor (0, dim) { i => v(i) += y(i) }; this }

    def -= (y: VectorD): VectorD = { cfor (0, dim) { i => v(i) -= y.v(i) }; this }
    def -= (y: IndexedSeq [Double]): VectorD = { cfor (0, dim) { i => v(i) -= y(i) }; this }

    def *= (y: VectorD): VectorD = { cfor (0, dim) { i => v(i) *= y.v(i) }; this }
    def *= (y: IndexedSeq [Double]): VectorD = { cfor (0, dim) { i => v(i) *= y(i) }; this }

    def /= (y: VectorD): VectorD = { cfor (0, dim) { i => v(i) /= y.v(i) }; this }
    def /= (y: IndexedSeq [Double]): VectorD = { cfor (0, dim) { i => v(i) /= y(i) }; this }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of this and scalar a.
     *  Perform operations in-place (destructive) to reduce memory allocations.
     *  @param a  the scalar second operand
     */
    def += (a: Double): VectorD = { cfor (0, dim) { i => v(i) += a} ; this }
    def -= (a: Double): VectorD = { cfor (0, dim) { i => v(i) -= a} ; this }
    def *= (a: Double): VectorD = { cfor (0, dim) { i => v(i) *= a} ; this }
    def /= (a: Double): VectorD = { cfor (0, dim) { i => v(i) /= a} ; this }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this vector and scalar a only at position i, e.g., x + (3, 5.5).
     *  @param ia = (i, a)  the (index position, scalar) to add
     */
    def + (ia: (Int, Double)): VectorD = { val c = copy; c.v(ia._1) += ia._2; c }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract from this vector the scalar a only at position i, e.g., x - (3, 5.5).
     *  @param ia = (i, a)  the (index position, scalar) to subtract
     */
    def - (ia: (Int, Double)): VectorD = { val c = copy; c.v(ia._1) -= ia._2; c }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise power function of this vector raised to scalar a.
     *  @param a  the scalar second operand (double)
     */
    def ~^ (a: Double): VectorD = new VectorD (dim, cfor (dim) { i => v(i) ~^ a })

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise power function of this vector raised to scalar a.
     *  Extended to handle a negative base.
     *  @param a  the scalar second operand (rational number)
     */
    def ↑ (a: Rat): VectorD = new VectorD (dim, cfor (dim) { i => v(i) ↑ a })

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this vector and vector y are nearly equal.
     *  @param y  the other vector
     */
    def =~ (y: VectorD): Boolean =
        if dim != y.dim then return false
        var close = true
        breakable {
            cfor (0, dim) { i =>
                if ! (v(i) =~ y.v(i)) then { close = false; break () }
            } // cfor
        } // breakable
        close
    end =~

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the difference between this vector and vector y.
     *  @param y  the other vector/indexed sequence
     */
    infix def diff (y: VectorD): VectorD = 
        val a = v `diff` y.v
        new VectorD (a.size, a)
    end diff

    infix def diff (y: IndexedSeq [Double]): VectorD = 
        val a = v `diff` y
        new VectorD (a.size, a)
    end diff

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the intersection of this vector and vector y.
     *  @param y  the other vector/indexed sequence
     */
    infix def intersect (y: VectorD): VectorD = 
        val a = v `intersect` y.v
        new VectorD (a.size, a)
    end intersect

    infix def intersect (y: IndexedSeq [Double]): VectorD = 
        val a = v `intersect` y
        new VectorD (a.size, a)
    end intersect

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form a new vector consisting of the unique values in this vector.
     */
    override def distinct: VectorD = { val a = v.distinct; new VectorD (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the number of unique/distinct values in this vector.
     */
    def countDistinct: Int = v.distinct.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reverse the elements in this vector.
     */
    override def reverse: VectorD = new VectorD (dim, v.reverse)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this vector is sorted in ascending order.
     */
    def isSorted: Boolean = 
        var i = 0
        while i < dim-1 do if v(i) > v(i+1) then return false else i += 1
        true
    end isSorted

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sort the elements in this vector according to ord.lt (ascending order)
     *  returning a new sorted vector.
     */
    def sorted: VectorD = new VectorD (dim, v.sorted)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sort the elements in this vector according to cmp (use '_ > _' for descending order).
     *  @param cmp  the comparison operator.
     */
    override def sortWith (cmp: (Double, Double) => Boolean): VectorD =
        new VectorD (dim, v.sortWith (cmp))
    end sortWith

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the elements in this vector based on the predicate.
     *  @param p  the filter predicate
     */
    override def filter (p: Double => Boolean): VectorD = 
        val a = v.filter (p)
        new VectorD (a.size, a)
    end filter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the elements in this vector based on the predicate and map these
     *  elements using the formula.
     *  Faster replacement: for ... if ... yield
     *  @see `not` method
     *  @param p        the filter predicate
     *  @param formula  the formula to apply to the elements
     */
    def filterMap (p: Double => Boolean)(formula: Double => Double): VectorD =
        val a = v.filter (p).map (formula (_))
        new VectorD (a.size, a)
    end filterMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the elements in this vector based on the negation of the predicate.
     *  @param p  the filter predicate
     */
    override def filterNot (p: Double => Boolean): VectorD = 
        val a = v.filterNot (p)
        new VectorD (a.size, a)
    end filterNot

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the elements in this vector based on the predicate, returning index positions.
     *  @param p  the filter predicate based on element values
     */
    def filterPos (p: Double => Boolean): IIndexedSeq [Int] =
        for i <- indices if p(v(i)) yield i
    end filterPos

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the dot (inner) product of vectors this and y.
     *  Σ (indices) { i => v(i) * y(i) }
     *  @param y  the other vector/indexed sequence
     */
    infix def dot (y: VectorD): Double =
        var sum = 0.0
        cfor (0, dim) { i => sum += v(i) * y.v(i) }
        sum
    end dot

    infix def dot (y: IndexedSeq [Double]): Double =
        var sum = 0.0
        cfor (0, dim) { i => sum += v(i) * y(i) }
        sum
    end dot

    infix def dot (y: IIndexedSeq [Double]): Double =
        var sum = 0.0
        cfor (0, dim) { i => sum += v(i) * y(i) }
        sum
    end dot

    inline def ∙ (y: VectorD): Double = dot (y)                 // Unicode bullet point

    inline def ∙ (y: IndexedSeq [Double]): Double = dot (y)     // Unicode bullet point

    inline def ∙ (y: IIndexedSeq [Double]): Double = dot (y)    // Unicode bullet point

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'valid' (no padding) convolution of cofilter vector c and input vector x.
     *  Take the dot product of c (this) with a slice of x, shift by one and repeat.
     *  Usage:  c conv x
     *  Caveat:  does not include reversal.
     *  @see `scalation.modeling.neuralnet.CoFilter_1D`
     *  @param x  the input/data vector
     */
    infix def conv (x: VectorD): VectorD =
        val y = new VectorD (x.dim - dim + 1)
        cfor (y.indices) { k => y(k) = this ∙ x(k until k + dim) }
        y
    end conv

    inline def *+ (x: VectorD): VectorD = conv (x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'valid' (no padding) convolution of cofilter vector c and input vector x.
     *  Take the dot product of c (this) reversed with a slice of x, shift by one and repeat.
     *  Computes the discrete convolution of cofilter vector c and input vector x.
     *  Usage:  c conv_ x
     *  @param x  the input/data vector
     */
    inline infix def conv_ (x: VectorD): VectorD = reverse.conv (x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'same' (with zero padding) convolution of cofilter vector c and
     *  input vector x.
     *  Same means that the size of the result is the same as the input.
     *  Usage:  c convs x
     *  @param x  the input/data vector
     */  
    infix def convs (x: VectorD): VectorD =
        val y = new VectorD (x.dim)
        cfor (y.indices) { k =>
            cfor (indices) { j => if k+j > 0 && k+j <= x.dim then y(k) += v(j) * x(k+j-1) }
        } // cfor
        y
    end convs

    inline def *~+ (x: VectorD): VectorD = convs (x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'full' convolution of cofilter vector c and input vector x.
     *  @note:  `convf` is less common than `conv` or convs`
     *  @param x  the input/data vector
     */
    infix def convf (x: VectorD): VectorD =
        val y = new VectorD (dim + x.dim - 1)
        cfor (y.indices) { k =>
            y(k) = Σ (0, math.min (k+1, dim)) { j => if k-j < x.dim then v(j) * x(k-j) else 0.0 }
        } // cfor
        y
    end convf

    inline def *++ (x: VectorD): VectorD = convf (x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new vector consisting of the maximum of this vector elements and zero.
     */
    inline def max0: VectorD =
        new VectorD (dim, cfor (dim) { i => if v(i) < 0.0 then 0.0 else v(i) })
    end max0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new vector consisting of the maximum of this and y's corresponding elements.
     *  @param y  the other vector/indexed sequence
     */
    infix def maxv (y: VectorD): VectorD =
        new VectorD (dim, cfor (dim) { i => if v(i) >= y(i) then v(i) else y.v(i) })
    end maxv

    infix def maxv (y: IndexedSeq [Double]): VectorD =
        new VectorD (dim, cfor (dim) { i => if v(i) >= y(i) then v(i) else y(i) })
    end maxv

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new vector consisting of the minimum of this and y's corresponding elements.
     *  @param y  the other vector/indexed sequence
     */
    infix def minv (y: VectorD): VectorD =
        new VectorD (dim, cfor (dim) { i => if v(i) <= y(i) then v(i) else y.v(i) })
    end minv

    infix def minv (y: IndexedSeq [Double]): VectorD =
        new VectorD (dim, cfor (dim) { i => if v(i) <= y(i) then v(i) else y(i) })
    end minv

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the range (min to max) of values in this vector.
     */
    def min_max: VectorD = VectorD (min, max)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the argument maximum of this vector (index of maximum element).
     *  @param e  the ending index (exclusive) for the search
     */
    def argmax (e: Int = dim): Int =
        var j = 0
        cfor (1, e) { i => if v(i) > v(j) then j = i }
        j
    end argmax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the argument maximum of this vector (index of maximum element).
     *  @param s  the starting index (inclusive) for the search
     *  @param e  the ending index (exclusive) for the search
     */
    def argmax (s: Int, e: Int): Int =
        var j = s
        cfor (s+1, e) { i => if v(i) > v(j) then j = i }
        j
    end argmax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the argument minimum of this vector (index of minimum element).
     *  @param e  the ending index (exclusive) for the search
     */
    def argmin (e: Int = dim): Int =
        var j = 0
        cfor (1, e) { i => if v(i) < v(j) then j = i }
        j
    end argmin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the argument minimum of 'this' vector (index of minimum element).
     *  @param s  the starting index (inclusive) for the search
     *  @param e  the ending index (exclusive) for the search
     */
    def argmin (s: Int, e: Int): Int =
        var j = s
        cfor (s+1, e) { i => if v(i) < v(j) then j = i }
        j
    end argmin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the argument maximum magnitude of this vector (index of most extreme element).
     *  @param e  the ending index (exclusive) for the search
     */
    def argmag (e: Int = dim): Int =
        var j = 0
        cfor (1, e) { i => if math.abs (v(i)) > math.abs (v(j)) then j = i }
        j
    end argmag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the magnitude of this vector, i.e., the element value farthest from zero.
     */
    def mag: Double = math.max (math.abs (min), math.abs (max))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute three sums for the k-prefix, middle and k-suffix of this vector.
     *  @param k  the integer specifying the size of the prefix
     */
    def sums (k: Int): (Double, Double, Double) =
        var s0, s1, s2 = 0.0
        cfor (0, dim) { i =>
           if i < k then          s0 += v(i)
           else if i < dim-k then s1 += v(i)
           else s2 += v(i)
        } // cfor
        (s0, s1, s2)
    end sums

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute three squared norms for the k-prefix, middle and k-suffix of this vector.
     *  @param k  the integer specifying the size of the prefix
     */
    def normSqs (k: Int): (Double, Double, Double) =
        var s0, s1, s2 = 0.0
        cfor (0, dim) { i =>
           if i < k then          s0 += v(i) * v(i)
           else if i < dim-k then s1 += v(i) * v(i)
           else s2 += v(i) * v(i)
        } // cfor
        (s0, s1, s2)
    end normSqs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Euclidean norm (2-norm) (or its square) of this vector.
     */
    def normSq: Double = v.fold (0.0)((s, e) => s + e*e)
    def norm: Double   = math.sqrt (normSq)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Manhattan norm (1-norm) of this vector.
     */
    def norm1: Double = v.fold (0.0)((s, e) => s + math.abs (e))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the q-norm (with the root) of this vector, i.e., ||x||_q^q
     *  @param q  the power (^q) to apply to each element e
     */
    def norm_qq (q: Double): Double = v.fold (0.0)((s, e) => s + math.abs (e~^q))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the infinity norm (max absolute value) of this vector.
     */
    def normInf: Double = math.max (math.abs (min), math.abs (max))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector that is the element-wise absolute value of this vector.
     */
    def abs: VectorD = map (math.abs (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector consisting of the square root of each element of this vector.
     */
    def sqrt: VectorD = map (math.sqrt (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector consisting of the reciprocal of each element of this vector.
     */
    def recip: VectorD = map (1.0 / _)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Log transform this vector by using math.log (avoiding the log (0) problem).
     */
    def log: VectorD = map (math.log (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Log transform this vector by using math.log1p (avoiding the log (0) problem).
     */
    def log1p: VectorD = map (math.log1p (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Exp transform this vector by using math.exp (the inverse of log).
     */
    def exp: VectorD = map (math.exp (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Exp transform this vector by using math.expm1 (the inverse of log1p).
     */
    def expm1: VectorD = map (math.expm1 (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cos transform this vector by using math.cos (the inverse of acos).
     */
    def cos: VectorD = map (math.cos (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Inverse cos transform this vector by using math.acos (the inverse of cos).
     */
    def acos: VectorD = map (math.acos (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sin transform this vector by using math.sin (the inverse of asin).
     */
    def sin: VectorD = map (math.sin (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Inverse sin transform this vector by using math.asin (the inverse of sin).
     */
    def asin: VectorD = map (math.asin (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector containing the mid-points between adjacent elements.
     *  VectorD (for i <- 1 until dim yield 0.5 * (v(i) + v(i-1)))
     */
    def mids: VectorD = 
        val a = Array.ofDim [Double] (dim-1)
        cfor (1, dim) { i => a(i) = 0.5 * (v(i) + v(i-1)) }
        new VectorD (dim-1, a)
    end mids

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cumulate the values of 'this' vector from left to right (e.g., create a
     *  CDF from a pmf).  Example: (4, 2, 3, 1) --> (4, 6, 9, 10)
     *  var sum = 0.0; VectorD (for i <- v.indices yield { sum += v(i); sum })
     */
    def cumulate: VectorD =
        val a   = Array.ofDim [Double] (dim)
        var sum = 0.0
        cfor (0, dim) { i => sum += v(i); a(i) = sum }
        new VectorD (dim, a)
    end cumulate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `Double` vector to an `Int` vector.
     */
    def toInt: VectorI = new VectorI (dim, v.map (_.toInt))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert 'this' `VectorD` into a `VectorD`.
     */
    def toDouble: VectorD = this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert to a probability vector, by normalizing so that it sums to one.
     */
    def toProbability: VectorD = this * (1.0 / v.sum)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Normalize this vector so its length is one (unit vector).
     */
    def normalize: VectorD = this * (1.0 / norm)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Normalize this vector to have a maximum of one.
     */
    def normalize1: VectorD = this * (1.0 / max)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the number of zero elements in the this vector.
     */
    def countZero: Int =
        (Σ (indices) { i => if v(i) == 0.0 then 1 else 0 }).toInt
    end countZero

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert vector to a string.
     */
    override def toString: String =
        val sb = new StringBuilder ("VectorD(")
        if dim == 0 then return sb.append (")").mkString
        cfor (indices) { i => sb.append (fString.format (v(i))) }
        sb.replace (sb.length-2, sb.length, ")").mkString
    end toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert vector to vector of strings.
     */
    def toString2: VectorS = new VectorS (dim, v.map (_.toString))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap (in-place) elements i and k in this vector.
     *  @param i  the first element in the swap
     *  @param k  the second element in the swap
     */
    def swap (i: Int, k: Int): Unit =
        val tmp = v(i); v(i) = v(k); v(k) = tmp
    end swap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly find the k-median of the p to r partition of array v
     *  using the QuickSelect algorithm.
     *  @see http://en.wikipedia.org/wiki/Quickselect
     *  @param rk  the rank order
     *  @param p   the left cursor
     *  @param r   the right cursor
     *  @param k   the type of median (k-th smallest element)
     */
    private def median (rk: Array [Int], p: Int, r: Int, k: Int): Double =
        if p == r then return v(rk(p))
        iswap (rk, r, med3 (p, (p+r)/2, r))                // use median-of-3, comment out for simple pivot
        val q = ipartition (rk, p, r)                      // partition into left (<=) and right (>=)
        if q == k-1 then     return v(rk(q))               // found k-median
        else if q > k-1 then median (rk, p, q - 1, k)      // recursively find median in left partition
        else median (rk, q + 1, r, k)                      // recursively find median in right partition
    end median

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly find the k-median (k-th smallest element) of array v.
     *  @param k  the type of median (e.g., k = (dim+1)/2 is the median)
     */
    def median (k: Int = (dim+1)/2): Double =
        if dim <= 0 then flaw ("median", s"no vector to take the median of k = $k, dim = $dim")
        median (Array.range (0, dim), 0, dim-1, k)
    end median

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the fraction quantile.
     *  @param fraction  the fraction/percentile to take
     */
    def quantile (fraction: Double): Double =
        var k = (fraction * dim).toInt
        if k >= dim then k = dim - 1
        if k <= 0   then k = 1
        median (k)
    end quantile

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the first Q1 (1/4) and third Q3 (3/4) quartiles.  Note: IQR = Q3 - Q1.
     */
    def q1_q3: VectorD = VectorD (quantile (0.25), quantile (0.75)) 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the averaged median, which is the median when dim is odd and
     *  the average of the median and the next k-median when dim is even.
     */
    def median_ : Double = if dim % 2 == 0 then (median () + median ((dim+2)/2)) / 2.0
                           else median ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the rank order for this vector, return its elements in that order.
     *  The rank order may be established using indirect sorting (e.g., iqsort).
     *  @param rank  the rank order of elements in this vector
     */
    def reorder (rank: Array [Int]): VectorD =
        new VectorD (dim, cfor (dim) { i => v(rank(i)) })
    end reorder

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively and indirectly sort the p to r partition of array v 
     *  using QuickSort.  This version avoids stack overflow.
     *  @author Sulaiman Owodunni
     *  @param rk  the rank order
     *  @param p   the left cursor
     *  @param r   the right cursor
     */
    private def iqsort (rk: Array [Int], p: Int, r: Int): Array [Int] =
        if r - p > 5 then
            iswap (rk, r, med3 (p, (p+r)/2, r))            // use median-of-3, comment out for simple pivot
            var (p_, r_) = (p, r)                          // use local cursors
            while p_ < r_ do
                val pivot = ipartition (rk, p_, r_)        // partition into left (<=) and right (>=)
                if pivot - p_ < r_ - pivot then            // recurse on the smaller sub-array
                    iqsort (rk, p_, pivot - 1)             // recursively sort left partition
                    p_ = pivot + 1
                else
                    iqsort (rk, pivot + 1, r_)             // recursively sort right partition
                    r_ = pivot - 1
            end while
        else
            iselsort (rk, p, r)                            // use simple sort when small
        rk
    end iqsort

/*
    private def iqsort_ (rk: Array [Int], p: Int, r: Int): Array [Int] =
        if r - p > 5 then
            iswap (rk, r, med3 (p, (p+r)/2, r))            // use median-of-3, comment out for simple pivot
            val pivot = ipartition (rk, p, r)              // partition into left (<=) and right (>=)
            iqsort (rk, p, pivot - 1)                      // recursively sort left partition
            iqsort (rk, pivot + 1, r)                      // recursively sort right partition
        else
            iselsort (rk, p, r)                            // use simple sort when small
        rk
    end iqsort_
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort this vector using QuickSort, returning the rank order.
     */
    inline def iqsort: Array [Int] = iqsort (Array.range (0, dim), 0, dim-1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort the p to r partition of array v using Selection Sort.
     *  @param rk  the rank order (index order, index of smallest element, etc.)
     *  @param p   the left cursor (inclusive)
     *  @param r   the right cursor (inclusive)
     */
    private def iselsort (rk: Array [Int], p: Int, r: Int): Array [Int] =
        cfor (p, r+1) { i =>
            var k = i
            cfor (i+1, r+1) { j => if v(rk(j)) < v(rk(k)) then k = j }
            if i != k then iswap (rk, i, k)
        } // cfor
        rk
    end iselsort

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort this vector using SelectionSort, returning the rank order.
     */
    def iselsort: Array [Int] = iselsort (Array.range (0, dim), 0, dim-1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort this vector using Selection Sort, returning the rank order
     *  of the stop smallest elements.
     *  @param stop  only sort stop number of smallest elements
     */
    def iselsort (stop: Int = dim): Array [Int] = 
        val rk = Array.range (0, dim)
        cfor (0, stop) { i =>
            var k = i
            cfor (i+1, dim) { j => if v(rk(j)) < v(rk(k)) then k = j }
            if i != k then iswap (rk, i, k)
        } // cfor
        rk.slice (0, stop)
    end iselsort

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly partition the array from 'p' to 'r' into a left partition
     *  (<= 'x') and a right partition (>= 'x').
     *  @param rk  the rank order
     *  @param p   the left cursor
     *  @param r   the right cursor
     */
    private def ipartition (rk: Array [Int], p: Int, r: Int): Int =
        val x = v(rk(r))                                   // pivot
        var i = p - 1
        cfor (p, r) { j => if v(rk(j)) <= x then { i += 1; iswap (rk, i, j) }}
        iswap (rk, i + 1, r)
        i + 1
    end ipartition

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the index of the median of three elements.
     *  @param i  element index 1
     *  @param j  element index 2
     *  @param k  element index 3
     */
    inline private def med3 (i: Int, j: Int, k: Int): Int =
        if v(i) < v(j) then
            if v(j) < v(k) then j else if v(i) < v(k) then k else i
        else
            if v(j) > v(k) then j else if v(i) > v(k) then k else i
    end med3

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly swap the elements at i and j, i.e., rk(i) <-> rk(j).
     *  @param rk  the rank order
     *  @param i   the first index position
     *  @param j   the second index position
     */
    inline private def iswap (rk: Array [Int], i: Int, j: Int): Unit =
        val t = rk(i); rk(i) = rk(j); rk(j) = t
    end iswap

// Compute statistics/metrics

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample mean (also the population mean, they are the same).
     *  >> E(X)
     */
    inline def mean: Double = v.sum / nd

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean of a subvector.
     *  @param j  the starting index (inclusive)
     *  @param k  the ending index (exclusive)
     */
    def mean (j: Int, k: Int = dim): Double =
        if k <= j then v(j)
        else (Σ (j, k) { i => v(i) }) / (k - j)
    end mean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the centered norm-squared of this vector.
     */
    def cnormSq: Double =
        var e, s, ss = 0.0
        cfor (indices) { i => e = v(i); s += e; ss += e * e }
        ss - s * s / nd
    end cnormSq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample variance (and population population).
     *  >> E(X - μ)^2
     */
    def variance: Double   = cnormSq / (nd-1)
    def variance_ : Double = cnormSq / nd
    def variance (mu: Double): Double  = (normSq - mu * mu * nd) / (nd-1)
    def variance_ (mu: Double): Double = (normSq - mu * mu * nd) / nd

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample covariance (or population covariance) of this vector with vector y.
     *  @param y  the other vector
     */
    infix def cov (y: VectorD): Double  = ((this dot y) - v.sum * y.v.sum / nd) / (nd-1)
    infix def cov (y: IndexedSeq [Double]): Double  = ((this dot y) - v.sum * y.sum / nd) / (nd-1)
    infix def cov_ (y: VectorD): Double = ((this dot y) - v.sum * y.v.sum / nd) / nd
    infix def cov_ (y: IndexedSeq [Double]): Double = ((this dot y) - v.sum * y.sum / nd) / nd

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean square (ms) (or root mean square (rms)) of this vector.
     */
    def ms: Double  = normSq / nd
    def rms: Double = math.sqrt (ms)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample standard deviation (and population standard deviation).
     */
    def stdev: Double   = math.sqrt (variance)
    def stdev_ : Double = math.sqrt (variance_)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the coefficient of variations.
     */
    def cv: Double = stdev / mean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute Pearson's correlation of this vector with vector y.
     *  If either variance is zero, will result in Not-a-Number (NaN), return
     *  one if the vectors are the same, or -0 (indicating undefined).
     *  @param y  the other vector
     */
    infix def corr (y: VectorD): Double =
        val c = cov (y) / math.sqrt (variance * y.variance)
        if c.isNaN then if this == y then 1.0 else -0.0 else c
    end corr

// Compute Auto-Covariance

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for STATIONARY series:
     *  - adjusted = false follows the standard definition that divides by the number of
     *    elements less owe, 'dim-1', to avoid singularity issues.
     *  - adjusted = true follows an intuitive definition that divides by the number of
     *    elements summed, i.e., 'dim-k'.
     *  @see stats.stackexchange.com/questions/56238/question-about-sample-autocovariance-function
     *  @param k         the lag parameter (0 <= k < n) 
     *  @param adjusted  whether to adjust to account for the number of elements in the sum Σ (or use dim-1)
     */
    def acov (k: Int = 1, adjusted: Boolean = true): Double =
        if k >= dim then flaw ("acov", s"the vector is not long enough to compute acov for lag k = $k")
        val n  = if adjusted then dim - k else dim - 1
        val mu = mean
        (Σ (0, dim-k) { i => (v(i) - mu) * (v(i+k) - mu) }) / n
    end acov

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for STATIONARY series
     *  when the mean has been pre-computed (passed in).
     *  @param k         the lag parameter (0 <= k < n) 
     *  @param mu        the pre-computed mean
     *  @param adjusted  whether to adjust to account for the number of elements in the sum Σ (or use dim-1)
     */
    def acov (k: Int, mu: Double, adjusted: Boolean): Double =
        if k >= dim then flaw ("acov", s"the vector is not long enough to compute acov for lag k = $k")
        val n = if adjusted then dim - k else dim - 1
        (Σ (0, dim-k) { i => (v(i) - mu) * (v(i+k) - mu) }) / n
    end acov

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag cross-covariance of this vector and vector y for STATIONARY
     *  series with both means pre-computed.
     *  @param k     the lag parameter (0 <= k < n)
     *  @param mu    the pre-computed mean for this
     *  @param y     the other vector
     *  @param mu_y  the pre-computed mean for y
     */
    def ccov (k: Int, mu: Double, y: VectorD, mu_y: Double): Double =
        if k >= dim then flaw ("ccov", s"the vector is not long enough to compute ccov for lag k = $k")
        (Σ (0, dim-k) { i => (v(i) - mu) * (y.v(i+k) - mu_y) }) / (dim-1)
    end ccov

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for NON-STATIONARY series.
     *  @param k  the lag parameter (0 <= k < n) 
     */
    def acov_ (k: Int = 1): Double =
        if k >= dim then flaw ("acov_", s"the vector is not long enough to compute acov_ for lag k = $k")
        val n  = dim - k
        val ss = sums (k)
        val mu = ((ss._1 + ss._2) / n, (ss._2 + ss._3) / n)
        (Σ (0, n) { i => (v(i) - mu._1) * (v(i+k) - mu._2) }) / n
    end acov_

// Compute Auto-Correlation

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-correlation of this vector (assumes a STATIONARY
     *  process vector, if not its an approximation).
     *  @see https://otexts.com/fpp3/acf.html
     *  Estimator: mle vs. adjusted
     *  @see www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.yule_walker.html
     *  @param k         the lag parameter (0 <= k < n) 
     *  @param adjusted  whether to adjust to account for size difference of numerator and denominator
     */
    def acorr (k: Int = 1, adjusted: Boolean = true): Double =
        if k >= dim then flaw ("acorr", s"the vector is not long enough to compute acorr for lag k = $k")
        val z = this - mean
        val mle = (Σ (k, dim) { i => z(i) * z(i-k) }) / (z~^2).sum
        if adjusted then mle * dim / (dim-k) else mle
    end acorr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-correlation of this vector (assumes a STATIONARY
     *  process vector, if not its an approximation) when the mean has been pre-computed
     *  (passed in).
     *  @see https://otexts.com/fpp3/acf.html
     *  Estimator: mle vs. adjusted
     *  @see www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.yule_walker.html
     *  @param k         the lag parameter (0 <= k < n)
     *  @param mu        the pre-computed mean
     *  @param adjusted  whether to adjust to account for size difference of numerator and denominator
     */
    def acorr (k: Int, mu: Double, adjusted: Boolean): Double =
        if k >= dim then flaw ("acorr", s"the vector is not long enough to compute acorr for lag k = $k")
        val z = this - mu
        val mle = (Σ (k, dim) { i => z(i) * z(i-k) }) / (z~^2).sum
        if adjusted then mle * dim / (dim-k) else mle
    end acorr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag cross-correlation of this vector (assumes a STATIONARY
     *  process vector, if not its an approximation).
     *  @param y  the other vector
     *  @param k  the lag parameter (0 <= k < n)
     */
    def ccorr (y: VectorD, k: Int = 1): Double =
        if k >= dim then flaw ("ccorr", s"the vector is not long enough to compute ccorr for lag k = $k")
        val mu   = mean
        val mu_y = y.mean
        ccov (k, mu, y, mu_y) / (stdev * y.stdev)
    end ccorr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-correlation of this vector for a NON-STATIONARY series.
     *  @param k  the lag parameter (0 <= k < n) 
     */
    def acorr_ (k: Int = 1): Double =
        if k >= dim then flaw ("acorr_", s"the vector is not long enough to compute acorr_ for lag k = $k")
        val n   = dim - k
        val ss  = sums (k)
        val sq  = normSqs (k)
        val mu  = ((ss._1 + ss._2) / n, (ss._2 + ss._3) / n)
        val vr  = ((sq._1 + sq._2 - (mu._1 * mu._1) * n) / n,
                   (sq._2 + sq._3 - (mu._2 * mu._2) * n) / n)
        val s = Σ (0, n) { i => (v(i) - mu._1) * (v(i+k) - mu._2) }
        (s / n) / (math.sqrt (vr._1) * math.sqrt (vr._2))
    end acorr_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute Spearman's rank correlation of this vector with vector y.
     *  The `iqsort` method gives the rank order of a vector.
     *  @see  en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
     *  @param y  the other vector
     */
    infix def scorr (y: VectorD): Double =
        val rk1 = iqsort                                       // rank order for this vector
        val rk2 = y.iqsort                                     // rank order for vector y
        val s = Σ (v.indices) { i => (rk1(i) - rk2(i))~^2 }
        1 - 6 * s / (nd * (nd*nd - 1))
    end scorr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the skewness of this vector.  Negative skewness indicates the
     *  distribution is elongated on the left, zero skewness indicates it is
     *  symmetric, and positive skewness indicates it is elongated on the right.
     *  @see www.mathworks.com/help/stats/skewness.html
     *  >> E(X - μ)^3 / σ^3
     */
    def skew: Double   = ((this - mean)~^3).sum / (nd * stdev_ ~^3)
    def skew_ : Double = skew * math.sqrt (nd * (nd-1)) / (nd-2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the kurtosis of 'x' vector.  High kurtosis (> 3) indicates a 
     *  distribution with heavier tails than a Normal distribution.
     *  @see www.mathworks.com/help/stats/kurtosis.html
     *  >> E(X - μ)^4 / σ^4
     */
    def kurtosis: Double = ((this - mean)~^4).sum / (nd * variance_ ~^2)
    def kurtosis_ : Double = (nd-1) * ((nd+1) * kurtosis - 3 * (nd-1)) / ((nd-2) * (nd-3)) + 3

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a standardized version of the vector by subtracting the mean and
     *  dividing by the standard deviation (e.g., Normal -> Standard Normal).
     */
    def standardize: VectorD = (this - mean) / stdev
    def standardize2: VectorD = (this - mean) / (stdev + EPSILON)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the first 2 moments (mean mu and standard deviation sig).
     */
    def mu_sig: VectorD = VectorD (mean, stdev)

end VectorD


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VectorD` object is the companion object for the `VectorD` class.
 */
object VectorD:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorD` from an immutable indexed sequence of `Double`s.
     *  @param xs  the sequence/array of the `Double` numbers
     */
    def apply (xs: IIndexedSeq [Double]): VectorD = new VectorD (xs.size, xs.toArray)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorD` from a mutable indexed sequence of `Double`s.
     *  @param xs  the sequence/array of the `Double` numbers
     */
    def apply (xs: IndexedSeq [Double]): VectorD = new VectorD (xs.size, xs.toArray)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorD` from an aeeay of `Double`s.
     *  @param xs  the sequence/array of the `Double` numbers
     */
    def apply (xs: Array [Double]): VectorD = new VectorD (xs.length, xs)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorD` from one or more values (repeated values `Double`*).
     *  @param x   the first `Double` number
     *  @param xs  the varargs of `Double` numbers
     */
    def apply (x: Double, xs: Double*): VectorD = new VectorD (xs.size + 1, x +: xs.toArray)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorD` from a 2-tuple of doubles (`Double`, `Double`).
     *  @param x_y  the 2-tuple of doubles
     */
    def apply (x_y: (Double, Double)): VectorD = new VectorD (2, Array (x_y._1, x_y._2))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorD` from one or more values (repeated values String*).
     *  For numeric types, assign missing value indicator upon format failure.
     *  @param x   the first string
     *  @param xs  the varargs of strings
     */
    def apply (x: String, xs: String*): VectorD =
        val y = new VectorD (1 + xs.length)
        cfor (y.indices) { i =>
            y(i) = if i == 0 then x.mkDouble else xs(i-1).mkDouble
        } // cfor
        y
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorD` from a mutable indexed sequence of `String`.
     *  @param xs  the sequence/array of the `String` numbers
     */
    def fromStrings (xs: IndexedSeq [String]): VectorD = VectorD (xs.map (_.mkDouble))
    def fromValueTypes (xs: IndexedSeq [ValueType]): VectorD = VectorD (xs.map (_.toDouble))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorD` from a mutable indexed sequence of `TimeNum`.
     *  FIX - for numeric types, assign missing value indicator upon format failure.
     *  @param xs  the sequence/array of the `TimeNum` numbers
     */
    def fromTimeNums (xs: IndexedSeq [TimeNum]): VectorD = VectorD (xs.map (_.toDouble))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a `VectorD` consisting of a sequence of increasing integers in a range.
     *  VectorD (for i <- r yield i.toDouble)
     *  @param r  the range of values
     */
    inline def range (r: Range): VectorD = range (r.start, r.end)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a `VectorD` consisting of a sequence of increasing integers in a range.
     *  Usage: VectorD.range (1, 4) = (1, 2, 3)
     *  @param start  the start of range (inclusive)
     *  @param end    the end of range (exclusive)
     */
    def range (start: Int, end: Int): VectorD =
        val n = end - start
        val a = Array.ofDim [Double] (n)
        cfor (start, end) { i => a(i - start) = i }
        new VectorD (n, a)
    end range

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorD` with n elements and fill it with the value x.
     *  @param n  the number of elements
     *  @param x  the value to assign to all elements
     */
    def fill (n: Int)(x: Double): VectorD = new VectorD (n, Array.fill (n)(x))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a one vector (all elements are one) of length n.
     *  @param size  the size of the new vector
     */
    def one (n: Int): VectorD = new VectorD (n, Array.fill (n)(1.0))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a vector of the form (0, ... 1, ... 0) where the 1 is at position j.
     *  @param j     the position to place the 1
     *  @param size  the size of the vector (upper bound = size - 1)
     */
    def oneAt (j: Int, size: Int): VectorD =
        val x = new VectorD (size)
        x.v(j) = 1.0
        x
    end oneAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** A null vector of type `VectorD`.
     */
    val nullv: VectorD = null.asInstanceOf [VectorD]

end VectorD


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VectorDOps` object provides extension methods to support scalar op vector
 *  operations, so that one can write 2.0 + x as well as x + 2.0.
 */
object VectorDOps:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Extension methods for `Double` <+> `VectorD`.  Compute the element-wise sum
     *  (or difference, product, quotient) of scalar a and vector x.
     *  @param a  the scalar first operand
     *  @param x  the vector second operand
     */
    extension (a: Double)
        inline def + (x: VectorD): VectorD = x + a
        inline def - (x: VectorD): VectorD = -x + a
        inline def * (x: VectorD): VectorD = x * a
        inline def / (x: VectorD): VectorD = x.recip * a

end VectorDOps


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorDTest` main function tests the operations provided by the `VectorD` class.
 *  Only the most commonly used inherited operations are shown.
 *  @see mutable.IndexedSeq for a complete list
 *  > runMain scalation.mathstat.vectorDTest
 */
@main def vectorDTest (): Unit =

    import VectorDOps._

    val x = VectorD (1, 2, 3)
    val y = VectorD (4, 6, 5)
    val z = VectorD (4, 6, 5)
    val w = VectorD (3, 4, 5, 5)
    val u = VectorD ("1", "2", "3", "4")
    val a = 2
    val b = 3.0

    banner ("Given Vectors:")
    println (s"x = $x")
    println (s"y = $y")
    println (s"z = $z")
    println (s"w = $w")
    println (s"u = $u")

    banner ("Inherited Operations:")

    println (s"x == y              = ${x == y}")                 // vector equality
    println (s"y == z              = ${y == z}")
    println (s"x != y              = ${x != y}")                 // vector inequality
    println (s"x < y               = ${x < y}")                  // less than
    println (s"x <= y              = ${x <= y}")                 // less than or equal
    println (s"x > y               = ${x > y}")                  // greater then
    println (s"x >= y              = ${x >= y}")                 // greater then or equal

    println (s"x `contains` 2      = ${x `contains` 2}")         // element contained in vector
    println (s"x `contains` 4      = ${x `contains` 4}")
    println (s"x.exists (_ > 2)    = ${x.exists (_ > 2)}")       // existence of element satisfying predicate
    println (s"x.groupBy (_ > 2)   = ${x.groupBy (_ > 2)}")      // group according function values
    println (s"x.indexOf (2)       = ${x.indexOf (2)}")          // index of first element equaling
    println (s"x.indexWhere (_ > 2) = ${x.indexWhere (_ > 2)}")  // index of first element satisfying predicate
    println (s"x.indices           = ${x.indices}")              // indices of the vector
    println (s"u.map (_ * 2)       = ${u.map (_ * 2)}")          // map vector elements using the function
    println (s"x.max               = ${x.max}")                  // maximum element
    println (s"x.min               = ${x.min}")                  // minimum element
    println (s"x.product           = ${x.product}")              // product of all elements
    println (s"x.sum               = ${x.sum}")                  // sum of all elements
    println (s"w.toArray           = ${stringOf (w.toArray)}")   // convert to array - use stringOf to print arrays
    println (s"w.toSet             = ${w.toSet}")                // convert to set

    banner ("Implemented Operations:")

    println (s"x(2)                = ${x(2)}")                   // value at index
    println (s"x(0 until 2)        = ${x(0 until 2)}")           // values in exclusive range

    println (s"-x                  = ${-x}")                     // unary minus
    println (s"x + y               = ${x + y}")                  // element-wise vector addition
    println (s"x - y               = ${x - y}")                  // element-wise vector subtraction
    println (s"x * y               = ${x * y}")                  // element-wise vector multiplication
    println (s"x / y               = ${x / y}")                  // element-wise vector division
    println (s"x ++ y              = ${x ++ y}")                 // concatenate vectors

    println (s"x + a               = ${x + a}")                  // add scalar a
    println (s"x - a               = ${x - a}")                  // subtract scalar a
    println (s"x * a               = ${x * a}")                  // multiply by scalar a
    println (s"x / a               = ${x / a}")                  // divide by scalar a
    println (s"x ~^ a              = ${x ~^ a}")                 // raise to power of scalar a
    println (s"a +: y              = ${a +: x}")                 // prepend scalar a
    println (s"x :+ a              = ${x :+ a}")                 // append scalar a

    println (s"b + x               = ${b + x}")                  // add scalar b and x
    println (s"b - x               = ${b - x}")                  // subtract from scalar b, x
    println (s"b * x               = ${b * x}")                  // multiply by scalar b and x
    println (s"b / x               = ${b / x}")                  // divide scalar b by x

    println (s"(x-y).abs           = ${(x-y).abs}")              // absolute value
    println (s"x.cnormSq           = ${x.cnormSq}")              // center norm squared
    println (s"x.cumulate          = ${x.cumulate}")             // cumulate all prior values
    println (s"x diff w            = ${x diff w}")               // multi-set difference
    println (s"w.distinct          = ${w.distinct}")             // extract distinct values
    println (s"x dot y             = ${x dot y}")                // dot product
    println (s"u.filter (_ > 2)    = ${u.filter (_ > 2)}")       // filter on predicate
    println (s"u.filterNot (_ > 2) = ${u.filterNot (_ > 2)}")    // filter on not predicate
    println (s"u.filterPos (_ > 2) = ${u.filterPos (_ > 2)}")    // filter return indices
    println (s"x intersect w       = ${x intersect w}")          // multi-set intersection
    println (s"y.iselsort          = ${stringOf (y.iselsort)}")  // indirect QuickSort in ascending order
    println (s"y.iqsort            = ${stringOf (y.iqsort)}")    // indirect SelectionSort in ascending order
    println (s"x.norm              = ${x.norm}")                 // Euclidean norm
    println (s"x.normalize         = ${x.normalize}")            // normalize the vector to a unit vector
    println (s"x.normalize1        = ${x.normalize1}")           // normalize the vector to max 1 vector
    println (s"x.normSq            = ${x.normSq}")               // Euclidean norm squared
    println (s"x.norm1             = ${x.norm1}")                // Manhattan norm
    println (s"x.recip             = ${x.recip}")                // reciprocal
    println (s"x.reverse           = ${x.reverse}")              // reverse elements in vector
    println (s"y.sorted            = ${y.sorted}")               // sort in ascending order
    println (s"y.sortWith (_ > _)  = ${y.sortWith (_ > _)}")     // sort in descending order
    println (s"y.toProbabilty      = ${y.toProbability}")        // convert to a probability vector

    banner ("Implemented Statistical Operations: (_ => population")

    println (s"x.acorr ()          = ${x.acorr ()}")             // auto-correlation
    println (s"x.acorr_ ()         = ${x.acorr_ ()}")            // auto-correlation - non-stationary case
    println (s"x.acov ()           = ${x.acov ()}")              // auto-covariance
    println (s"x.acov_ ()          = ${x.acov_ ()}")             // auto-covariance - non-stationary case
    println (s"x corr y            = ${x corr y}")               // correlation
    println (s"x cov y             = ${x cov y}")                // covariance      
    println (s"x cov_ y            = ${x cov_ y}")
    println (s"x.kurtosis          = ${x.kurtosis}")             // kurtosis
    println (s"x.mean              = ${x.mean}")                 // mean
    println (s"u.median ()         = ${u.median ()}")            // median
    println (s"u.median_           = ${u.median_}")              // averaged median
    println (s"x ms y              = ${x.ms}")                   // mean square
    println (s"x rms y             = ${x.rms}")                  // root mean square
    println (s"x scorr y           = ${x scorr y}")              // Spearman's rank correlation
    println (s"x.skew              = ${x.skew}")                 // skewness
    println (s"x.skew_             = ${x.skew_}")
    println (s"x.standardize       = ${x.standardize}")          // standardize the vector
    println (s"x.stdev             = ${x.stdev}")                // standard deviation
    println (s"x.stdev_            = ${x.stdev_}")
    println (s"x.variance          = ${x.variance}")             // variance
    println (s"x.variance_         = ${x.variance_}")

end vectorDTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorDTest2` main function tests the indirect sorting operations provided
 *  by the `VectorD` class.
 *  > runMain scalation.mathstat.vectorDTest2
 */
@main def vectorDTest2 (): Unit =

    val y = VectorD (4, 6, 5, 2, 8, 3, 9, 7, 5, 2)
    val is = y.iselsort
    val iq = y.iqsort

    banner ("Index Order")
    println (s"y.iselsort = ${stringOf (is)}")              // indirect QuickSort in ascending order
    println (s"y.iqsort   = ${stringOf (iq)}")              // indirect SelectionSort in ascending order

    banner ("Value Order")
    println (s"y.iselsort = ${stringOf (is.map (y(_)))}")   // indirect QuickSort in ascending order
    println (s"y.iqsort   = ${stringOf (iq.map (y(_)))}")   // indirect SelectionSort in ascending order

end vectorDTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorDTest3` main function tests covariance and correlation methods provided
 *  by the `VectorD` class.  Okay to use `MatrixD` in `VectorD` test code only.
 *  > runMain scalation.mathstat.vectorDTest3
 */
@main def vectorDTest3 (): Unit =

    val x = VectorD (1, 2, 3, 4, 5, 6)
    val y = VectorD (1, 3, 4, 6, 4, 3)
    val z = MatrixD (x, y).transpose

    println (s"z = $z")

    banner ("Sample Covariance")
    println (s"x cov y  = ${x cov y}")

    banner ("Population Covariance")
    println (s"x cov_ y = ${x cov_ y}")

    banner ("Correlation")
    println (s"x corr y = ${x corr y}")

    banner ("Covariance Matrix (cov)")
    println (s"z.cov = ${z.cov}")

    banner ("Population Covariance Matrix (cov_)")
    println (s"z.cov_ = ${z.cov_}")

    banner ("Correlation Matrix (corr)")
    println (s"z.corr = ${z.corr}")

end vectorDTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorDTest4` main function tests auto-covariance and auto-correlation methods
 *  provided by the `VectorD` class.
 *  > runMain scalation.mathstat.vectorDTest4
 */
@main def vectorDTest4 (): Unit =

    val y    = VectorD (1, 2, 4, 7, 9, 8, 6, 5, 3)
    val z    = VectorD (2, 1, 9, 7, 4, 8, 6, 3, 5)
    val mu_y = y.mean
    val mu_z = z.mean
    val sd_y = y.stdev
    val sd_z = z.stdev
    val sd_p = sd_y * sd_z                                   // product of stdev's

    println (s"y    = $y")
    println (s"z    = $z")
    println (s"mu_y = $mu_y")
    println (s"mu_z = $mu_z")
    println (s"sd_y = $sd_y")
    println (s"sd_z = $sd_z")
    println (s"sd_p = $sd_p")

    banner (s"Regular Covariance and Correlation of y and z")
    println (s"Covariance   y cov z          = ${y cov z}")
    println (s"Correlation  y corr z         = ${y corr z}")
    println (s"Correlation  (y cov z) / sd_p = ${(y cov z) / sd_p}")

    for k <- 0 to 2 do
        banner (s"Lag-$k Auto-Covariance")
        println (s"Stationary     y acov $k   = ${y.acov (k)}")
        println (s"Non-Stationary y acov_ $k  = ${y.acov_ (k)}")

        banner (s"Lag-$k Auto-Correlation")
        println (s"Stationary     y acorr $k  = ${y.acorr (k)}")
        println (s"Non-Stationary y acorr_ $k = ${y.acorr (k)}")

        banner (s"Lag-$k Cross-Covariance")
        println (s"Stationary     y.ccov ($k, mu_y, z, mu_z)  = ${y.ccov (k, mu_y, z, mu_z)}")

        banner (s"Lag-$k Cross-Correlation")
        println (s"Stationary     y.ccorr (z, $k) = ${y.ccorr (z, k)}")
    end for

end vectorDTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorDTest5` main function tests convolution operators defined in the VectorD
 *  class:  conv (*+), conv_, convs and convs_.
 *  @see https://gregorygundersen.com/blog/2017/02/24/cnns/
 *  > runMain scalation.mathstat.vectorDTest5
 */
@main def vectorDTest5 (): Unit =

    val c = VectorD (1.0/3, 1.0/3, 1.0/3)
    val x = VectorD (10, 9, 10, 9, 10, 1, 10, 9, 10, 8)

    banner ("Convolution Operators")
    println (s"c conv x  = ${c conv x}")                 // conv   valid convolution, no reversal
    println (s"c *+ x    = ${c *+ x}")                   // *+     valid convolution, no reversal
    println (s"c conv_ x = ${c conv_ x}")                // conv_  valid convolution, with reversal
    println (s"c convs x = ${c convs x}")                // convs  same convolution, with reversal
    println (s"c *~+ x   = ${c *~+ x}")                  // *~+    same convolution, with reversal
    println (s"c convf x = ${c convf x}")                // convf  full convolution, with reversal
    println (s"c *++ x   = ${c convf x}")                // *++    full convolution, with reversal

end vectorDTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorDTest6` test the dot product implementations.  Run this code for performance results.
 *  Note, due to JIT, reordering the code may change the relative performance.  Generally, case 0
 *  "x dot x (VectorD dot VectorD)" is the fastest.
 *  @see Timer.scala
 *  > runMain scalation.mathstat.vectorDTest6
 */
@main def vectorDTest6 (): Unit =

    import scala.math.sqrt

    val n    = 1                          // number of repeats to get average time (gauge's amplify parameter)
    val skip = false                      // whether to skip timing the first time through code (slow due to JIT)

    val tims = Array.ofDim [Double] (4)
    val dotp = Array.ofDim [Double] (4)

    val x: VectorD = new VectorD (100000); cfor (0, 100000) { i => x(i) = sqrt (i) }
    val y: IIndexedSeq [Double] = for i <- 0 until 100000 yield sqrt (i)

// Collect results

    tims(0) = gauge (n, skip) { dotp(0) = x dot x }

    tims(1) = gauge (n, skip) { dotp(1) = x dot y }

    tims(2) = gauge (n, skip) { dotp(2) = x dot x }

    tims(3) = gauge (n, skip) { dotp(3) = x dot y }

// Show results

    banner ("x dot x (VectorD dot VectorD)") 
    println (s"case 0: dotp = ${dotp(0)}")
    println (s"case 0: time = ${tims(0)}")

    banner ("x dot y (VectorD dot IndexedSeq)")
    println (s"case 1: dotp = ${dotp(1)}")
    println (s"case 1: time = ${tims(1)}")

    banner ("x dot x (VectorD dot VectorD) again") 
    println (s"case 0: dotp = ${dotp(2)}")
    println (s"case 0: time = ${tims(2)}")

    banner ("x dot y (VectorD dot IndexedSeq) again")
    println (s"case 1: dotp = ${dotp(3)}")
    println (s"case 1: time = ${tims(3)}")

end vectorDTest6

