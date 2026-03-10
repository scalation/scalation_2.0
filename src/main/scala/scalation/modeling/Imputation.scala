
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Santosh Uttam Bobade, Vinay Kumar Bingi, Mohammad Toutiaee, John Miller
 *  @version 2.0
 *  @date    Sat Jun 9 14:09:25 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Collection of Simple Imputation Techniques for Missing Values or Outliers
 *
 *  Common Imputation Techniques:  Multiple Imputation of Chained Equations (MICE),
 *  Regression Imputation (RI), kNN, Decision Trees, Random Forests (missForest), SoftImpute
 *
 *  FIX -- implement Multiple Imputation of Chained Equations (MICE)
 *  @see https://pmc.ncbi.nlm.nih.gov/articles/PMC3074241/
 *  @see https://www.machinelearningplus.com/machine-learning/mice-imputation
 */

package scalation
package modeling

import scala.annotation.unused

import scalation.mathstat._
import scalation.random.Normal

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Imputation` trait specifies an imputation operation called impute to be defined
 *  by the objects implementing it, i.e.,
 *      `ImputeMICE`        - impute missing values using MICE
 *      `ImputeMRegression` - impute missing values using `Regression` (RI)
 *      `ImputeSRegression` - impute missing values using `SimpleRegression`
 *      `ImputeForward`     - impute missing values using previous values and slopes
 *      `ImputeBackward`    - impute missing values using subsequent values and slopes
 *      `ImputeMean`        - impute missing values using the filtered mean
 *      `ImputeMedian`      - impute missing values using the filtered median
 *      `ImputeNormal`      - impute missing values using the median of Normal random variates
 *      `ImputeMovingAvg`   - impute missing values using the moving average
 *      `ImputeNormalWin`   - impute missing values using the median of Normal random variates for a window
 */
trait Imputation:

    protected val debug   = debugf ("Imputation", false)               // debug function
    protected val DAMPEN  = 0.8                                        // dampening factor for slope
    protected val q       = 5                                          // number of elements in moving average

    protected var missVal = NO_DOUBLE                                  // default for missing value indicator
    protected var dist    = 3                                          // distance before and after point

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the missing value missVal to the new missing value indicator missVal_.
     *  @param missVal_  the new missing value indicator
     */
    def setMissVal (missVal_ : Double): Unit = missVal = missVal_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the distance dist to the new value dist_.
     *  @param dist_  the new value for the distance
     */
    def setDist (dist_ : Int): Unit = dist = dist_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for vector x at index i.
     *  Does not modify the vector.
     *  @param x  the vector with missing values
     *  @param i  the index position for which to impute a value
     */
    def imputeAt (x: VectorD, i: Int): Double
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in vector x from index i.
     *  The type (Int, Double) returns (vector index for imputation, imputed value).
     *  Does not modify the vector.
     *  @param x  the vector with missing values
     *  @param i  the starting index to look for missing values
     */
    def impute (x: VectorD, i: Int = 0): (Int, Double) = findMissing (x, i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Replace all missing values in vector x with imputed values.  Will change
     *  the values in vector x.  Make a copy to preserve values x.copy.
     *  @param x  the vector with missing values
     */
    def imputeAll (x: VectorD): VectorD =
        var i = 0                                                      // starting index
        while true do
            val (im, z) = impute (x, i)                                // get index of missing and imputed value
            if im != -1 then x(im) = z else return x                   // update the vector or return
            i = im + 1                                                 // set index to missing position + 1
        end while
        x                                                              // return updated vector
    end imputeAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Replace all missing values in matrix x with imputed values.  Will change
     *  the values in matrix x.  Make a copy to preserve values x.copy.
     *  @param x  the matrix with missing values
     */
    def impute (x: MatrixD): MatrixD =
        for j <- x.indices2 do imputeAll (x(?, j))
        x
    end impute

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in column c from index i.
     *  The type (Int, Double) returns (vector index for imputation, imputed value).
     *  Does not modify the column.
     *  @param c  the column with missing values
     *  @param i  the starting index to look for missing values
     *
    def imputeCol (c: VectorS, i: Int = 0): (Int, VectorS) =
        val x = c.toDouble                                             // convert to a double vector
        val (im, z) = impute (x, i)                                    // get index of missing and imputed value
        (im, VectorS.fromDouble (c, z))                                // convert back to column type
    end imputeCol
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the median of three normally distributed random numbers.
     *  @param mu    the mean
     *  @param sig2  the variance
     */
    protected def normalMedian (mu: Double, sig2: Double): Double =
        val rn = Normal (mu, sig2)                                     // Normal random variate generator
        median3 (rn.gen, rn.gen, rn.gen)
    end normalMedian 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the next non-missing value in vector x from index i.  If none, return missVal.
     *  @param x  the vector to be searched for a non-missing value
     *  @param i  the starting index to look for non-missing value
     */
    protected def nextVal (x: VectorD, i: Int): Double =
        val j = x.indexWhere (_ != missVal, i)                         // find first non-missing from i
        debug ("nextVal", s"unable to find any non-missing values from $i to ${x.dim -1}")
        if j >= 0 then x(j) else missVal
    end nextVal

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the previous non-missing value in vector x from index i.  If none, return missVal.
     *  @param x  the vector to be searched (backwards) for a non-missing value
     *  @param i  the starting index to look for non-missing value
     */
    protected def prevVal (x: VectorD, i: Int): Double =
        val j = x.lastIndexWhere (_ != missVal, i)                     // find first non-missing from i backward
        debug ("prevVal", s"unable to find any non-missing values from $i downto 0")
        if j >= 0 then x(j) else missVal
    end prevVal

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the index of first missing value in vector x from index i and the
     *  new imputed value.
     *  @param x  the vector with missing values
     *  @param i  the starting index to look for missing value
     */
    def findMissing (x: VectorD, i: Int = 0): (Int, Double) =
        val j = x.indexOf (missVal, i)                                 // find first missing from i
        if j >= 0 then (j,  imputeAt (x, j))                           // return (index, imputed value)
        else           (-1, nextVal (x, i))                            // return (not found, first value)
    end findMissing

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the index of last missing value in vector x from index i and the
     *  new imputed value.
     *  @param x   the vector with missing values
     *  @param i_  the starting index to look for missing value
     */
    def findLastMissing (x: VectorD, i_ : Int = -1): (Int, Double) =
        val i = if i_ < 0 then x.dim-1 else i_
        val j = x.lastIndexOf (missVal, i)                             // find last missing from i
        if j >= 0 then (j,  imputeAt (x, j))                           // return (index, imputed value)
        else           (-1, prevVal (x, i))                            // return (not found, last value)
    end findLastMissing

end Imputation


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeMICE` object imputes missing values using MICE.
 *  Use the columns in matrix z to impute values for target vector x.
 */
object ImputeMICE extends Imputation:

    private val max_iter   = 5                                         // maximum number of impute steps
    private var z: MatrixD = null                                      // matrix holding the dataset

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the value for the z matrix containing the dataset (consisting of multiple
     *  columns/variables) where missing values are to be imputed.
     *  @param z_  the matrix to be assigned
     */
    def setZ (z_ : MatrixD): Unit = z = z_

    def imputeAt (x: VectorD, i: Int): Double =
         throw new UnsupportedOperationException ("ImputeImputeMICE: 'imputeAt' not supported, use 'imputeAll'")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make initial imputation by replacing missing values with the column means.
     *  Return the positions imputed and the new imputed matrix.
     */
    def initialImpute (): (Array [IndexedSeq [Int]], MatrixD) =
        val idx = Array.fill (z.dim2) (IndexedSeq [Int] ())            // hold the indices of missing values
        val zz  = z.copy                                               // put imputed values in a copy of the z matrix
        for j <- z.indices2 do
            idx(j) = for i <- z.indices if z(i, j) != missVal yield i  // find indices where z has missing values
            ImputeMean.imputeAll (zz(?, j))                            // replace missing values with column mean
        (idx, zz)
    end initialImpute

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform a MICE imputation step by fixing each column that has missing values.
     */
    def imputeStep (): Unit =
        val (idx, zz) = initialImpute ()
        for j <- z.indices2 if idx(j).size < z.dim2 do                 // look for columns with missing values
            val y  = zz(?, j)                                          // column to impute values
            val x  = zz.not(?, j)                                      // other columns used to predict the impute column
            val rg = new Regression (x, y, null)                       // create a multiple regression model
            rg.train (x(idx(j)), y(idx(j)))                            // train using rows with no missing values in y
            val pidx = IndexedSeq.range (0, x.dim) diff idx(j)         // indices where values are to be predicted
            for i <- pidx do zz(i, j) = rg.predict (zz(i))                // call predict to get values where y had missing values
                                                                       // FIX - may add noise to prediction
        end for
    end imputeStep

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Replace all missing values in vector x with imputed values.  Will change
     *  the values in vector x.  Make a copy to preserve values x.copy.
     *  @param x  the vector with missing values (target column)
     */
    override def imputeAll (@unused x: VectorD): VectorD =
        for _ <- 1 to max_iter do imputeStep ()
        null
    end imputeAll

end ImputeMICE

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeMRegression` object imputes missing values using multiple `Regression`.
 *  Use the columns in matrix z to impute values for target vector x.
 */
object ImputeMRegression extends Imputation:

    private var z: MatrixD = null                                      // matrix holding the dataset

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the value for the z matrix containing the other columns used to predict
     *  values for target column x.
     *  @param z_  the matrix to be assigned
     */
    def setZ (z_ : MatrixD): Unit = z = z_

    def imputeAt (x: VectorD, i: Int): Double =
         throw new UnsupportedOperationException ("ImputeImputeMRegression: 'imputeAt' not supported, use 'imputeAll'")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Replace all missing values in vector x with imputed values.  Will change
     *  the values in vector x.  Make a copy to preserve values x.copy.
     *  @param x  the vector with missing values (target column)
     */
    override def imputeAll (x: VectorD): VectorD =
        val idx = for i <- x.indices if x(i) != missVal yield i        // find indices where x does not have missing values
        val rg  = new Regression (z, x, null)                          // create a multiple regression model
        rg.train (z(idx), x(idx))                                      // train using rows with no missing values in x
        val pidx = IndexedSeq.range (0, x.dim) diff idx                // indices where values are to be predicted
        for i <- pidx do x(i) = rg.predict (z(i))                      // call predict to get values where x had missing values
        x(pidx)                                                        // return the new predicted values for x
    end imputeAll

end ImputeMRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeSRegression` object imputes missing values using `SimpleRegression`.
 */
object ImputeSRegression extends Imputation:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in vector x from index i
     *  using `SimpleRegression`.
     *  @param x  the vector with missing values
     *  @param i  the starting index to look for missing values
     */
    def imputeAt (x: VectorD, i: Int): Double =
        val xf = x.filter (_ != missVal)                               // x-vector with missing values removed
        val t  = VectorD.range (0, xf.dim)                             // vector = 0, 1, ..., xf.dim-1
        val rg = SimpleRegression (t, xf, null)                        // Simple Regression model of xf onto t
        rg.train (rg.getX, xf)                                         // train the model: X = [1, t], y = xf
        rg.predict (VectorD (1, t(i)))                                 // return the predicted value for the i-th element
    end imputeAt

end ImputeSRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeForward` object imputes missing values using the previous value and slope.
 */
object ImputeForward extends Imputation:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in vector x from index i
     *  using the previous value and slope.
     *  @param x  the vector with missing values
     *  @param i  the starting index to look for missing values
     */
    def imputeAt (x: VectorD, i: Int): Double =
        if i == 0 then      nextVal (x, 1)                             // next non-missing value
        else if i == 1 then x(0)                                       // first value
        else x(i-1) + DAMPEN * (x(i-1) - x(i-2))                       // slope adjusted previous value
    end imputeAt

end ImputeForward


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeBackward` object imputes missing values using the subsequent value and slope.
 */
object ImputeBackward extends Imputation:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in vector x from index i
     *  using the next value and slope.
     *  @param x   the vector with missing values
     *  @param i_  the starting index to look for missing values
     */
    def imputeAt (x: VectorD, i_ : Int): Double =
        val l = x.dim - 1                                              // last index position
        val i = if i_ < 0 then l else i_                               // -1 => last position

        if i == l then        prevVal (x, l-1)                         // previous non-missing value
        else if i == l-1 then x(l)                                     // last value 
        else x(i+1) - DAMPEN * (x(i+2) - x(i+1))                       // slope adjusted next value
    end imputeAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in vector x from index i.
     *  The type (Int, Double) returns (vector index for imputation, imputed value).
     *  Does not modify the vector.
     *  @param x  the vector with missing values
     *  @param i  the starting index to look for missing values
     */
    override def impute (x: VectorD, i: Int = -1): (Int, Double) = findLastMissing (x, i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Replace all missing values (in reverse) in vector x with imputed values.
     *  @param x  the vector with missing values
     */
    override def imputeAll (x: VectorD): VectorD =
        var i = x.dim - 1                                              // starting index
        while true do
            val (im, z) = impute (x, i)                                // get index of missing and imputed value
            if im != -1 then x(im) = z else return x                   // update the vector or return
            i = im - 1                                                 // set index to missing position + 1
        end while
        x                                                              // return updated vector
    end imputeAll

end ImputeBackward


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeMean` object imputes missing values using the filtered mean.
 */
object ImputeMean extends Imputation:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in vector x from index i
     *  using the filtered mean.
     *  @param x  the vector with missing values
     *  @param i  the starting index to look for missing values (ignored)
     */
    def imputeAt (x: VectorD, i: Int):  Double = 
        val xf = x.filter (_ != missVal)
        xf.mean
    end imputeAt

end ImputeMean


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeMean` object imputes missing values using the filtered median.
 */
object ImputeMedian extends Imputation:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in vector x from index i
     *  using the filtered mean.
     *  @param x  the vector with missing values
     *  @param i  the starting index to look for missing values (ignored)
     */
    def imputeAt (x: VectorD, i: Int):  Double =
        val xf = x.filter (_ != missVal)
        xf.median ()
    end imputeAt

end ImputeMedian


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeNormal` object imputes missing values using the median Normal variates.
 */
object ImputeNormal extends Imputation:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in vector x from index i
     *  using the median of three Normally distributed random values.
     *  @param x  the vector with missing values
     *  @param i  the starting index to look for missing values (ignored)
     */
    def imputeAt (x: VectorD, i: Int): Double =
        val xf = x.filter (_ != missVal)
        normalMedian (xf.mean, xf.variance)
    end imputeAt

end ImputeNormal


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeMovingAvg` object imputes missing values using the moving average.
 */
object ImputeMovingAvg extends Imputation:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute a value for the first missing value in vector x from index i
     *  using the moving average of the last dist values.
     *  @param x  the vector with missing values
     *  @param i  the starting index to look for missing values
     */
    def imputeAt (x: VectorD, i: Int): Double =
        var (sum, cnt) = (0.0, 0)
        for k <- i - dist to i + dist do
            if k >= 0 && k < x.dim && k != i && x(k) != missVal then { sum += x(k); cnt += 1 }
        end for
        sum / cnt
    end imputeAt

end ImputeMovingAvg


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ImputeNormalWin` object imputes the missing values in the vector using
 *  Normal Distribution for a sliding window.
 */
object ImputeNormalWin extends Imputation:

    def imputeAt (x: VectorD, i: Int): Double =
        throw new UnsupportedOperationException ("ImputeNormalWin: 'imputeAt' not supported, use 'imputeAll'")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Impute all the missing values in vector x using Normal Distribution for
     *  a sliding window.
     *  @param q  size of the sliding window
     */
    override def imputeAll (x: VectorD): VectorD =
        val z    = new VectorD (x.dim)
        val sumq = new SumSqQueue (q)
        sumq    += nextVal (x, 0)                                      // prime with first non-missing value

        for i <- x.indices do
            debug ("imputeAll", s"mu = ${sumq.mean}, sig2 = ${sumq.variance}")
            z(i) = if x(i) == missVal then normalMedian (sumq.mean, sumq.variance)
                   else x(i)
            sumq += z(i)
        end for

        z
    end imputeAll

end ImputeNormalWin


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `imputationTest` main function is used to test the objects extending the
 *  `Imputation` trait.
 *  > runMain scalation.modeling.imputationTest
 */
@main def imputationTest (): Unit =

    val x  = VectorD (1, 2, 3, 4, NO_DOUBLE, 6, 7, 8, 9)
    val x2 = x.copy
    val x3 = x.copy
    var iv = (-1, NO_DOUBLE)

    banner ("ImputeSRegression.impute")
    iv = ImputeSRegression.impute (x)
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeForward.impute")
    iv = ImputeForward.impute (x)
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeBackward.impute")
    iv = ImputeBackward.impute (x)
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeMean.impute")
    iv = ImputeMean.impute (x)
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeNormal.impute")
    iv = ImputeNormal.impute (x)
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeMovingAvg.impute")
    iv = ImputeMovingAvg.impute (x)
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeSRegression.imputeAll")
    println ("x3 = " + ImputeSRegression.imputeAll (x3.copy))

    banner ("ImputeForward.imputeAll")
    println ("x3 = " + ImputeForward.imputeAll (x3.copy))

    banner ("ImputeBackward.imputeAll")
    println ("x3 = " + ImputeBackward.imputeAll (x3.copy))

    banner ("ImputeMean.imputeAll")
    println ("x3 = " + ImputeMean.imputeAll (x3.copy))

    banner ("ImputeNormal.imputeAll")
    println ("x3 = " + ImputeNormal.imputeAll (x3.copy))

    banner ("ImputeMovingAvg.imputeAll")
    println ("x3 = " + ImputeMovingAvg.imputeAll (x3.copy))

    banner ("ImputeNormalWin.imputeAll")
    println ("x3 = " + ImputeNormalWin.imputeAll (x3.copy))

end imputationTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `imputationTest2` main function is used to test the objects extending the
 *  `Imputation` trait.
 *  > runMain scalation.modeling.imputationTest2
 */
@main def imputationTest2 (): Unit =

    val x  = VectorD (NO_DOUBLE, NO_DOUBLE, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var x2 = null.asInstanceOf [VectorD]
    val x3 = x.copy
    var iv = (-1, NO_DOUBLE)

    banner ("ImputeSRegression.impute")
    iv = ImputeSRegression.impute (x)
    x2 = x.copy
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeForward.impute")
    iv = ImputeForward.impute (x)
    x2 = x.copy
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeBackward.impute")
    iv = ImputeBackward.impute (x)
    x2 = x.copy
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeMean.impute")
    iv = ImputeMean.impute (x)
    x2 = x.copy
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeNormal.impute")
    iv = ImputeNormal.impute (x)
    x2 = x.copy
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeMovingAvg.impute")
    iv = ImputeMovingAvg.impute (x)
    x2 = x.copy
    x2(iv._1) = iv._2
    println (s"x  = $x")
    println (s"x2 = $x2")

    banner ("ImputeSRegression.imputeAll")
    println ("x3 = " + ImputeSRegression.imputeAll (x3.copy))

    banner ("ImputeForward.imputeAll")
    println ("x3 = " + ImputeForward.imputeAll (x3.copy))

    banner ("ImputeBackward.imputeAll")
    println ("x3 = " + ImputeBackward.imputeAll (x3.copy))

    banner ("ImputeMean.imputeAll")
    println ("x3 = " + ImputeMean.imputeAll (x3.copy))

    banner ("ImputeNormal.imputeAll")
    println ("x3 = " + ImputeNormal.imputeAll (x3.copy))

    banner ("ImputeMovingAvg.imputeAll")
    println ("x3 = " + ImputeMovingAvg.imputeAll (x3.copy))

    banner ("ImputeNormalWin.imputeAll")
    println ("x3 = " + ImputeNormalWin.imputeAll (x3.copy))

end imputationTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `imputationTest3` main function imputes a missing value for the Texas Temperatures
 *  dataset that contains temperatures from counties in Texas where the variables/factors
 *  to consider are Latitude (x1), Elevation (x2) and Longitude (x3).  The model equation
 *  is the following:
 *      y  =  b dot x  =  b0 + b1*x1 + b2*x2 + b3*x3
 *  > runMain scalation.modeling.imputationTest3
 */
@main def imputationTest3 (): Unit =

    // 16 data points:         one      x1      x2       x3     y
    //                                 Lat    Elev     Long  Temp        County
    val xy = MatrixD ((16, 5), 1.0, 29.767,   41.0,  95.367, 56.0,       // Harris
                               1.0, 32.850,  440.0,  96.850, 48.0,       // Dallas
                               1.0, 26.933,   25.0,  97.800, 60.0,       // Kennedy
                               1.0, 31.950, 2851.0, 102.183, 46.0,       // Midland
                               1.0, 34.800, 3840.0, 102.467, 38.0,       // Deaf Smith
                               1.0, 33.450, 1461.0,  99.633, 46.0,       // Knox
                               1.0, 28.700,  815.0, 100.483, 53.0,       // Maverick
                               1.0, 32.450, 2380.0, 100.533, NO_DOUBLE,  // Nolan (46.0)
                               1.0, 31.800, 3918.0, 106.400, 44.0,       // El Paso
                               1.0, 34.850, 2040.0, 100.217, 41.0,       // Collington
                               1.0, 30.867, 3000.0, 102.900, 47.0,       // Pecos
                               1.0, 36.350, 3693.0, 102.083, 36.0,       // Sherman
                               1.0, 30.300,  597.0,  97.700, 52.0,       // Travis
                               1.0, 26.900,  315.0,  99.283, 60.0,       // Zapata
                               1.0, 28.450,  459.0,  99.217, 56.0,       // Lasalle
                               1.0, 25.900,   19.0,  97.433, 62.0)       // Cameron

    println (s"xy = $xy")

    banner ("Texas Temperatures Regression before Imputation")
    var mod = Regression (xy)()                               // create model with intercept (else pass x)
    mod.trainNtest ()()                                       // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    val x = xy.not (?, 4)
    val y = xy(?, 4)

    banner ("ImputeMRegression.imputeAll")
    ImputeMRegression.setZ (x)
    println ("y = " + ImputeMRegression.imputeAll (y))

    banner ("Texas Temperatures Regression after Imputation")
    mod = new Regression (x, y)                               // create model with intercept (else pass x)
    mod.trainNtest ()()                                       // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

end imputationTest3

