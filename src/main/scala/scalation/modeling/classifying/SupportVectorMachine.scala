
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Naman Fatehpuria
 *  @version 2.0
 *  @date    Mon Mar  3 14:39:17 EST 2014
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Support Vector Machine
 *
 *  @see fbim.fh-regensburg.de/~saj39122/Diplomarbeiten/Miklos/Papers/Keerthi%20improvement%20on%20SMO.pdf
 *  @see www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
 *  @see repository.ias.ac.in/127671/1/smo_mod.pdf
 */

//  U N D E R   D E V E L O P M E N T

package scalation
package modeling
package classifying

import scala.collection.mutable.Set
import scala.math.abs

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SupportVectorMachine` class is a translation of Pseudo-Code from a
 *  modified SMO (Modification 2) found at the above URL's into Scala and includes
 *  a few simplifications (e.g., currently only works for linear kernels, dense
 *  data and binary classification).
 *  @param x       the input/data matrix with points stored as rows
 *  @param y       the classification of the data points stored in a vector
 *  @param fname_  the feature/variable names
 *  @param cname_  the class names
 *  @param hparam  the hyper-parameters
 */
class SupportVectorMachine (x: MatrixD, y: VectorI, fname_ : Array [String] = null,
                            cname_ : Array [String] = Array ("-", "+"),
                            hparam: HyperParameter = Classifier.hp)
      extends Classifier (x, y, fname_, 2, cname_, hparam)
         with FitC ():

    type Pair = (Double, Double)
    
    private val debug   = debugf ("SupportVectorMachine", false) // debug flag
    private val EPSILON = 1E-3                                   // a number close to zero
    private val TOL     = 1E-3                                   // tolerance level
    private val C       = 0.05                                   // crossing penalty

    private var alp: VectorD    = null                           // alpha (Lagrange multipliers)
    private var fCache: VectorD = null                           // errors for all non-bound examples
    private var w: VectorD      = null                           // weights
    
    private val I_0     = Set [Int] ()                           // {i: 0 < alp(i) < C}
    private val I_1     = Set [Int] ()                           // {i: y(i) =  1, alp(i) = 0}
    private val I_2     = Set [Int] ()                           // {i: y(i) = -1, alp(i) = C}
    private val I_3     = Set [Int] ()                           // {i: y(i) =  1, alp(i) = C}
    private val I_4     = Set [Int] ()                           // {i: y(i) = -1, alp(i) = 0}
    
    private var b       =  0.0                                   // threshold/offset
    private var b_Low   =  1.0                                   // lower threshold
    private var b_Up    = -1.0                                   // upper threshold
    private var i_Low   = -1                                     // index for b_Low
    private var i_Up    = -1                                     // index for b_Up
    
    private var al1     = 0.0                                    // old Lagrange multiplier 1
    private var a1      = 0.0                                    // new Lagrange multiplier 1
    private var al2     = 0.0                                    // old Lagrange multiplier 2
    private var a2      = 0.0                                    // new Lagrange multiplier 2

    _modelName = "SupportVectorMachine"                          // the name of the model

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of model parameter values [ w | b ]
     */
    override def parameter: VectorD = w :+ b
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a classification model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output vector.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training set.
     *  Train uses SMO (Sequential Minimum Optimization) algorithm to solve the 
     *  optimization problem for the weight vector w and the threshold b for 
     *  the model (w dot z) - b.
     *  Fill Set I_1 and I_4, as initially alp[i] = 0
     *  Initialize i_Up to any index of class +1 
     *  Initialize i_Low to any index of class -1
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
//      super.train (x_, y_)   
        alp     = new VectorD (x_.dim)                            // alpha (Lagrange multipliers)
        fCache  = new VectorD (x_.dim)                            // errors for all non-bound examples
        w       = new VectorD (x_.dim2)                           // weights
        for i <- y_.indices do
            if y_(i) == 1 then
                I_1 += i; i_Up = i
            else
                I_4 += i; i_Low = i
        end for
        
        fCache(i_Low) =  1
        fCache(i_Up)  = -1
      
        var nChanged = 0
        var checkAll = true
        
        while nChanged > 0 || checkAll do
            nChanged = 0
            if checkAll then for k <- x_.indices if checkExample (k) do nChanged += 1
            else
                var success = true
                nChanged = 0
                while (b_Up < b_Low - 2 * TOL) && success do
                    success = takeStep (i_Up, i_Low)
                    if success then nChanged += 1
                end while
            end if  
            if checkAll then checkAll = false
            else if nChanged == 0 then checkAll = true
        end while
        
        b = (b_Low + b_Up) / 2.0
        debug ("train", s"w = $w, b = $b")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the predictive model y_ = f(x_) + e and return its predictions and QoF vector.
     *  Testing may be in-sample (on the full dataset) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorI = y): (VectorI, VectorD) =
        val yp  = predictI (x_)                                          // predicted classes
        val qof = diagnose (y_, yp)                                      // diagnose from actual and predicted
        debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures,
     *  from the error/residual vector and the predicted & actual responses.
     *  Requires the actual and predicted responses to be non-negative integers.
     *  This override maps -1 to 0.
     *  @param y_  the actual response/output vector to use (test/full)
     *  @param yp  the predicted response/output vector (test/full)
     */
    override def diagnose (y_ : VectorI, yp: VectorI): VectorD =
        super.diagnose (y_.map2 (-1, 0), yp.map2 (-1, 0))                // compute basic QoF from `FitM`
    end diagnose

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new continuous data vector z, determine which class it belongs to.
     *  Classify returns 1 meaning z belongs to the positive (+) class, while 
     *  -1 means it belongs to the negative (-) class.
     *  @param z  the vector to classify
     */
    override def predictI (z: VectorD): Int = if (w dot z) >= b then 1 else -1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor x_0, x_1,
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = null, fname_ : Array [String] = null,
                          b_ : VectorD = p_y, vifs: VectorD = null): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the value using the learned function (assumes linear, dense).
     *  @param k  the row index into the data matrix
     */
    private def func (k: Int): Double = w dot x(k)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the value using the kernel function (assumes linear, dense).
     *  @param i1  the first row index into the data matrix
     *  @param i2  the second row index into the data matrix
     */
    private def kernel_func (i1: Int, i2: Int): Double = x(i1) dot x(i2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Optimize by replacing old values of Lagrange multipliers al1, al2 with
     *  new values a1 and a2.
     *  @param i1  the index for the first Lagrange multipliers (alpha) 
     *  @param i2  the index for the second Lagrange multipliers (alpha)
     */
    private def takeStep (i1: Int, i2: Int): Boolean =
        debug ("takeStep", s"(i1, i2) = ($i1, $i2)")
        if i1 == i2 then { debug ("takeStep", s"skip if i1 == i2"); return false }
        
        al1 = alp(i1); al2 = alp(i2)
        val y1 = y(i1); val y2 = y(i2)
        val F1 = fCache(i1); val F2 = fCache(i2) 
        val s  = y1 * y2
        
        val (l, h) = computeLH (y1, y2)
        if l == h then { debug ("takeStep", s"skip if l == h"); return false }
        
        // compute eta
        val k11 = kernel_func (i1, i1)
        val k12 = kernel_func (i1, i2)
        val k22 = kernel_func (i2, i2)
        val eta = 2.0 * k12 - (k11 + k22)

        if eta < 0.0 then
            a2 = al2 - y2 * (F1 - F2) / eta
            if a2 < l then a2 = l else if a2 > h then a2 = h
        else
            val c1 = eta / 2.0
            val c2 = y2 * (F1 - F2) - eta * al2
            val lObj = c1 * l * l + c2 * l
            val hObj = c1 * h * h + c2 * h
            a2 = if lObj > hObj + EPSILON then l else if lObj < hObj - EPSILON then h else al2
        end if
        
        if abs (a2 - al2) < EPSILON * (a2 + al2 + EPSILON) then
            debug ("takeStep", s"skip if a2 = $a2 ~=  al2 = $al2")      // almost no change
            return false
        
        a1 = al1 + s * (al2 - a2)
        if a1 < 0.0 then
            a2 += s * a1; a1 = 0
        else if a1 > C then
            val t = a1 - C; a2 += s * t; a1 = C
        
        update (i1, i2, y1, y2)                // weights and fCache
        alp(i1) = a1; alp(i2) = a2             // store a1, a2 in alp array
        
        // Compute update F values for i1 and i2.
        fCache(i1) = F1 + y1 * (a1 - al1) * k11 + y2 * (a2 - al2) * k12
        fCache(i2) = F2 + y1 * (a1 - al1) * k12 + y2 * (a2 - al2) * k22
        
        // Update I_0, I_1, I_2, I_3, I_4 for i1
        if a1 > 0 && a1 < C then       I_0 += i1 else I_0 -= i1
        if y(i1) == 1 && a1 == 0 then  I_1 += i1 else I_1 -= i1
        if y(i1) == -1 && a1 == C then I_2 += i1 else I_2 -= i1
        if y(i1) == 1 && a1 == C then  I_3 += i1 else I_3 -= i1
        if y(i1) == -1 && a1 == 0 then I_4 += i1 else I_4 -= i1
        
        // Update I_0, I_1, I_2, I_3, I_4 for i2
        if a2 > 0 && a2 < C then       I_0 += i2 else I_0 -= i2
        if y(i2) == 1 && a2 == 0 then  I_1 += i2 else I_1 -= i2
        if y(i2) == -1 && a2 == C then I_2 += i2 else I_2 -= i2
        if y(i2) == 1 && a2 == C then  I_3 += i2 else I_3 -= i2
        if y(i2) == -1 && a2 == 0 then I_4 += i2 else I_4 -= i2
        
        // Compute (i_Low, b_Low) and (i_Up, b_Up), using i1, i2 and indices in I_0
        val I = I_0
        I ++= Set (i1, i2)
        for i <- I do
            if fCache (i) < b_Up then
                b_Up = fCache (i)
                i_Up = i
            if fCache (i) > b_Low then
                b_Low = fCache (i)
                i_Low = i
        end for
        true
    end takeStep
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute l and h.
     *  @param y1  the first target value
     *  @param y2  the second target value
     */
    private def computeLH (y1: Int, y2: Int): Pair =
        if y1 == y2 then
            val gamma = al1 + al2
            return if gamma > C then (gamma - C, C) else (0.0, gamma)
        else
            val gamma = al1 - al2
            return if gamma > 0.0 then (0.0, C - gamma) else (-gamma, C)
    end computeLH
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update weights w and error cache fCache.
     *  @param i1  the index for the first Lagrange multipliers (alpha) 
     *  @param i2  the index for the second Lagrange multipliers (alpha)
     *  @param y1  the first target value
     *  @param y2  the second target value
     */
    private def update (i1: Int, i2: Int, y1: Int, y2: Int): Unit =
        val t1 = y1 * (a1 - al1)
        val t2 = y2 * (a2 - al2)
        
        // Update weight vector to reflect change in a1 and a2
        for j <- x.indices2 do w(j) += x(i1, j) * t1 + x(i2, j) * t2

        // Update fCache(i) for i in I_0 using new Lagrange multipliers (a1 & a2)
        for i <- I_0 if i != i1 && i != i2 do
            fCache(i) += t1 * kernel_func (i1, i) + t2 * kernel_func (i2, i)
//          fCache(i) += (y1 * a1 * kernel_func (i1, i)) - y(i) + (y2 * a2 * kernel_func (i2, i)) - y(i)
        end for
        debug ("update", s"w = $w")
    end update
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Using the index i2 from training data, update (b_Low, i_Low) or (b_Up, i_Up)
     *  Then check for optimality using current b_Low and b_Up and, if
     *  violated, find an index i1 to do joint optimization with i2.
     *  @param i2  the second data point for optimization
     */
    private def checkExample (i2: Int): Boolean =
        var i1 = -1
        var F2 = 0.0
        al2 = alp(i2)
        
        if I_0 contains i2 then
            F2 = fCache (i2)
        else
            F2 = func (i2) - y (i2)
            fCache (i2) = F2

            // Update (b_Low, i_Low) or (b_Up, i_Up) using (F2, i2)
            if ((I_1 contains i2) || (I_2 contains i2)) && (F2 < b_Up) then
                b_Up = F2
                i_Up = i2
            else if ((I_3 contains i2) || (I_4 contains i2)) && (F2 > b_Low) then
                b_Low = F2
                i_Low = i2
        end if
        
        // Check optimality using current b_Low and b_Up and, if
        // violated, find an index i1 to do joint optimization with i2.
        var optimality = true
        
        if (I_0 contains i2) || (I_1 contains i2) || (I_2 contains i2) then
            if b_Low - F2 > 2 * TOL then
                optimality = false
                i1 = i_Low
        end if

        if (I_0 contains i2) || (I_3 contains i2) || (I_4 contains i2) then
            if F2 - b_Up > 2 * TOL then
                optimality = false
                i1 = i_Up
        end if
        
        if optimality then return false
        
        // For i2 in I_0 choose the better i1.
        if I_0 contains i2 then
            if b_Low - F2 > F2 - b_Up then i1 = i_Low
            else i1 = i_Up
        
        takeStep (i1, i2)
    end checkExample
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert svm to a string showing (w, b).
     */
    override def toString: String = "(w, b) = " + (w, b)

end SupportVectorMachine


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SupportVectorMachine` companion object provides a factory method.
 */
object SupportVectorMachine:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a support vector machine for the given combined matrix where the
     *  column col is the response/classification vector.
     *  @param xy      the combined data matrix (features and response)
     *  @param fname   the names for all features/variables
     *  @param cname   the names for all classes
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               cname: Array [String]  = Array ("No", "Yes"),
               hparam: HyperParameter = Classifier.hp)
              (col: Int = xy.dim2 - 1): SupportVectorMachine =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new SupportVectorMachine (x, y.map2 (0, -1), fname, cname, hparam)
    end apply

end SupportVectorMachine


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SupportVectorMachineTest` main function tests the `SupportVectorMachine` class.
 *  > runMain scalation.modeling.classifying.supportVectorMachineTest
 */
@main def supportVectorMachineTest (): Unit =

    // Test 1
    val x = MatrixD ((4, 2), 1.0, 2.0,                       // 4 data points
                             2.0, 1.0,
                             2.0, 3.0,
                             3.0, 2.0)
    val y = VectorI (-1, -1, 1, 1)                           // classification of points
    val z = VectorD (4.0, 3.0)
    
    val mod = new SupportVectorMachine (x, y)                // create optimizer
    mod.trainNtest ()()
    println (mod.summary ())
    println (mod.predict (z))

end supportVectorMachineTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `supportVectorMachineTest2` main function tests the `SupportVectorMachine` class.
 *  > runMain scalation.modeling.classifying.supportVectorMachineTest2
 */
@main def supportVectorMachineTest2 (): Unit =

    val xy = MatrixD ((8, 3), 2, 2, -1,                      // 8 data points
                              4, 2, -1,
                              2, 4, -1,
                              2, 6,  1,
                              4, 4,  1,
                              6, 2,  1,
                              6, 4,  1,
                              4, 6,  1)

    val (x, y) = (xy.not(?, 2), xy(?, 2).toInt)

    val mod = new SupportVectorMachine (x, y)                // create optimizer
    mod.trainNtest ()()
    println (mod.summary ())

end supportVectorMachineTest2

