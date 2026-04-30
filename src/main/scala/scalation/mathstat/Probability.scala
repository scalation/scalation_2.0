
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Oct 24 11:59:36 EDT 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Probability and Entropy
 */

package scalation
package mathstat

import scala.math.{abs, log}

import MatrixD.outer

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Probability` object provides methods for operating on univariate and
 *  bivariate probability distributions of discrete random variables X and Y.
 *  A probability distribution is specified by its probability mass functions (pmf)
 *  stored either as a "probability vector" for a univariate distribution or
 *  a "probability matrix" for a bivariate distribution.
 *      joint probability matrix:       pxy(i, j)  = P(X = x_i, Y = y_j)
 *      marginal probability vector:    px(i)      = P(X = x_i)
 *      conditional probability matrix: px_y(i, j) = P(X = x_i|Y = y_j)
 *  In addition to computing joint, marginal and conditional probabilities,
 *  methods for computing entropy and mutual information are also provided.
 *  Entropy provides a measure of disorder or randomness.  If there is
 *  little randomness, entropy will close to 0, while when randomness is
 *  high, entropy will be close to, e.g., log2 (px.dim).  Mutual information
 *  provides a robust measure of dependency between random variables
 *  (contrast with correlation).
 */
object Probability:

    private val debug   = debugf ("Probability", false)               // debug function
    private val flaw    = flawf ("Probability")                       // flaw function
    private val EPSILON = 1E-9                                        // a number close to zero

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the vector px is a legitimate "probability vector".
     *  The elements of the vector must be non-negative and add to one.
     *  @param px  the probability vector
     */
    def isProbability (px: VectorD): Boolean = px.min >= 0.0 && abs (px.sum - 1.0) < EPSILON

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the matrix pxy is a legitimate joint "probability matrix".
     *  The elements of the matrix must be non-negative and add to one.
     *  @param pxy  the probability matrix
     */
    def isProbability (pxy: MatrixD): Boolean = pxy.mmin >= 0.0 && abs (pxy.sum - 1.0) < EPSILON

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Joint Frequency Table (JFT) for vector x and vector y.
     *  Count the number of cases where x(i) = v and y(i) = c.
     *  @param x   the variable/feature vector
     *  @param vc  the number of disctinct values in vector x (value count)
     *  @param y   the response/classification vector
     *  @param k   the maximum value of y + 1 (number of classes)
     */
    def freq (x: VectorI, vc: Int, y: VectorI, k: Int): MatrixD =
        val jft = new MatrixD (vc, k)
        for i <- x.indices do jft(x(i), y(i)) += 1
        debug ("freq", s" conditional frequency table: jft = $jft")
        jft
    end freq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the frequency of occurrence in vector x of value vl for each of y's
     *  classification values, e.g., x is column 2 (Humidity), vl is 1 (High)) and
     *  y can be 0 (no) or 1 (yes).  Also, determine the fraction of training cases
     *  where the feature has this value (e.g., fraction where Humidity is High = 7/14).
     *  @param x   the feature/column vector (e.g., column j of matrix)
     *  @param y   the response/classification vector
     *  @param k   the maximum value of y + 1
     *  @param vl  one of the possible branch values for feature x (e.g., 1 (High))
     */
    def freq (x: VectorI, y: VectorI, k: Int, vl: Int): (Double, VectorI) =
        val nu  = new VectorI (k)                                     // frequency counts
        var cnt = 0                                                   // count for branch value
        for i <- x.indices if x(i) == vl do { nu(y(i)) += 1; cnt += 1 }
        debug ("freq", s"k = $k, vl = $vl, nu = $nu")
        (cnt.toDouble / x.dim, nu)                                    // return fraction and frequency vector
    end freq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the frequency of occurrence in vector x of value vl for each of y's
     *  classification values, e.g., x is column 2 (Humidity), vl is 1 (High)) and
     *  y can be 0 (no) or 1 (yes).  Also, determine the fraction of training cases
     *  where the feature has this value (e.g., fraction where Humidity is High = 7/14).
     *  @param x     the feature/column vector (e.g., column j of matrix)
     *  @param y     the response/classification vector
     *  @param k     the maximum value of y + 1
     *  @param vl    one of the possible branch values for feature x (e.g., 1 (High))
     *  @param idx_  the index positions within x (if null, use all index positions)
     */
    def freq (x: VectorI, y: VectorI, k: Int, vl: Int, idx_ : VectorI): (Double, VectorI) =
        val idx = if idx_ == null then VectorI.range (0, y.dim) else idx_
        val nu  = new VectorI (k)                                 // frequency counts
        var cnt = 0                                               // count for branch value
        for i <- idx if x(i) == vl do { nu(y(i)) += 1; cnt += 1 }
        debug ("freq", s"k = $k, vl = $vl, nu = $nu")
        (cnt.toDouble / idx.size, nu)                             // return fraction and frequency vector
    end freq

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the frequency of occurrence in vector x of value vl for each of y's
     *  classification values, e.g., x is column 2 (Humidity), vl is 1 (High)) and
     *  y can be 0 (no) or 1 (yes).  Also, determine the fraction of training cases
     *  where the feature has this value (e.g., fraction where Humidity is High = 7/14).
     *  This method works for vectors with integer or continuous values.
     *  @param x      the feature/column vector (e.g., column j of matrix)
     *  @param y      the response/classification vector
     *  @param k      the maximum value of y + 1
     *  @param vl     one of the possible branch values for feature x (e.g., 1 (High))
     *  @param idx_   the index positions within x (if null, use all index positions)
     *  @param cont   whether feature/variable x is to be treated as continuous
     *  @param thres  the splitting threshold for features/variables treated as continuous
     */
    def freq (x: VectorD, y: VectorI, k: Int, vl: Int, idx_ : VectorI, cont: Boolean,
              thres: Double): (Double, VectorI) =
        val idx = if idx_ == null then VectorI.range (0, y.dim) else idx_
        val nu  = new VectorI (k)                                     // frequency counts
        var cnt = 0                                                   // count for the value branch
        if cont then
            if vl == 0 then
                for i <- idx if x(i) <= thres do { nu(y(i)) += 1; cnt += 1 }
            else
                for i <- idx if x(i) >  thres do { nu(y(i)) += 1; cnt += 1 }
        else
            for i <- idx if x(i) == vl do { nu(y(i)) += 1; cnt += 1 }
        debug ("freq", s"k = $k, vl = $vl, cont = $cont, thres = $thres, nu = $nu")
        (cnt.toDouble / x.dim, nu)                                    // return fraction and frequency vector
    end freq

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the total number of occurrence in vector x of value vl,  e.g.,
     *  x is column 2 (Humidity), vl is 1 (High) matches 7 rows.
     *  This method works for vectors with integer or continuous values.
     *  @param x      the feature/column vector (e.g., column j of matrix)
     *  @param y      the response/classification vector
     *  @param k      the maximum value of y + 1
     *  @param vl     one of the possible branch values for feature x (e.g., 1 (High))
     *  @param cont   whether feature/variable x is to be treated as continuous
     *  @param thres  the splitting threshold for features/variables treated as continuous
     */
    def count (x: VectorD, vl: Int, cont: Boolean, thres: Double): Int =
        var cnt = 0                                                   // count for the value branch
        if cont then
            if vl == 0 then
                for i <- x.indices if x(i) <= thres do cnt += 1
            else
                for i <- x.indices if x(i) >  thres do cnt += 1
        else
            for i <- x.indices if x(i) == vl do cnt += 1
        cnt
    end count

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a frequency vector, convert it to a probability vector.
     *  @param nu  the frequency vector
     */
    def toProbability (nu: VectorI): VectorD = nu.toDouble / nu.sum.toDouble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a frequency vector, convert it to a probability vector.
     *  @param nu  the frequency vector
     *  @param n   the total number of instances/trials collected
     */
    def toProbability (nu: VectorI, n: Int): VectorD = nu.toDouble / n.toDouble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a frequency matrix, convert it to a probability matrix.
     *  @param nu  the frequency matrix
     */
    def toProbability (nu: MatrixD): MatrixD = nu / nu.sum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a frequency matrix, convert it to a probability matrix.
     *  @param nu  the frequency matrix
     *  @param n   the total number of instances/trials collected
     */
    def toProbability (nu: MatrixD, n: Int): MatrixD = nu / n.toDouble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the probability of discrete random variable y taking on any of k values
     *  @param y  the feature/column vector of integer values whose frequency counts are sought
     *  @param k  the maximum value of y + 1, e.g., { 0, 1, 2} => k = 3
     */
    def probY (y: VectorI, k: Int): VectorD = y.freq (k)._2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given two independent random variables X and Y, compute their
     *  "joint probability", which is the outer product of their probability
     *  vectors px and py, i.e., P(X = x_i, Y = y_j).
     */
    def jProbXY (px: VectorD, py: VectorD): MatrixD = outer (px, py)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Joint Probability Table (JPT) for vector x and vector y.
     *  Count the number of cases where x(i) = v and y(i) = c and divide by
     *  the number of instances/datapoints.
     *  @param x   the variable/feature vector
     *  @param vc  the number of disctinct values in vector x (value count)
     *  @param y   the response/classification vector
     *  @param k   the maximum value of y + 1 (number of classes)
     */
    def jProbXY (x: VectorI, vc: Int, y: VectorI, k: Int): MatrixD =
        freq (x, vc, y, k) / x.dim.toDouble
    end jProbXY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a joint probability matrix pxy, compute the "marginal probability"
     *  for random variable X, i.e, P(X = x_i).
     *  @param pxy  the probability matrix
     */
    def margProbX (pxy: MatrixD): VectorD =
        val px = new VectorD (pxy.dim)
        for i <- pxy.indices do px(i) = pxy(i).sum
        px
    end margProbX

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a joint probability matrix pxy, compute the "marginal probability"
     *  for random variable Y, i.e, P(Y = y_j).
     *  @param pxy  the probability matrix
     */
    def margProbY (pxy: MatrixD): VectorD =
        val py = new VectorD (pxy.dim2)
        for j <- pxy.indices2 do py(j) = pxy(?, j).sum
        py
    end margProbY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a joint probability matrix pxy, compute the "conditional probability"
     *  for random variable Y given random variable X, i.e, P(Y = y_j|X = x_i).
     *  @param pxy  the joint probability matrix
     *  @param px_  the marginal probability vector for X
     */
    def condProbY_X (pxy: MatrixD, px_ : VectorD = null): MatrixD =
        val px   = if px_ == null then margProbX (pxy) else px_
        val py_x = new MatrixD (pxy.dim, pxy.dim2)
        for i <- pxy.indices do py_x(i) = pxy(i) / px(i)
        py_x
    end condProbY_X

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a joint probability matrix pxy, compute the "conditional probability"
     *  for random variable X given random variable Y, i.e, P(X = x_i|Y = y_j).
     *  @param pxy  the joint probability matrix
     *  @param py_  the marginal probability vector for Y
     */
    def condProbX_Y (pxy: MatrixD, py_ : VectorD = null): MatrixD =
        val py   = if py_ == null then margProbY (pxy) else py_
        val px_y = new MatrixD (pxy.dim, pxy.dim2)
        for j <- pxy.indices2 do px_y(?, j) = pxy(?, j) / py(j)
        px_y
    end condProbX_Y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a probability p, compute the "positive log-probability".
     *  Requires the probability to be non-zero.
     *  @param p  the given probability
     */
    inline def plog (p: Double): Double = - log2 (p)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a probability vector px, compute the "positive log-probability".
     *  Requires each probability to be non-zero.
     *  @param px  the probability vector
     */
    def plog (px: VectorD): VectorD = px.map (plog (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a probability vector px, compute the "entropy" of random
     *  variable X.
     *  @see http://en.wikipedia.org/wiki/Entropy_%28information_theory%29
     *  @param px  the probability vector
     */
    def entropy (px: VectorD): Double =
        var sum = 0.0
        for p <- px if p > 0.0 do sum -= p * log2 (p)
        sum
    end entropy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a frequency vector nu, compute the "entropy" of random variable X.
     *  @see http://en.wikipedia.org/wiki/Entropy_%28information_theory%29
     *  @param nu  the frequency vector
     */
    def entropy (nu: VectorI): Double =
        val tot = nu.sum.toDouble
        var sum = 0.0
        for c <- nu if c > 0 do { val p = c / tot; sum -= p * log2 (p) }
        sum
    end entropy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a probability vector px, compute the " base-k entropy" of random
     *  variable X.
     *  @see http://en.wikipedia.org/wiki/Entropy_%28information_theory%29
     *  @param px  the probability vector
     *  @param b   the base for the logarithm
     */
    def entropy (px: VectorD, b: Int): Double =
        var sum = 0.0
        for p <- px if p > 0.0 do sum -= p * logb (b, p)
        sum
    end entropy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a probability vector px, compute the "normalized entropy" of
     *  random variable X.
     *  @see http://en.wikipedia.org/wiki/Entropy_%28information_theory%29
     *  @param px  the probability vector
     */
    def nentropy (px: VectorD): Double =
        val k = px.dim                                  // let the base k = # elements in probability vector
        var sum = 0.0
        for p <- px if p > 0.0 do sum -= p * logb (k, p)
        sum
    end nentropy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given probability vectors px and qx, compute the "relative entropy".
     *  @param px  the first probability vector
     *  @param qx  the second probability vector (requires qx.dim >= px.dim)
     */
    def rentropy (px: VectorD, qx: VectorD): Double =
        var sum = 0.0
        for i <- px.indices if px(i) > 0.0 do sum += px(i) * log2 (px(i)/qx(i))
        sum
    end rentropy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given probability vectors px and qx, compute the "cross entropy".
     *  May also pass in response vectors: y (actual) and yp (predicted).
     *  @param px      the first probability vector
     *  @param qx      the second probability vector (requires qx.dim >= px.dim)
     *  @param base_e  whether to use base e or base 2 logarithms (defaults to e)
     */
    def centropy (px: VectorD, qx: VectorD, base_e: Boolean = true): Double =
        var sum = 0.0
        if base_e then for i <- px.indices if px(i) > 0.0 do sum -= px(i) * log (qx(i))
        else for i <- px.indices if px(i) > 0.0 do sum -= px(i) * log2 (qx(i))
        sum
    end centropy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a joint probability matrix pxy, compute the "joint entropy"
     *  of random variables X and Y.
     *  @param pxy  the joint probability matrix
     */
    def entropy (pxy: MatrixD): Double =
        var sum = 0.0
        for i <- pxy.indices; j <- pxy.indices2 do
            val p = pxy(i, j)
            if p > 0.0 then sum -= p * log2 (p)
        end for
        sum
    end entropy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a joint probability matrix pxy and a conditional probability
     *  matrix py_x, compute the "conditional entropy" of random variable X
     *  given random variable Y.
     *  @param pxy   the joint probability matrix
     *  @param px_y  the conditional probability matrix
     */
    def entropy (pxy: MatrixD, px_y: MatrixD): Double =
        if pxy.dim != px_y.dim || pxy.dim2 != px_y.dim2 then
            flaw ("entropy", "joint and conditional probability matrices are not compatible")

        var sum = 0.0
        for i <- pxy.indices; j <- pxy.indices2 do
            val p = pxy(i, j)
            if p > 0.0 then sum -= p * log2 (px_y(i, j))
        end for
        sum
    end entropy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a joint probability matrix pxy, compute the mutual information
     *  for random variables X and Y.
     *  @param pxy  the probability matrix
     *  @param px   the marginal probability vector for X
     *  @param py   the marginal probability vector for Y
     */
    def muInfo (pxy: MatrixD, px: VectorD, py: VectorD): Double =
        var sum = 0.0
        for i <- pxy.indices; j <- pxy.indices2 do
            val p = pxy(i, j)
            if p > 0.0 then sum += p * log2 (p / (px(i) * py(j)))
        end for
        sum
    end muInfo

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a joint probability matrix pxy, compute the mutual information
     *  for random variables X and Y.
     *  @param pxy  the probability matrix
     */
    def muInfo (pxy: MatrixD): Double = muInfo (pxy, margProbX (pxy), margProbY (pxy))

end Probability

import Probability._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `probabilityTest` main function is used to test the `Probability` object.
 *  > runMain scalation.mathstat.probabilityTest
 */
@main def probabilityTest (): Unit =

    // coin experiment: probability for 0, 1, 2 heads, when flipping 2 coins
    val px = VectorD (.25, .5, .25)

    // dice experiment: probability for 2, 3, ... 11, 12, when rolling 2 dice
    val py = VectorD (1/36.0, 2/36.0, 3/36.0, 4/36.0, 5/36.0, 6/36.0,
                      5/36.0, 4/36.0, 3/36.0, 2/36.0, 1/36.0)

    // joint probability for coin and dice experiments
    val pxy  = jProbXY (px, py)

    println (s"isProbability ($px)   = ${isProbability (px)}")
    println (s"isProbability ($py)   = ${isProbability (py)}")
    println (s"joint probability pxy = $pxy")
    println (s"isProbability (pxy)   = ${isProbability (pxy)}")

    val x = VectorD.range (0, 3)
    new Plot (x, px, null, "Plot px vs. x")   // plot the pmf for random variable X
    val y = VectorD.range (2, 13)
    new Plot (y, py, null, "Plot py vs. y")   // plot the pmf for random variable Y

    val mpx  = margProbX (pxy)                // marginal probability should be the same as px
    val mpy  = margProbY (pxy)                // marginal probability should be the same as py
    val px_y = condProbX_Y (pxy)              // conditional probability P(X = x_i|Y = y_j)
    val py_x = condProbY_X (pxy)              // conditional probability P(Y = y_j|X = x_i)

    println (s"marginal probability mpx     = $mpx")
    println (s"marginal probability mpy     = $mpy")
    println (s"conditional probability px_y = $px_y")
    println (s"conditional probability py_y = $py_x")

    val hx   = entropy (px)                   // entropy of random variable X
    val hy   = entropy (py)                   // entropy of random variable Y
    val hkx  = nentropy (px)                  // nentropy of random variable X
    val hky  = nentropy (py)                  // nentropy of random variable Y
    val hxy  = entropy (pxy)                  // joint entropy of random variables X and Y
    val hx_y = entropy (pxy, px_y)            // conditional entropy of random variables X given Y
    val hy_x = entropy (pxy.transpose, py_x)  // conditional entropy of random variables Y given X
    val ixy  = muInfo (pxy)                   // mutual information of random variables X given Y

    println (s"entropy hx               = $hx")
    println (s"entropy hy               = $hy")
    println (s"nentropy hkx             = $hkx")
    println (s"nentropy hky             = $hky")
    println (s"joint entropy hxy        = $hxy")
    println (s"conditional entropy hx_y = $hx_y")
    println (s"conditional entropy hy_x = $hy_x")
    println (s"mutual information ixy   = $ixy")

end probabilityTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `probabilityTest2` main function provides upper bounds for entropy and nentropy
 *  and plots entropy.
 *  > runMain scalation.mathstat.probabilityTest2
 */
@main def probabilityTest2 (): Unit =

    banner ("maximum entropy for k-dimensional probability vector")
    for k <- 2 to 32 do
        println (s"max entropy for k  = $k  \tis ${log2 (k)}")
    end for

    banner ("maximum normalized entropy for k-dimensional probability vector")
    for k <- 2 to 32 do
        println (s"max nentropy for k = $k  \tis ${logb (k, k)}")
    end for

    banner ("Test: ProbabilityTest3: Plot entropy")
    val _1 = VectorD.one (99)
    val p  = VectorD.range (1, 100) / 100.0
    val h  = p.map (p => -p * log2 (p) - (1-p) * log2 (1-p))
    new Plot (p, h, _1, "Plot of entropy h vs. p")

    val h2 = p.map (q => { val px = VectorD (q, 1-q); entropy (px) })
    new Plot (p, h2, _1, "Plot of entropy h2 vs. p")

end probabilityTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `probabilityTest3` main function is used to test the `Probability` class.
 *  Plot probability and log-probability entropy and normalized entropy for
 *  binomial distributions.
 *  > runMain scalation.mathstat.probabilityTest3
 */
@main def probabilityTest3 (): Unit =

    import scalation.random.Binomial

    val p = 0.6
    for n <- 2 to 16 do
        val k  = VectorD.range (0, n+1)
        val y  = Binomial (p, n)
        val pp = VectorD (y.pmf (0).toIndexedSeq)                   // probability vector
        val lp = plog (pp)                                          // log-probability vector
        new Plot (k, pp, null, s"Plot for n = $n of pp vs. k", lines = true)
        new Plot (k, lp, null, s"Plot for n = $n of lp vs. k", lines = true)
    end for

    for n <- 1 to 16 do
        val p  = VectorD.range (0, 100) / 100.0
        val h  = new VectorD (p.dim)                                // vector of entropies
        val hk = new VectorD (p.dim)                                // vector of normalized entropies

        for i <- p.indices do
            val y  = Binomial (p(i), n)
            val pp = VectorD (y.pmf (0).toIndexedSeq)
            h(i)   = entropy (pp)
            hk(i)  = nentropy (pp)
        end for

        new Plot (p, h, hk, s"Plot for n = $n of entropy h, hk vs. p", lines = true)
    end for

end probabilityTest3 


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `probabilityTest4` main function is used to test the `Probability` class.
 *  It computes entropy, relative entropy and cross entropy and verifies that
 *  cross entropy == entropy + relative entropy
 *  @see arxiv.org/pdf/0808.4111.pdf
 *  > runMain scalation.mathstat.probabilityTest4
 */
@main def probabilityTest4 (): Unit =

    val px = Array (VectorD (0.1, 0.9),
                    VectorD (0.2, 0.8),
                    VectorD (0.3, 0.7),
                    VectorD (0.4, 0.6),
                    VectorD (0.5, 0.5),
                    VectorD (0.6, 0.4),
                    VectorD (0.7, 0.3),
                    VectorD (0.8, 0.2),
                    VectorD (0.9, 0.1))

    banner ("entropy")
    for i <- px.indices do
        println (s"entropy (${px(i)}) = ${entropy (px(i))}")
    end for

    banner ("relative entropy (KL divergence)")
    for i <- px.indices do
        for j <- px.indices do
             val re = rentropy (px(i), px(j))
             println (s"rentropy (${px(i)}, ${px(j)}) = $re")
        end for
    end for

    banner ("cross entropy")
    for i <- px.indices do
        for j <- px.indices do
             val ce = centropy (px(i), px(j))
             println (s"centropy (${px(i)}, ${px(j)}) = $ce")
             assert (ce =~ entropy (px(i)) + rentropy (px(i), px(j)) )
        end for
    end for

end probabilityTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `probabilityTest5` main function is used to test the `Probability` class.
 *  It computes joint, marginal and conditional probabilities as well as
 *  measures of independence.
 *  @see ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and
    -statistics-spring-2014/readings/MIT18_05S14_Reading7a.pdf
 *  > runMain scalation.mathstat.probabilityTest5
 */
@main def probabilityTest5 (): Unit =

   // X  - dice 1: 1, 2, 3, 4, 5, 6
   // X2 - dice 2: 1, 2, 3, 4, 5, 6
   // Y  = X + X2: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 
   val nuxy = MatrixD ((6, 11), 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                                0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                                0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                                0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)

   println (s"nuxy = $nuxy")
   println (s"1/6  = ${1/6.0}")
   println (s"1/36 = ${1/36.0}")

   banner ("Joint Probability Distribution")
   val pxy = toProbability (nuxy)
   println (s"Joint XY: pxy = $pxy")

   banner ("Marginal Probability Distributions")
   val px = margProbX (pxy)
   val py = margProbY (pxy)
   println (s"Marginal X: px = $px")
   println (s"Marginal Y: py = $py")

   banner ("Conditional Probability Distributions")
   val py_x = condProbY_X (pxy, px)
   val px_y = condProbX_Y (pxy, py)
   println (s"Y given X: py_x = $py_x")
   println (s"X given Y: px_y = $px_y")

   banner ("Independence: zero relative entropy, mutual information")
   val re = rentropy (px, py)
   val mi = muInfo (pxy)
   println (s"rentropy = $re")
   println (s"muInfo   = $mi")

   banner ("Mutual Information = Information Gain")
   val hx   = entropy (px)
   val hy   = entropy (py)
   val hy_x = entropy (pxy, py_x)
   val hx_y = entropy (pxy, px_y)
   println (s"Entropy: hx    = $hx")
   println (s"Entropy: hy    = $hy")
   println (s"CondEnt: hy_x  = $hy_x")
   println (s"CondEnt: hx_y  = $hx_y")
   println (s"mi = hx - hx_y = %{hx - hx_y}")
   println (s"mi = hy - hy_x = (hy - hy_x}")
   assert (mi =~ hx - hx_y)
   assert (mi =~ hy - hy_x)

end probabilityTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `probabilityTest6` main function is used to test the `Probability` class.
 *  It computes joint, marginal and conditional probabilities.
 *  > runMain scalation.mathstat.probabilityTest6
 */
@main def probabilityTest6 (): Unit =

   val mat = MatrixD ((3, 3), 1, 2, 3,
                              4, 5, 6,
                              7, 8, 9)

   banner ("Joint Probability Distribution")
   println (s"mat     = $mat")
   println (s"mat.sum = ${mat.sum}")
   val pxy = toProbability (mat)
   println (s"Joint XY: pxy = $pxy")

   banner ("Marginal Probability Distributions")
   val px = margProbX (pxy)
   val py = margProbY (pxy)
   println (s"Marginal X: px = $px")
   println (s"Marginal Y: py = $py")
   println (s"45*Marg  X: px = ${(px * 45).toInt}")
   println (s"45*Marg  Y: py = ${(py * 45).toInt}")

   banner ("Conditional Probability Distributions")
   val py_x = condProbY_X (pxy, px)
   val px_y = condProbX_Y (pxy, py)
   println (s"Y given X: py_x = $py_x")
   println (s"X given Y: px_y = $px_y")
   println (s"45*Y | X: py_x  = ${(py_x * 45)}")
   println (s"45*X | Y: px_y  = ${(px_y * 45)}")

end probabilityTest6

