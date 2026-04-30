
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Feb 10 23:33:57 EST 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Test-n-Train (TnT) Split for Datasets
 */

package scalation
package mathstat

import scala.collection.immutable.Set
import scala.collection.mutable.IndexedSeq

import scalation.random.PermutedVecI

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TnT_Split` object provides methods for splitting datasets into TESTING-sets (.)
 *  and TRAINING-sets (-).  There are three options on how to split the full dataset:
 *  (1) select at RANDOM `n_test` indices for the test-set (@see tnT_SplitTest`)
 *                 | -.--.---.--.----.--.---.-----.|
 *  (2) select the FIRST `n_test` indices for the test-set (@see tnT_SplitTest2`)
 *                 | test-set |    training-set    |
 *  (3) select the LAST `n_test` indices for the test-set (@see tnT_SplitTest3`)
 *                 |    training-set    | test-set |
 */
object TnT_Split:

    private val flaw = flawf ("TnT_Split")                                   // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a permutation generator for integers from 0 until limit.
     *  @param limit   the upper limit of integers (exclusive) 
     *  @param stream  the random number stream to use
     */
    def makePermGen (limit: Int, stream: Int = 0): PermutedVecI =
        PermutedVecI (VectorI.range (0, limit), stream)
    end makePermGen

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the indices for the test-set.  If rando = false, pick the FIRST indices.
     *  @oaram permGen  the permutation generator (may be null when rando = false)
     *  @param n_test   the size of test-set
     *  @param rando    whether to select indices randomly or in blocks (defaults to true)
     */
    def testIndices (permGen: PermutedVecI, n_test: Int, rando: Boolean = true): IndexedSeq [Int] =
        (if rando then permGen.igen (0 until n_test)                         // permuted indices
         else VectorI.range (0, n_test)).toMuIndexedSeq                      // ordered indices
    end testIndices

    def testIndices2 (permGen: PermutedVecI, n_test: Int, rando: Boolean = true): Set [Int] =
        if rando then permGen.igen (0 until n_test).toSet [Int]              // permuted indices
        else Set.range (0, n_test)                                           // ordered indices
    end testIndices2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the indices for the test-set.  If rando = false, pick the LAST indices.
     *  @oaram permGen  the permutation generator (may be null when rando = false)
     *  @param n_total  the size of full dataset (train plus test)
     *  @param n_test   the size of test-set
     *  @param rando    whether to select indices randomly or in blocks (defaults to true)
     */
    def testIndices (permGen: PermutedVecI, n_total: Int, n_test: Int, rando: Boolean): IndexedSeq [Int] =
        (if rando then permGen.igen (0 until n_test)                         // permuted indices
         else VectorI.range (n_total - n_test, n_total)).toMuIndexedSeq      // ordered indices
    end testIndices

    def testIndices2 (permGen: PermutedVecI, n_total: Int, n_test: Int, rando: Boolean): Set [Int] =
        if rando then permGen.igen (0 until n_test).toSet [Int]              // permuted indices
        else Set.range (n_total - n_test, n_total)                           // ordered indices
    end testIndices2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the dataset given as a combined data-response matrix into a testing-set
     *  and training-set based on the given indices.
     *  @param xy   the combined data-response matrix
     *  @param idx  the indices for the testing-set
     */
    def apply (xy: MatrixD, idx: IndexedSeq [Int]): (MatrixD, MatrixD) =
        val (xy_test, xy_train) = xy.split (idx)
        (xy_test, xy_train)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the dataset given as a data matrix and a response vector into a testing-set
     *  and training-set based on the given indices.
     *  @see `scalation.modeling.Predictor`
     *  @param x    the input/data matrix (for some models this may be null => return (null, null)
     *  @param y    the output/response vector
     *  @param idx  the set of indices for the testing-set
     */
    def apply (x: MatrixD, y: VectorD, idx: Set [Int]): (MatrixD, MatrixD, VectorD, VectorD) =
        if x.dim != y.dim then flaw ("apply", s"x.dim ${x.dim} != y.dim = ${y.dim}")

        val (x_test, x_train) = if x == null then (null, null) else x.split (idx)
        val (y_test, y_train) = y.split (idx)
        (x_test, x_train, y_test, y_train)
    end apply

    def apply (x: MatrixD, y: VectorD, idx: IndexedSeq [Int]): (MatrixD, MatrixD, VectorD, VectorD) =
        apply (x, y, idx.toSet [Int])
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the dataset given as a data matrix and an integer-valued response vector
     *  into a testing-set and training-set based on the given indices.
     *  @see `scalation.modelingq.classifying.Classifier`
     *  @param x    the input/data matrix (for some models this may be null => return (null, null)
     *  @param y    the integer-valued output/response vector
     *  @param idx  the indices for the testing-set
     */
    def apply (x: MatrixD, y: VectorI, idx: IndexedSeq [Int]): (MatrixD, MatrixD, VectorI, VectorI) =
        if x.dim != y.dim then flaw ("apply", s"x.dim ${x.dim} != y.dim = ${y.dim}")

        val (x_test, x_train) = if x == null then (null, null) else x.split (idx)
        val (y_test, y_train) = y.split (idx)
        (x_test, x_train, y_test, y_train)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the dataset given as a data matrix and a response matrix into a testing-set
     *  and training-set based on the given indices.
     *  @see `scalation.modeling.neuralnet.PredictorMV`
     *  @param x    the input/data matrix
     *  @param y    the output/response matrix
     *  @param idx  the indices for the testing-set
     */
    def apply (x: MatrixD, y: MatrixD, idx: IndexedSeq [Int]): (MatrixD, MatrixD, MatrixD, MatrixD) =
        if x.dim != y.dim then flaw ("apply", s"x.dim ${x.dim} != y.dim = ${y.dim}")

        val (x_test, x_train) = if x == null then (null, null) else x.split (idx)
        val (y_test, y_train) = y.split (idx)
        (x_test, x_train, y_test, y_train)
    end apply

end TnT_Split

import TnT_Split.{makePermGen, testIndices}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tnT_SplitTest` main function tests the `TnT_Split` object using the Texas
 *  Temperatures dataset.  It is split into a testing-set and a training-set.
 *  Tests when rando = true and RANDOMLY picks the indices for the testing-set.
 *  > runMain scalation.mathstat.tnT_SplitTest
 */
@main def tnT_SplitTest (): Unit =

    // Combined data-response matrix
    // 16 data points:         one      x1      x2       x3     y
    //                                 Lat    Elev     Long  Temp          County
    val xy = MatrixD ((16, 5), 1.0, 29.767,   41.0,  95.367, 56.0,    // 0.  Harris
                               1.0, 32.850,  440.0,  96.850, 48.0,    // 1.  Dallas
                               1.0, 26.933,   25.0,  97.800, 60.0,    // 2.  Kennedy
                               1.0, 31.950, 2851.0, 102.183, 46.0,    // 3.  Midland
                               1.0, 34.800, 3840.0, 102.467, 38.0,    // 4.  Deaf Smith
                               1.0, 33.450, 1461.0,  99.633, 46.0,    // 5.  Knox
                               1.0, 28.700,  815.0, 100.483, 53.0,    // 6.  Maverick
                               1.0, 32.450, 2380.0, 100.533, 46.0,    // 7.  Nolan
                               1.0, 31.800, 3918.0, 106.400, 44.0,    // 8.  El Paso
                               1.0, 34.850, 2040.0, 100.217, 41.0,    // 9.  Collington
                               1.0, 30.867, 3000.0, 102.900, 47.0,    // 10. Pecos
                               1.0, 36.350, 3693.0, 102.083, 36.0,    // 11. Sherman
                               1.0, 30.300,  597.0,  97.700, 52.0,    // 12. Travis
                               1.0, 26.900,  315.0,  99.283, 60.0,    // 13. Zapata
                               1.0, 28.450,  459.0,  99.217, 56.0,    // 14. Lasalle
                               1.0, 25.900,   19.0,  97.433, 62.0)    // 15. Cameron

    println (s"xy = $xy")

    banner ("Testing-set indices")
    val permGen = makePermGen (xy.dim)                                // make a permutation generator
    val n_test  = (0.4 * xy.dim).toInt                                // determine the size of the test-set (40%)
    val idx     = testIndices (permGen, n_test)                       // produce the indices for the test-set (RANDOMLY)
    println (s"n_test = $n_test, idx = $idx")

    // Test with combined data-response matrix

    banner ("TnT Split combined data-response matrix")
    val (xy_test, xy_train) = TnT_Split (xy, idx)                     // TnT split the dataset xy (row split)

    banner ("Testing-set")
    println (s"xy_test = $xy_test")

    banner ("Training-set")
    println (s"xy_train = $xy_train")

    // Test with separate data matrix and response vector

    banner ("TnT Split separate data matrix and response vector")
    val (x, y) = (xy.not (?, 4), xy(?, 4))                            // make data matrix and response vector (column split)

    val (x_test, x_train, y_test, y_train) = TnT_Split (x, y, idx)    // TnT split the dataset (x, y) (row split)

    banner ("Testing-set")
    println (s"x_test = $x_test")
    println (s"y_test = $y_test")

    banner ("Training-set")
    println (s"x_train = $x_train")
    println (s"y_train = $y_train")

end tnT_SplitTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tnT_SplitTest2` main function tests the `TnT_Split` object using the Texas
 *  Temperatures dataset.  It is split into a testing-set and a training-set.
 *  Tests when rando = false and picks the FIRST indices for the testing-set.
 *  > runMain scalation.mathstat.tnT_SplitTest2
 */
@main def tnT_SplitTest2 (): Unit =

    // Combined data-response matrix
    // 16 data points:         one      x1      x2       x3     y
    //                                 Lat    Elev     Long  Temp          County
    val xy = MatrixD ((16, 5), 1.0, 29.767,   41.0,  95.367, 56.0,    // 0.  Harris
                               1.0, 32.850,  440.0,  96.850, 48.0,    // 1.  Dallas
                               1.0, 26.933,   25.0,  97.800, 60.0,    // 2.  Kennedy
                               1.0, 31.950, 2851.0, 102.183, 46.0,    // 3.  Midland
                               1.0, 34.800, 3840.0, 102.467, 38.0,    // 4.  Deaf Smith
                               1.0, 33.450, 1461.0,  99.633, 46.0,    // 5.  Knox
                               1.0, 28.700,  815.0, 100.483, 53.0,    // 6.  Maverick
                               1.0, 32.450, 2380.0, 100.533, 46.0,    // 7.  Nolan
                               1.0, 31.800, 3918.0, 106.400, 44.0,    // 8.  El Paso
                               1.0, 34.850, 2040.0, 100.217, 41.0,    // 9.  Collington
                               1.0, 30.867, 3000.0, 102.900, 47.0,    // 10. Pecos
                               1.0, 36.350, 3693.0, 102.083, 36.0,    // 11. Sherman
                               1.0, 30.300,  597.0,  97.700, 52.0,    // 12. Travis
                               1.0, 26.900,  315.0,  99.283, 60.0,    // 13. Zapata
                               1.0, 28.450,  459.0,  99.217, 56.0,    // 14. Lasalle
                               1.0, 25.900,   19.0,  97.433, 62.0)    // 15. Cameron

    println (s"xy = $xy")

    banner ("Testing-set indices")
    val n_test  = (0.4 * xy.dim).toInt                                // determine the size of the test-set (40%)
    val idx     = testIndices (null, n_test, rando = false)           // produce the indices for the test-set (FIRST)
    println (s"n_test = $n_test, idx = $idx")

    // Test with combined data-response matrix

    banner ("TnT Split combined data-response matrix")
    val (xy_test, xy_train) = TnT_Split (xy, idx)                     // TnT split the dataset xy (row split)

    banner ("Testing-set")
    println (s"xy_test = $xy_test")

    banner ("Training-set")
    println (s"xy_train = $xy_train")

    // Test with separate data matrix and response vector

    banner ("TnT Split separate data matrix and response vector")
    val (x, y) = (xy.not (?, 4), xy(?, 4))                            // make data matrix and response vector (column split)

    val (x_test, x_train, y_test, y_train) = TnT_Split (x, y, idx)    // TnT split the dataset (x, y) (row split)

    banner ("Testing-set")
    println (s"x_test = $x_test")
    println (s"y_test = $y_test")

    banner ("Training-set")
    println (s"x_train = $x_train")
    println (s"y_train = $y_train")

end tnT_SplitTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tnT_SplitTest3` main function tests the `TnT_Split` object using the Texas
 *  Temperatures dataset.  It is split into a testing-set and a training-set.
 *  Tests when rando = false and picks the LAST indices for the testing-set.
 *  > runMain scalation.mathstat.tnT_SplitTest3
 */
@main def tnT_SplitTest3 (): Unit =

    // Combined data-response matrix
    // 16 data points:         one      x1      x2       x3     y
    //                                 Lat    Elev     Long  Temp          County
    val xy = MatrixD ((16, 5), 1.0, 29.767,   41.0,  95.367, 56.0,    // 0.  Harris
                               1.0, 32.850,  440.0,  96.850, 48.0,    // 1.  Dallas
                               1.0, 26.933,   25.0,  97.800, 60.0,    // 2.  Kennedy
                               1.0, 31.950, 2851.0, 102.183, 46.0,    // 3.  Midland
                               1.0, 34.800, 3840.0, 102.467, 38.0,    // 4.  Deaf Smith
                               1.0, 33.450, 1461.0,  99.633, 46.0,    // 5.  Knox
                               1.0, 28.700,  815.0, 100.483, 53.0,    // 6.  Maverick
                               1.0, 32.450, 2380.0, 100.533, 46.0,    // 7.  Nolan
                               1.0, 31.800, 3918.0, 106.400, 44.0,    // 8.  El Paso
                               1.0, 34.850, 2040.0, 100.217, 41.0,    // 9.  Collington
                               1.0, 30.867, 3000.0, 102.900, 47.0,    // 10. Pecos
                               1.0, 36.350, 3693.0, 102.083, 36.0,    // 11. Sherman
                               1.0, 30.300,  597.0,  97.700, 52.0,    // 12. Travis
                               1.0, 26.900,  315.0,  99.283, 60.0,    // 13. Zapata
                               1.0, 28.450,  459.0,  99.217, 56.0,    // 14. Lasalle
                               1.0, 25.900,   19.0,  97.433, 62.0)    // 15. Cameron

    println (s"xy = $xy")

    banner ("Testing-set indices")
    val n_test  = (0.4 * xy.dim).toInt                                // determine the size of the test-set (40%)
    val idx     = testIndices (null, xy.dim, n_test, rando = false)   // produce the indices for the test-set (LAST)
    println (s"n_test = $n_test, idx = $idx")

    // Test with combined data-response matrix

    banner ("TnT Split combined data-response matrix")
    val (xy_test, xy_train) = TnT_Split (xy, idx)                     // TnT split the dataset xy (row split)

    banner ("Testing-set")
    println (s"xy_test = $xy_test")

    banner ("Training-set")
    println (s"xy_train = $xy_train")

    // Test with separate data matrix and response vector

    banner ("TnT Split separate data matrix and response vector")
    val (x, y) = (xy.not (?, 4), xy(?, 4))                            // make data matrix and response vector (column split)

    val (x_test, x_train, y_test, y_train) = TnT_Split (x, y, idx)    // TnT split the dataset (x, y) (row split)

    banner ("Testing-set")
    println (s"x_test = $x_test")
    println (s"y_test = $y_test")

    banner ("Training-set")
    println (s"x_train = $x_train")
    println (s"y_train = $y_train")

end tnT_SplitTest3

