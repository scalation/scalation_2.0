
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Wed Oct  7 10:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Time Series Data: Electricity Transformer Temperature, hourly (ETTh1)
 *
 *  ETTh1 is one of the four ETT datasets of Zhou et al. (2021, Informer, AAAI):
 *  two years (2016-07-01 00:00 .. 2018-06-26 19:00, 17420 hourly rows) of one
 *  electricity transformer in a county of China -- six power-load features (high,
 *  middle and low useful/useless load) and the oil temperature (OT), the target of
 *  the long-term forecasting benchmarks (Informer, Autoformer, PatchTST, DLinear,
 *  Time-LLM, ... use ETTh1 with horizons 96..720 on z-scored data).  Here the oil
 *  temperature is forecast h = 1..6 hours ahead from its own lags and the lagged
 *  loads: a physical, lagged response (thermal accumulation and decay) that a
 *  symbolic model may recover.
 *  The file is the original `ETTh1.csv` of the ETDataset repository (date column +
 *  seven numeric columns), unchanged.
 *  @see github.com/zhouhaoyi/ETDataset (ETT-small/ETTh1.csv)
 *  @see arxiv.org/abs/2012.07436 (Informer, Zhou et al. 2021)
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Example_ETTh1` object provides a convenient way to load the ETTh1 hourly data.
 */
object Example_ETTh1:

    import scala.collection.mutable.HashMap

    val fileName = "ETTh1.csv"

    val header = Array ("HUFL",                         // high useful load
                        "HULL",                         // high useless load
                        "MUFL",                         // middle useful load
                        "MULL",                         // middle useless load
                        "LUFL",                         // low useful load
                        "LULL",                         // low useless load
                        "OT")                           // oil temperature (the target)

    val response = "OT"                                 // main response/output variable

    /** The six load columns (the candidate drivers) and their short formula names.
     */
    val LOADS      = Array ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL")
    val LOAD_NAMES = Array ("hufl", "hull", "mufl", "mull", "lufl", "lull")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load the ETTh1 hourly data into a matrix for the exogenous variables x
     *  and a vector for the response/endogenous variable y.
     *  @param x_strs  the column names for the exogenous variables x
     *  @param y_str   the column name for the endogenous variable y
     *  @param rng     the rows to keep (null => all 17420 rows)
     */
    def loadData (x_strs: Array [String], y_str: String, rng: Range = null): (MatrixD, VectorD) =
        val col = HashMap [String, Int] ()
        for i <- header.indices do col += header(i) -> i

        val all  = MatrixD.load (fileName, 1, 1)           // skip first row (header) + trim first column (date)
        val data = if rng == null then all else all(rng)
        val x_cols = for s <- x_strs yield col(s)
        (data(?, x_cols), data(?, col(y_str)))
    end loadData

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load the ETTh1 hourly data into a vector for the response/endogenous variable y.
     *  @param y_str  the column name for the endogenous variable y
     *  @param rng    the rows to keep (null => all rows)
     */
    def loadData_y (y_str: String = response, rng: Range = null): VectorD =
        val col = HashMap [String, Int] ()
        for i <- header.indices do col += header(i) -> i

        val all = MatrixD.load (fileName, 1, 1)
        val y   = all(?, col(y_str))
        if rng == null then y else y(rng)
    end loadData_y

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the named-column dataset map used by the formula layer (`ARX_Formula`,
     *  `ARX_D.fromTerms`, a formula search built on them): the response under
     *  the key "y" and each exogenous column under its short name.  Usage:
     *  given dataset: Map [String, VectorD] = context (LOADS, LOAD_NAMES, rng = 0 until 8760).
     *  @param x_strs  the column names for the exogenous variables x
     *  @param names   the short (formula) names for them (defaults to x_strs)
     *  @param y_str   the column name for the endogenous variable y
     *  @param rng     the rows to keep (null => all rows)
     */
    def context (x_strs: Array [String] = LOADS, names: Array [String] = LOAD_NAMES,
                 y_str: String = response, rng: Range = null): Map [String, VectorD] =
        val nms = if names == null then x_strs else names
        require (nms.length == x_strs.length, "context: one short name per exogenous column")
        val (xe, y) = loadData (x_strs, y_str, rng)
        val exo = for j <- x_strs.indices yield nms(j) -> xe(?, j)
        Map ("y" -> y) ++ exo
    end context

end Example_ETTh1

import Example_ETTh1._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_ETTh1Test` main function tests the `Example_ETTh1` object:
 *  loads the file, checks its shape and plots the oil temperature.
 *  > runMain scalation.modeling.forecasting.example_ETTh1Test
 */
@main def example_ETTh1Test (): Unit =

    val y = loadData_y ()
    banner (s"Plot the response = $response column for the ETTh1 dataset (${y.dim} points)")
    println (s"OT: min ${y.min}, max ${y.max}, mean ${y.mean}")
    assert (y.dim == 17420, s"ETTh1 should have 17420 rows, found ${y.dim}")
    new Plot (null, y, null, s"y ($response)", lines = true)

    val data = context (rng = 0 until 8760)
    println (s"first year: ${data.keys.toSeq.sorted.mkString (", ")}, ${data("y").dim} rows")
    assert (data.size == 7 && data("y").dim == 8760, "context: y + six loads over the first year")
    println ("example_ETTh1Test: passed")

end example_ETTh1Test

