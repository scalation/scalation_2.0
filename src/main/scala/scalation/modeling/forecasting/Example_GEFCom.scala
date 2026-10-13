
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Wed Oct  7 10:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Time Series Data: GEFCom2014 Electric Load (US utility), daily
 *
 *  The load track of the Global Energy Forecasting Competition 2014 (Hong et al.
 *  2016, IJF) gives the hourly load of a US utility from 2005-01-01 to 2010-09-30
 *  with the hourly temperature of 25 weather stations (file `L1-train.csv`, Load
 *  Task 1, which also carries the temperatures back to 2001).  The daily file
 *  `gefcom2014_load_daily.csv` (2099 days, built by tools/build_gefcom_daily.py)
 *  aggregates the hour-ending 1..24 of every day:
 *      peak     the daily peak load (max of the 24 hourly loads)
 *      avg      the daily mean load
 *      temp     the daily mean of the 25-station mean temperature (deg F)
 *      tmax     the daily max of the 25-station mean temperature
 *      tmin     the daily min of the 25-station mean temperature
 *      weekend  1 on Saturday and Sunday, else 0
 *  The daily peak load is forecast h = 1..6 days ahead from its own lags and the
 *  lagged temperatures (no weather forecasts).  The load-temperature relation is
 *  V-shaped (heating below ~55 F, cooling above ~65 F; the daily peak is lowest
 *  for mean temperatures of 50-70 F), the kind of threshold response a `hinge`
 *  term with a free parameter may recover.
 *  @see www.sciencedirect.com/science/article/pii/S0169207016000133 (Hong et al. 2016)
 *  @see blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Example_GEFCom` object provides a convenient way to load the daily
 *  GEFCom2014 load data.
 */
object Example_GEFCom:

    import scala.collection.mutable.HashMap

    val fileName = "gefcom2014_load_daily.csv"

    val header = Array ("peak",                         // daily peak load (the target)
                        "avg",                          // daily mean load
                        "temp",                         // daily mean temperature (deg F)
                        "tmax",                         // daily maximum temperature
                        "tmin",                         // daily minimum temperature
                        "weekend")                      // 1 => Saturday or Sunday

    val response = "peak"                               // main response/output variable

    /** The temperature columns (the candidate drivers; `avg` is another measure of the response).
     */
    val TEMPS = Array ("temp", "tmax", "tmin")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load the daily GEFCom2014 data into a matrix for the exogenous variables x
     *  and a vector for the response/endogenous variable y.
     *  @param x_strs  the column names for the exogenous variables x
     *  @param y_str   the column name for the endogenous variable y
     *  @param rng     the rows to keep (null => all 2099 days)
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
    /** Load the daily GEFCom2014 data into a vector for the response/endogenous variable y.
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
     *  given dataset: Map [String, VectorD] = context (Array ("temp", "tmax")).
     *  @param x_strs  the column names for the exogenous variables x
     *  @param names   the short (formula) names for them (defaults to x_strs)
     *  @param y_str   the column name for the endogenous variable y
     *  @param rng     the rows to keep (null => all rows)
     */
    def context (x_strs: Array [String] = TEMPS, names: Array [String] = null,
                 y_str: String = response, rng: Range = null): Map [String, VectorD] =
        val nms = if names == null then x_strs else names
        require (nms.length == x_strs.length, "context: one short name per exogenous column")
        val (xe, y) = loadData (x_strs, y_str, rng)
        val exo = for j <- x_strs.indices yield nms(j) -> xe(?, j)
        Map ("y" -> y) ++ exo
    end context

end Example_GEFCom

import Example_GEFCom._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_GEFComTest` main function tests the `Example_GEFCom` object:
 *  loads the file, checks its shape, prints the mean peak load per 10-degree
 *  temperature band (the V shape) and plots the peak load.
 *  > runMain scalation.modeling.forecasting.example_GEFComTest
 */
@main def example_GEFComTest (): Unit =

    val (xe, y) = loadData (Array ("temp"), response)
    banner (s"Plot the response = $response column for the daily GEFCom2014 load dataset (${y.dim} points)")
    println (s"peak: min ${y.min}, max ${y.max}, mean ${y.mean}")
    assert (y.dim == 2099, s"the daily file should have 2099 rows, found ${y.dim}")

    banner ("mean peak load per 10-degree band of the daily mean temperature")
    val temp = xe(?, 0)
    for lo <- 20 to 80 by 10 do
        val idx = (0 until y.dim).filter (i => temp(i) >= lo && temp(i) < lo + 10)
        if idx.nonEmpty then println (f"$lo%3d-${lo+10}%3d F: ${idx.map (y(_)).sum / idx.size}%7.1f  (${idx.size} days)")
    new Plot (null, y, null, s"y ($response)", lines = true)

    val data = context (Array ("temp", "tmax"))
    assert (data.size == 3 && data("y").dim == 2099, "context: y + two temperatures")
    println ("example_GEFComTest: passed")

end example_GEFComTest

