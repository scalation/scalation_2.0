
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb  7 21:26:42 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Time Series Data: gas_furnace (9-minute) Data
 *
 *  @see openmv.net/info/gas-furnace
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Example_GasFurnace` object provides a convenient way to load gas_furnace data.
 */
object Example_GasFurnace:

    import scala.collection.mutable.HashMap

    val header = Array ("input_gas_rate",
                        "co2")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load the gas_furnace data into a matrix for the exogenous variables x
     *  and a vector for the response/endogenous variable y.
     *  @param x_strs  the column names for the exogenous variables x
     *  @param y_str   the column name for the endogenous variable y
     *  @param trim    the number of initial rows to trim away (e.g., they are all 0)
     */
    def loadData (x_strs: Array [String], y_str: String, trim: Int = 0): (MatrixD, VectorD) =
        val col = HashMap [String, Int] ()
        for i <- header.indices do col += header(i) -> i

        val data = MatrixD.load ("gas_furnace.csv", 1+trim, 0)      // skip first row (header) + trim no columns
        val x_cols = for s <- x_strs yield col(s)
        (data(?, x_cols), data(?, col(y_str)))
    end loadData

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load the gas_furnace data into a vector for the response/endogenous variable y.
     *  @param y_str  the column name for the endogenous variable y
     *  @param trim   the number of initial rows to trim away (e.g., they are all 0)
     */
    def loadData_y (y_str: String, trim: Int = 0): VectorD =
        val col = HashMap [String, Int] ()
        for i <- header.indices do col += header(i) -> i

        val data = MatrixD.load ("gas_furnace", 1+trim, 0)          // skip first row (header) + trim no columns
        data(?, col(y_str))
    end loadData_y

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load the gas_furnace data into a matrix for the variables y.
     *  @param y_str  the column names for the variables y (e.g., used in a VAR model)
     *  @param trim   the number of initial rows to trim away (e.g., they are all 0)
     */
    def loadData_yy (y_strs: Array [String], trim: Int = 0): MatrixD =
        val col = HashMap [String, Int] ()
        for i <- header.indices do col += header(i) -> i

        val data = MatrixD.load ("gas_furnace.csv", 1+trim, 0)      // skip first row (header) + trim no columns
        val y_cols = for s <- y_strs yield col(s)
        data(?, y_cols)
    end loadData_yy

end Example_GasFurnace


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_GasFurnaceTest` main function test the `Example_GasFurnace` object.
 *  > runMain scalation.modeling.forecasting.example_GasFurnaceTest
 */
@main def example_GasFurnaceTest (): Unit =

    import Example_GasFurnace._

    val y = loadData_yy (header)

    new Plot (null, y.col(0), null, header(0), lines = true)
    new Plot (null, y.col(1), null, header(1), lines = true)

end example_GasFurnaceTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_GasFurnaceTest2` main function test the `Example_GasFurnace` object.
 *  This performs Exploratory Data Analysis (EDA) to find relationships
 *  between contemporaneous variables.
 *  > runMain scalation.modeling.forecasting.example_GasFurnaceTest2
 */
@main def example_GasFurnaceTest2 (): Unit =

    import Example_GasFurnace._
    import scala.collection.mutable.{LinkedHashSet => LSET}

    val (x, y) = loadData (header, "co2")

    new Plot (null, y, null, "co2", lines = true)

    for j <- x.indices2 do
        banner (s"EDA for co2 vs. ${header(j)}")
        var xj  = x(?, j)                                               // get column j
        xj = scaleV (extreme (xj), (0.0, 2.0))(xj)                      // rescale vector xj to [0, 2]
        val xxj = MatrixD.fromVector (xj)
//      val mod = SymbolicRegression.quadratic (xxj, y)
//      val mod = SymbolicRegression.rescale (xxj, y, null, LSET (1.0, 2.0, 3.0), cross = false)
        val mod = SymbolicRegression (xxj, y, null, LSET (0.5, 1.0, 2.0, 3.0), cross = false)
        mod.trainNtest ()()
        val yp = mod.predict (mod.getX)
        println (mod.summary ())
        new Plot (xj, y, yp, s"y, yp (co2) vs. x_$j (${header(j)})")
    end for

end example_GasFurnaceTest2

