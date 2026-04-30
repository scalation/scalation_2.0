
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira
 *  @version 2.0
 *  @note    Mon Jan 22 15:21:01 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Functions for Testing Optimizers
 */

package scalation
package optimization
package functions

import scala.math.{cos, Pi, sin}

import scalation.mathstat.VectorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BealeFunction` object to represent the Beale function for tests and benchmarks
 *  performed on function optimization and gradient descent classes.
 */
object BealeFunction extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (3, 0.5)

    def objFunction (x: VectorD): Double =
        (1.5 - x(0) + x(0) * x(1)) ~^ 2 + (2.25 - x(0) + x(0) * (x(1) ~^ 2)) ~^ 2 + (2.625 - x(0) + x(0) * (x(1) ~^ 3)) ~^ 2

    override def gradFunction (x: VectorD): VectorD =
        VectorD (2 * (1.5 - x(0) + x(0) * x(1)) * (-1 + x(1)) +
                 2 * (2.25 - x(0) + x(0) * (x(1) ~^ 2)) * (-1 + (x(1) ~^ 2)) +
                 2 * (2.625 - x(0) + x(0) * (x(1) ~^ 3)) * (-1 + (x(1) ~^ 3)),
                 2 * (1.5 - x(0) + x(0) * x(1)) * x(0) + 
                 2 * (2.25 - x(0) + x(0) * (x(1) ~^ 2)) * (2 * x(0) * x(1)) +
                 2 * (2.625 - x(0) + x(0) * (x(1) ~^ 3)) * (3 * x(0) * (x(1) ~^ 2)))

end BealeFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Bohachevsky1Function` object to represent the Bohachevsky1 function for tests
 *  and benchmarks performed on function optimization and gradient descent classes.
 */
object Bohachevsky1Function extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (0, 0)

    def objFunction(x: VectorD): Double =
        x(0) ~^ 2 + 2 * x(1) ~^ 2 - 0.3 * cos(3 * Pi * x(0)) - 0.4 * cos(4 * Pi * x(1)) + 0.7

    override def gradFunction (x: VectorD): VectorD =
        VectorD (2 * x(0) - 0.3 * 3 * Pi * sin(3 * Pi * x(0)),
                 4 * x(1) - 0.4 * 4 * Pi * sin(4 * Pi * x(1)))

end Bohachevsky1Function


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Bohachevsky2Function` object to represent the Bohachevsky2 function for tests
 *  and benchmarks performed on function optimization and gradient descent classes.
 */
object Bohachevsky2Function extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (0, 0)

    def objFunction (x: VectorD): Double =
        x(0) ~^ 2 + 2 * x(1) ~^ 2 - 0.3 * cos(3 * Pi * x(0)) * cos(4 * Pi * x(1)) + 0.3

    override def gradFunction(x: VectorD): VectorD =
        VectorD (2 * x(0) + 0.3 * 3 * Pi * sin(3 * Pi * x(0)) * cos(4 * Pi * x(1)),
                 4 * x(1) - 0.3 * 4 * Pi * cos(3 * Pi * x(0)) * sin(4 * Pi * x(1)))

end Bohachevsky2Function


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Bohachevsky3Function` object to represent the Bohachevsky3 function for tests
 *  and benchmarks performed on function optimization and gradient descent classes.
 */
object Bohachevsky3Function extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (0, 0)

    def objFunction (x: VectorD): Double =
        x(0) ~^ 2 + 2 * x(1) ~^ 2 - 0.3 * cos(3 * Pi * x(0) + 4 * Pi * x(1)) + 0.3

    override def gradFunction(x: VectorD): VectorD = VectorD(
        2 * x(0) + 0.3 * 3 * Pi * sin(3 * Pi * x(0) + 4 * Pi * x(1)),
        4 * x(1) + 0.3 * 4 * Pi * sin(3 * Pi * x(0) + 4 * Pi * x(1)))

end Bohachevsky3Function


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BoothFunction` object to represent the Booth function for tests and benchmarks
 *  performed on function optimization and gradient descent classes.
 */
object BoothFunction extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (1, 3)

    def objFunction (x: VectorD): Double =
        (x(0) + 2 * x(1) - 7) ~^ 2 + (2 * x(0) + x(1) - 5) ~^ 2

    override def gradFunction (x: VectorD): VectorD =
        VectorD (10 * x(0) + 8 * x(1) - 34, 8 * x(0) + 10 * x(1) - 38)

end BoothFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The ``BoothFunction` object to represent the Camel3 function for tests and benchmarks
 *  performed on function optimization and gradient descent classes.
 */
object Camel3Function extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (0, 0)

    def objFunction (x: VectorD): Double =
        2 * x(0) ~^ 2 - 1.05 * x(0) ~^ 4 + (1 / 6.0) * x(0) ~^ 6 + x(0) * x(1) + x(1) ~^ 2

    override def gradFunction (x: VectorD): VectorD =
        VectorD (4 * x(0) - 4.2 * x(0) ~^ 3 + x(0) ~^ 5 + x(1), x(0) + 2 * x(1))

end Camel3Function


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubeFunction` object to represent the Cube function for tests and benchmarks
 *  performed on function optimization and gradient descent classes.
 */
object CubeFunction extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (-1, 1)

    def objFunction(x: VectorD): Double =
        100 * (x(1) - x(0) ~^ 3) ~^ 2 + (1 - x(0)) ~^ 2

    override def gradFunction(x: VectorD): VectorD =
        VectorD (-200 * (x(1) - x(0) ~^ 3) * (3 * x(0) ~^ 2) - 2 * (1 - x(0)), 200 * (x(1) - x(0) ~^ 3))

end CubeFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*  The `FreudensteinRothFunction` object to represent the Freudenstein-Roth function
 *  for tests and benchmarks performed on function optimization and gradient descent classes.
 */
object FreudensteinRothFunction extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (5, 4)

    def objFunction (x: VectorD): Double =
        (x(0) - 13 + x(1) * ((5 - x(1)) * x(1) - 2)) ~^ 2 + (x(0) - 29 + x(1) * ((x(1) + 1) * x(1) - 14)) ~^ 2

    override def gradFunction (x: VectorD): VectorD =
        VectorD (2 * (x(0) - 13 + x(1) * ((5 - x(1)) * x(1) - 2)) + 2 * (x(0) - 29 + x(1) * ((x(1) + 1) * x(1) - 14)),
                 2 * x(1) * ((5 - x(1)) * x(1) - 2) + 2 * (x(1) * ((x(1) + 1) * x(1) - 14) +
                    (x(0) - 13 + x(1) * ((5 - x(1)) * x(1) - 2)) * ((5 - x(1)) * x(1) - 2)))

end FreudensteinRothFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `McCormickFunction` object to represent the McCormick function for tests and
 *  benchmarks performed on function optimization and gradient descent classes.
 */
object McCormickFunction extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (-0.54719, -1.54719)

    def objFunction (x: VectorD): Double = 
        sin(x(0) + x(1)) + (x(0) - x(1)) ~^ 2 - 1.5 * x(0) + 2.5 * x(1) + 1

    override def gradFunction (x: VectorD): VectorD =
        VectorD (-1.5 + 2 * x(0) - 2 * x(1) + cos(x(0) + x(1)),
                  2.5 - 2 * x(0) + 2 * x(1) + cos(x(0) + x(1)))

end McCormickFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ParaboloidFunction` object to represent an example of a Paraboloid function for
 *  tests and  benchmarks performed on function optimization and gradient descent classes.
 */
object ParaboloidFunction extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (3, 4)

    def objFunction (x: VectorD): Double =
        (x(0) - 3.0) ~^ 2 + (x(1) - 4.0) ~^ 2 + 1.0

    override def gradFunction (x: VectorD): VectorD =
        VectorD (2 * x(0) - 6, 2 * x(1) - 8)

end ParaboloidFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuarticFunction` object to represent an example of a Quartic function for tests
 *  and benchmarks performed on function optimization and gradient descent classes.
 */
object QuarticFunction extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (1, 4)

    def objFunction (x: VectorD): Double =
        x(0) ~^ 4 + (x(0) - 3.0) ~^ 2 + (x(1) - 4.0) ~^ 2 + 1.0

    override def gradFunction (x: VectorD): VectorD =
        VectorD (4.0 * x(0) ~^ 3 + 2 * x(0) - 6, 2 * x(1) - 8)

end QuarticFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ReciprocalFunction` object to represent an example of a Reciprocal function for
 *  tests and benchmarks performed on function optimization and gradient descent classes.
 */
object ReciprocalFunction extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (1.06035, 4)

    override val bound: (VectorD, VectorD) = (VectorD (0.2, 0.2), VectorD (10.0, 10.0))

    def objFunction (x: VectorD): Double =
        1 / x(0) + x(0) ~^ 4 + (x(0) - 3.0) ~^ 2 + (x(1) - 4.0) ~^ 2 + 1.0

    override def gradFunction (x: VectorD): VectorD =
        VectorD (-(x(0) ~^ (-2)) + 4.0 * x(0) ~^ 3 + 2 * x(0) - 6, 2 * x(1) - 8)

end ReciprocalFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RosenbrockFunction` object to represent the Rosenbrock function for tests and
 *  benchmarks performed on function optimization and gradient descent classes.
 */
object RosenbrockFunction extends BenchmarkFunction:

    val functionMinimum: VectorD = VectorD (1, 1)

    def objFunction (x: VectorD): Double =
        (1.0 - x(0)) ~^ 2 + 100.0 * (x(1) - x(0) ~^ 2) ~^ 2

    override def gradFunction(x: VectorD): VectorD =
        VectorD (-2.0 * (1 - x(0)) - 400.0 * x(0) * (x(1) - x(0) ~^ 2),
                 200.0 * (x(1) - x(0) ~^ 2))

end RosenbrockFunction

