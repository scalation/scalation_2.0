
package scalation
package theory

import scala.math._
import scalation.mathstat.VectorD
 
// runMain scalation.theory.testExpr

@main def testExpr (): Unit =

    val y = new VectorD (10)                                      // endogenous time series
    val (x0, x1) = (new VectorD (10), new VectorD (10))           // two exogenous time series
    val c = VectorD (1, 2, 3, 4)                                  // parameters
    val t = 5                                                     // time

    y(t) = c(0) + c(1) * y(t-1) + c(2) * sin(y(t-2)) + c(3) * x0(t-1) * x1(t-1)~^2

    println (s"y = $y")

end testExpr

